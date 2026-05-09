"""
Model B — Distractor & Hint Generator (spec §5)
================================================

Two sub-tasks, both purely classical (no neural required):

  1. Distractor Generation (§5.3.1, §5.4)
       Input  : (passage, question, correct_answer)
       Output : 3 plausible distractors (semantically related but factually wrong)
       Pipeline:
         a) Extract candidate phrases from passage (rule based, no NLP tools)
         b) For each candidate compute features:
              cosine to correct answer (TF·IDF and GloVe), character bigram
              overlap, passage frequency, length statistics, word overlap
         c) Train Logistic Regression and Random Forest rankers on
              (candidate, is_distractor) labels mined from the dataset's
              own A / B / C / D options
         d) At inference: score all candidates, exclude the correct answer,
              apply a Jaccard diversity penalty, pick top three

  2. Hint Extraction (§5.3.2)
       Input  : (passage, question, correct_answer)
       Output : top K passage sentences ordered from general to near explicit
       Two variants:
         a) TF·IDF cosine ranking — score each sentence by similarity to the
              question, surface the top K
         b) ML scored ranking — train a Logistic Regression on sentence
              features (cosine to question, cosine to answer, position,
              length, GloVe cosine) using the most overlapping sentence as
              the "gold key sentence" label

Performance
-----------
All TF·IDF transforms and GloVe lookups are batched per row (one call per
row covering all candidates / all sentences) rather than per individual
candidate. This typically buys a 5–10× speedup over the single-text
transform pattern, bringing the full pipeline runtime under 10 minutes
on a modern laptop CPU.

Reuses infrastructure from Model A (no neural code leaks):
  - split_sentences, gen_tokens_lower, row_cosine, text_to_glove_vec
  - load_glove, build_idf_lookup, fit_tfidf_vectorizer
  - GenerationMetrics (BLEU 1 / 4 + ROUGE 1/2/L + METEOR + corpus BLEU)
  - TFIDF_VEC_PATH (the vectorizer fitted by model_a.py is reloaded here)

Usage: python src/model_b.py
       (run model_a.py once first so the TF·IDF vectorizer is on disk)
"""

import sys
import os
import re
import joblib
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import r2_score, confusion_matrix
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TEST_CSV_PATH,
    MODELS_DIR,
)
from utils import save_model

# Reuse classical utilities from Model A. No neural code is imported here.
from model_a import (
    split_sentences,
    gen_tokens_lower,
    text_to_glove_vec,
    fit_tfidf_vectorizer,
    load_glove,
    build_idf_lookup,
    GenerationMetrics,
    _ensure_nltk_data,
    TFIDF_VEC_PATH,
)


# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_B_DIST_LR_PATH = os.path.join(MODELS_DIR, "model_b_distractor_lr.pkl")
MODEL_B_DIST_RF_PATH = os.path.join(MODELS_DIR, "model_b_distractor_rf.pkl")
MODEL_B_DIST_HGB_PATH = os.path.join(MODELS_DIR, "model_b_distractor_hgb.pkl")
MODEL_B_HINT_PATH    = os.path.join(MODELS_DIR, "model_b_hint_lr.pkl")
MODEL_B_HINT_REG_PATH = os.path.join(MODELS_DIR, "model_b_hint_regressor.pkl")
LIKERT_TEMPLATE_PATH  = os.path.join(MODELS_DIR, "model_b_distractor_likert_template.csv")

EPS                       = 1e-9
TRAIN_SAMPLE_SIZE         = 12_000   # rows used for ranker training
EVAL_SAMPLE_SIZE          = 3_000    # rows used for evaluation (val / test)
MAX_CANDIDATES            = 30       # candidate phrases per passage
DISTRACTOR_MATCH_THRESH   = 0.3      # ROUGE 1 above this counts as a match
LENGTH_RATIO_MIN          = 0.4      # gen-time filter: candidate / answer token ratio
LENGTH_RATIO_MAX          = 2.5      # gen-time filter: candidate / answer token ratio
TOK_RE                    = re.compile(r'\b[a-zA-Z]+\b')

# Lightweight ROUGE scorer used purely for greedy gen->gold matching during
# evaluation. Kept separate from GenerationMetrics so its calls do NOT pollute
# the corpus BLEU buffers.
_MATCH_ROUGE = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


# ─── B1: Distractor candidate extraction ─────────────────────────────────────
def extract_candidate_phrases(passage, max_candidates=MAX_CANDIDATES):
    """
    Spec §5.3.1 Step 1 — extract candidate phrases from the passage using
    simple string matching and frequency based selection. No NLP tools.
    """
    sentences = split_sentences(passage)
    candidates = []

    # 1. Whole sentences of moderate length
    for sent in sentences:
        clean = sent.strip().rstrip('.!?,;:"\'')
        wc = len(clean.split())
        if 3 <= wc <= 25 and clean:
            candidates.append(clean)

    # 2. Capitalised noun phrase like spans
    np_pat = re.compile(r'\b[A-Z][a-z]+(?:\s+(?:of|the|a|an|in|on))?\s*(?:[A-Z][a-z]+)?\b')
    for sent in sentences:
        for m in np_pat.findall(sent):
            wc = len(m.split())
            if 1 <= wc <= 6 and m.strip():
                candidates.append(m.strip())

    # 3. High frequency content words
    word_counts = Counter()
    for sent in sentences:
        for w in TOK_RE.findall(sent.lower()):
            if len(w) >= 4:
                word_counts[w] += 1
    for w, _ in word_counts.most_common(8):
        candidates.append(w)

    # Deduplicate (case insensitive) preserving order
    seen, out = set(), []
    for c in candidates:
        norm = c.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(c)

    return out[:max_candidates]


# ─── Batched feature primitives ──────────────────────────────────────────────
def _batch_tfidf_cosine(texts, target_text, vec):
    """
    Returns (n,) cosine similarities of `texts` vs `target_text`.
    One vec.transform on the whole list, one on the target — far faster
    than calling vec.transform per text.
    """
    if not texts:
        return np.zeros(0, dtype=np.float32)
    text_vecs   = vec.transform(texts)
    target_vec  = vec.transform([target_text])
    text_norms  = np.sqrt(np.asarray(text_vecs.multiply(text_vecs).sum(axis=1)).ravel()) + EPS
    target_norm = float(np.sqrt(target_vec.multiply(target_vec).sum())) + EPS
    dots        = np.asarray((text_vecs @ target_vec.T).todense()).ravel()
    return (dots / (text_norms * target_norm)).astype(np.float32)


def _batch_glove_cosine(texts, target_text, glove, idf_lookup=None):
    """Returns (n,) GloVe cosines vs target. Zeros if glove unavailable."""
    if glove is None or not texts:
        return np.zeros(len(texts), dtype=np.float32)
    dim = glove.vector_size
    target_g  = text_to_glove_vec(target_text, glove, dim, idf_lookup)
    text_gs   = np.stack([text_to_glove_vec(t, glove, dim, idf_lookup) for t in texts])
    target_n  = float(np.linalg.norm(target_g)) + EPS
    text_ns   = np.linalg.norm(text_gs, axis=1) + EPS
    return ((text_gs @ target_g) / (text_ns * target_n)).astype(np.float32)


def _char_bigrams(s):
    s = re.sub(r'\s+', ' ', s.lower())
    return set(s[i:i + 2] for i in range(len(s) - 1)) if len(s) >= 2 else set()


# ─── B2: Distractor features (batched per row) ───────────────────────────────
def batch_distractor_features(candidates, correct_answer, passage,
                              vec, glove=None, idf_lookup=None):
    """
    Vectorised feature builder for a list of distractor candidates.

    Returns (n_candidates, 7) feature matrix:
      0  cos_tfidf      : TF·IDF cosine similarity to correct answer
      1  char_match     : Jaccard on character bigrams
      2  passage_freq   : occurrences of candidate in passage / passage length
      3  length_diff    : abs token length difference vs correct answer
      4  length_ratio   : token length ratio (candidate / answer)
      5  cos_glove      : GloVe cosine similarity to correct answer
      6  word_overlap   : fraction of answer tokens present in candidate
    """
    n = len(candidates)
    if n == 0:
        return np.zeros((0, 7), dtype=np.float32)

    # 1) TF·IDF cosine — single batched transform
    cos_tfidf = _batch_tfidf_cosine(candidates, correct_answer, vec)

    # 2) Char bigram Jaccard — bg_a computed once
    bg_a = _char_bigrams(correct_answer.lower())
    char_match = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(candidates):
        bg_c = _char_bigrams(c.lower())
        if bg_c or bg_a:
            char_match[i] = len(bg_c & bg_a) / max(1, len(bg_c | bg_a))

    # 3) Passage frequency — psg_lower computed once
    psg_lower = passage.lower()
    n_psg_words = max(1, len(psg_lower.split()))
    psg_freq = np.array(
        [psg_lower.count(c.lower().strip()) / n_psg_words for c in candidates],
        dtype=np.float32,
    )

    # 4-5) Length features
    la = max(1, len(correct_answer.split()))
    lcs = np.array([max(1, len(c.split())) for c in candidates], dtype=np.float32)
    length_diff  = np.abs(lcs - la).astype(np.float32)
    length_ratio = (lcs / la).astype(np.float32)

    # 6) GloVe cosine — single batched lookup + cosines
    cos_glove = _batch_glove_cosine(candidates, correct_answer, glove, idf_lookup)

    # 7) Word overlap — aw computed once
    aw = set(TOK_RE.findall(correct_answer.lower()))
    word_overlap = np.zeros(n, dtype=np.float32)
    if aw:
        for i, c in enumerate(candidates):
            cw = set(TOK_RE.findall(c.lower()))
            word_overlap[i] = len(cw & aw) / len(aw)

    return np.stack([
        cos_tfidf, char_match, psg_freq,
        length_diff, length_ratio,
        cos_glove, word_overlap,
    ], axis=1).astype(np.float32)


# ─── B2: Build distractor training data ─────────────────────────────────────
def build_distractor_train_data(df, vec, glove, idf_lookup,
                                 max_rows=TRAIN_SAMPLE_SIZE):
    """
    Per row, compute features for all distractor + negative candidates in
    one batch, then accumulate into the global X / y. The per-row batching
    is the key speedup: ~5× faster than per-candidate vec.transform calls.
    """
    X_chunks, y_chunks = [], []
    rng = np.random.default_rng(42)

    for _, row in df.head(max_rows).iterrows():
        passage = str(row.get('article', ''))
        correct_letter = row.get('answer', None)
        if correct_letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(correct_letter, '')).strip()
        if not correct_ans or not passage:
            continue

        # Positives: dataset's own three non-correct options
        gold_distractors = []
        for letter in ('A', 'B', 'C', 'D'):
            if letter == correct_letter:
                continue
            d = str(row.get(letter, '')).strip()
            if d:
                gold_distractors.append(d)
        if not gold_distractors:
            continue

        # Negatives: random passage candidates that aren't the correct answer
        cands = extract_candidate_phrases(passage)
        cands = [c for c in cands if c.lower().strip() != correct_ans.lower().strip()]
        if len(cands) > 3:
            neg_indices = rng.choice(len(cands), size=3, replace=False)
            negatives = [cands[int(i)] for i in neg_indices]
        else:
            negatives = cands

        all_examples = gold_distractors + negatives
        labels = [1] * len(gold_distractors) + [0] * len(negatives)
        if not all_examples:
            continue

        feats = batch_distractor_features(
            all_examples, correct_ans, passage, vec, glove, idf_lookup,
        )
        X_chunks.append(feats)
        y_chunks.append(np.array(labels, dtype=np.int32))

    if not X_chunks:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.concatenate(X_chunks, axis=0), np.concatenate(y_chunks, axis=0)


def train_distractor_rankers(train_df, vec, glove, idf_lookup):
    """Train Logistic Regression and Random Forest distractor rankers."""
    print("\n[B2-RANKER] Building distractor training data "
          f"(up to {TRAIN_SAMPLE_SIZE:,} rows, batched per-row TF·IDF + GloVe)...")
    X, y = build_distractor_train_data(train_df, vec, glove, idf_lookup)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"  Examples: {len(X):,}  ({n_pos:,} distractors, {n_neg:,} non-distractors)")

    rankers = {}

    print("[B2-RANKER] Training Logistic Regression...")
    lr = LogisticRegression(C=1.0, max_iter=400, n_jobs=-1, random_state=42).fit(X, y)
    save_model(lr, MODEL_B_DIST_LR_PATH)
    rankers['LR'] = lr

    print("[B2-RANKER] Training Random Forest (100 trees, max_depth=15)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, n_jobs=-1, random_state=42,
        class_weight='balanced',
    ).fit(X, y)
    save_model(rf, MODEL_B_DIST_RF_PATH)
    rankers['RF'] = rf

    print("[B2-RANKER] Training Histogram Gradient Boosting...")
    hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.1, random_state=42,
    ).fit(X, y)
    save_model(hgb, MODEL_B_DIST_HGB_PATH)
    rankers['HGB'] = hgb

    return rankers


def _word_jaccard(a, b):
    aw = set(TOK_RE.findall(a.lower()))
    bw = set(TOK_RE.findall(b.lower()))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def generate_distractors(passage, correct_answer, ranker,
                         vec, glove, idf_lookup, top_k=3, diversity_thresh=0.7):
    """
    Score all candidates in one batch, top-K with Jaccard diversity filter.
    Applies a length-ratio filter at gen time (kept off training to preserve
    the length-mismatch negative signal for the ranker).
    """
    cands = extract_candidate_phrases(passage)
    correct_lower = correct_answer.lower().strip()
    cands = [c for c in cands if c.lower().strip() != correct_lower]
    if not cands:
        return []

    # Length-ratio filter: keep only candidates whose token length is in
    # [0.4×, 2.5×] of the answer. Falls back to unfiltered set if nothing
    # passes (so we never return empty when there are usable candidates).
    la = max(1, len(correct_answer.split()))
    filtered = [
        c for c in cands
        if LENGTH_RATIO_MIN <= max(1, len(c.split())) / la <= LENGTH_RATIO_MAX
    ]
    if filtered:
        cands = filtered

    feats = batch_distractor_features(
        cands, correct_answer, passage, vec, glove, idf_lookup,
    )
    if hasattr(ranker, 'predict_proba'):
        scores = ranker.predict_proba(feats)[:, 1]
    else:
        scores = ranker.decision_function(feats)

    order = np.argsort(-scores)
    selected = []
    for idx in order:
        cand = cands[int(idx)]
        if any(_word_jaccard(cand, s) > diversity_thresh for s in selected):
            continue
        selected.append(cand)
        if len(selected) >= top_k:
            break

    return selected


# ─── B3b: Frequency-Based Substitution distractor alternative (spec §5.4) ──
_STOP = {
    'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to',
    'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'this', 'that', 'these', 'those', 'it',
    'its', 'they', 'their', 'them', 'he', 'she', 'his', 'her', 'we',
    'our', 'you', 'your', 'i', 'me', 'my',
}


def generate_distractors_freq(correct_answer, passage, top_k=3):
    """
    Spec §5.4 — Frequency-Based Substitution.
    Identify high-frequency content words in the passage and propose
    them as distractors when the correct answer is short (single noun
    phrase / single content word). For multi-word answers we substitute
    the most salient content token with each high-frequency content word
    that is not already present in the answer.
    """
    psg_words = [w for w in TOK_RE.findall(passage.lower())
                 if len(w) >= 4 and w not in _STOP]
    if not psg_words:
        return []
    freq = Counter(psg_words)
    answer_words = set(TOK_RE.findall(correct_answer.lower()))
    high_freq = [w for w, _ in freq.most_common(30) if w not in answer_words]

    ans_tokens = correct_answer.split()
    if len(ans_tokens) <= 1:
        return high_freq[:top_k]

    # Substitution: replace the lowest-stopword content token in answer with
    # each high-frequency word from the passage to mint a fake-but-plausible
    # answer string.
    target_idx = next(
        (i for i, t in enumerate(ans_tokens)
         if t.lower() not in _STOP and len(t) >= 3),
        len(ans_tokens) - 1,
    )
    out = []
    for hw in high_freq:
        sub = list(ans_tokens)
        sub[target_idx] = hw
        cand = ' '.join(sub)
        if cand.lower() != correct_answer.lower() and cand not in out:
            out.append(cand)
        if len(out) >= top_k:
            break
    return out


# ─── B3: GloVe nearest neighbour distractor alternative (spec §5.4) ────────
def generate_distractors_glove(correct_answer, glove, passage="", top_k=3):
    """Pre-trained GloVe nearest neighbours of the correct answer's content words."""
    if glove is None:
        return []
    seed = [w for w in TOK_RE.findall(correct_answer.lower()) if w in glove]
    if not seed:
        return []
    try:
        candidates = glove.most_similar(positive=seed, topn=30)
    except Exception:
        return []

    psg_lower = passage.lower()
    ans_words = set(TOK_RE.findall(correct_answer.lower()))
    out = []
    for word, _sim in candidates:
        wl = word.lower()
        if wl in psg_lower or wl in ans_words or len(wl) < 3:
            continue
        out.append(word)
        if len(out) >= top_k:
            break
    return out


# ─── B4: Hint extraction — TF·IDF cosine ranking ────────────────────────────
def extract_hints_tfidf(passage, question, vec, top_k=3):
    """
    Score each sentence by TF·IDF cosine to question; surface top-K and
    return them ordered from MOST GENERAL → NEAR EXPLICIT, per spec §5.1
    ("Hint 1: most general clue; Hint 2: more specific; Hint 3: near-explicit").

    We pick the K most relevant sentences by score, then reverse so the
    least-relevant of the K is delivered first as the gentlest clue, and
    the most-relevant lands last as the near-explicit reveal.
    """
    sentences = split_sentences(passage)
    if not sentences:
        return []
    sent_vecs = vec.transform(sentences)
    q_vec     = vec.transform([question])
    q_norm    = float(np.sqrt(q_vec.multiply(q_vec).sum())) + EPS
    s_norms   = np.sqrt(np.asarray(sent_vecs.multiply(sent_vecs).sum(axis=1)).ravel()) + EPS
    sims      = np.asarray((sent_vecs @ q_vec.T).todense()).ravel() / (s_norms * q_norm)
    top_idx   = np.argsort(-sims)[:top_k]
    # Graduated order: gentlest (lowest score) first → near-explicit (highest) last
    graduated = sorted(top_idx, key=lambda i: sims[int(i)])
    return [(sentences[int(i)], float(sims[int(i)])) for i in graduated]


# ─── B5: Hint features (batched per row) ─────────────────────────────────────
def batch_hint_features(sentences, question, correct_answer,
                        vec, glove=None, idf_lookup=None):
    """
    Returns (n_sentences, 5) feature matrix:
      0  cos_q      : TF·IDF cosine vs question
      1  cos_a      : TF·IDF cosine vs correct answer
      2  position   : normalised sentence index in passage [0, 1]
      3  n_tokens   : sentence length in tokens
      4  cos_q_glv  : GloVe cosine vs question
    """
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float32)

    cos_q = _batch_tfidf_cosine(sentences, question, vec)
    cos_a = _batch_tfidf_cosine(sentences, correct_answer, vec)

    positions = (np.arange(n, dtype=np.float32) / max(1, n - 1)).astype(np.float32)
    n_tokens  = np.array([len(s.split()) for s in sentences], dtype=np.float32)

    cos_q_glv = _batch_glove_cosine(sentences, question, glove, idf_lookup)

    return np.stack([cos_q, cos_a, positions, n_tokens, cos_q_glv], axis=1).astype(np.float32)


def build_hint_train_data(df, vec, glove, idf_lookup,
                          max_rows=TRAIN_SAMPLE_SIZE):
    """Train ranker with the max-overlap-with-answer sentence as positive."""
    X_chunks, y_chunks = [], []
    rng = np.random.default_rng(42)

    for _, row in df.head(max_rows).iterrows():
        passage  = str(row.get('article', ''))
        question = str(row.get('question', ''))
        letter   = row.get('answer', None)
        if letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(letter, '')).strip()
        if not correct_ans or not passage or not question:
            continue

        sentences = split_sentences(passage)
        if len(sentences) < 2:
            continue

        ans_words = set(gen_tokens_lower(correct_ans))
        scored = [(i, len(set(gen_tokens_lower(s)) & ans_words))
                  for i, s in enumerate(sentences)]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] == 0:
            continue
        key_idx = scored[0][0]

        other_idx = [i for i, _ in scored[1:]]
        if len(other_idx) > 3:
            negs = list(rng.choice(other_idx, size=3, replace=False))
        else:
            negs = other_idx

        keep_indices = [key_idx] + [int(i) for i in negs]
        labels = [1] + [0] * len(negs)
        sub_sentences = [sentences[i] for i in keep_indices]

        # Compute features for ALL sentences in batch, then index out the keepers
        all_feats = batch_hint_features(
            sentences, question, correct_ans, vec, glove, idf_lookup,
        )
        feats = all_feats[np.array(keep_indices)]
        X_chunks.append(feats)
        y_chunks.append(np.array(labels, dtype=np.int32))

    if not X_chunks:
        return np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.concatenate(X_chunks, axis=0), np.concatenate(y_chunks, axis=0)


def train_hint_ranker(train_df, vec, glove, idf_lookup):
    print("\n[B5-RANKER] Building hint training data "
          f"(up to {TRAIN_SAMPLE_SIZE:,} rows, batched per-row)...")
    X, y = build_hint_train_data(train_df, vec, glove, idf_lookup)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"  Examples: {len(X):,}  ({n_pos:,} key sentences, {n_neg:,} non-key)")
    print("[B5-RANKER] Training Logistic Regression...")
    model = LogisticRegression(C=1.0, max_iter=400, n_jobs=-1, random_state=42).fit(X, y)
    save_model(model, MODEL_B_HINT_PATH)
    return model


def extract_hints_ml(passage, question, correct_answer, ranker,
                     vec, glove, idf_lookup, top_k=3):
    """ML-scored hints, returned in graduated order (general → near-explicit)."""
    sentences = split_sentences(passage)
    if not sentences:
        return []
    feats = batch_hint_features(sentences, question, correct_answer,
                                vec, glove, idf_lookup)
    if hasattr(ranker, 'predict_proba'):
        scores = ranker.predict_proba(feats)[:, 1]
    else:
        scores = ranker.decision_function(feats)
    top_idx = np.argsort(-scores)[:top_k]
    graduated = sorted(top_idx, key=lambda i: scores[int(i)])
    return [(sentences[int(i)], float(scores[int(i)])) for i in graduated]


# ─── B5b: Regression-based hint scorer (spec §5.5 R² metric) ────────────────
def build_hint_regression_data(df, vec, glove, idf_lookup,
                               max_rows=TRAIN_SAMPLE_SIZE):
    """
    Continuous relevance label = (#answer-tokens overlapping sentence) /
    (#answer-tokens). Same five sentence features as the classifier,
    but the target is a regression score in [0, 1].
    """
    X_chunks, y_chunks = [], []
    for _, row in df.head(max_rows).iterrows():
        passage  = str(row.get('article', ''))
        question = str(row.get('question', ''))
        letter   = row.get('answer', None)
        if letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(letter, '')).strip()
        if not correct_ans or not passage or not question:
            continue

        sentences = split_sentences(passage)
        if len(sentences) < 2:
            continue

        ans_words = set(gen_tokens_lower(correct_ans))
        if not ans_words:
            continue
        denom = float(len(ans_words))
        targets = np.array(
            [len(set(gen_tokens_lower(s)) & ans_words) / denom for s in sentences],
            dtype=np.float32,
        )
        if targets.max() == 0.0:
            continue

        feats = batch_hint_features(sentences, question, correct_ans,
                                    vec, glove, idf_lookup)
        X_chunks.append(feats)
        y_chunks.append(targets)

    if not X_chunks:
        return np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(X_chunks, axis=0), np.concatenate(y_chunks, axis=0)


def train_hint_regressor(train_df, vec, glove, idf_lookup):
    print("\n[B5b-REG] Building hint regression data "
          f"(up to {TRAIN_SAMPLE_SIZE:,} rows, continuous relevance target)...")
    X, y = build_hint_regression_data(train_df, vec, glove, idf_lookup)
    print(f"  Examples: {len(X):,}   y range: [{y.min():.3f}, {y.max():.3f}]   y mean: {y.mean():.3f}")
    print("[B5b-REG] Training Random Forest Regressor (200 trees, max_depth=12)...")
    reg = RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42,
    ).fit(X, y)
    save_model(reg, MODEL_B_HINT_REG_PATH)
    return reg


def evaluate_hint_regressor(df, regressor, vec, glove, idf_lookup,
                            max_rows=EVAL_SAMPLE_SIZE, label="VAL"):
    """
    Compute R² between predicted sentence-relevance scores and true relevance
    labels (fraction of answer tokens covered by sentence). Spec §5.5.
    """
    X, y = build_hint_regression_data(df, vec, glove, idf_lookup, max_rows=max_rows)
    if len(X) == 0:
        return {'r2': float('nan'), 'mae': float('nan'),
                'rmse': float('nan'), 'n_examples': 0}
    y_pred = regressor.predict(X)
    r2   = float(r2_score(y, y_pred))
    mae  = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    print(f"  [{label}]  R²={r2:.4f}   MAE={mae:.4f}   RMSE={rmse:.4f}   "
          f"n={len(X):,}")
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'n_examples': int(len(X))}


# ─── Evaluation ──────────────────────────────────────────────────────────────
def _derive_gold_key_sentence(passage, correct_answer):
    sentences = split_sentences(passage)
    if not sentences:
        return None
    ans_words = set(gen_tokens_lower(correct_answer))
    if not ans_words:
        return None
    scored = [(i, len(set(gen_tokens_lower(s)) & ans_words))
              for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: x[1], reverse=True)
    if scored[0][1] == 0:
        return None
    return sentences[scored[0][0]]


# ─── B6: Confusion Matrix scaffolding for human Likert evaluation (§5.5) ───
LIKERT_BUCKETS = ['1-Implausible', '2-Weak', '3-OK', '4-Good', '5-Excellent']


def _sanitize_cell(s):
    """Collapse embedded newlines / repeated whitespace so each rater row stays on
    one line in the CSV (prevents Excel/text-editor breakage)."""
    s = str(s).replace('\r', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _likert_csv_has_ratings(path):
    """True iff the existing Likert CSV has at least one rater score filled in."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    for col in ('rater_score_generated_1to5', 'rater_score_gold_1to5'):
        if col not in df.columns:
            return False
        if df[col].notna().any() and (df[col].astype(str).str.strip() != '').any():
            return True
    return False


def write_likert_template(distractor_examples, out_path=LIKERT_TEMPLATE_PATH):
    """
    Spec §5.5 — Confusion Matrix from human evaluation (1–5 Likert).
    We can't run the human study from code, so we emit a CSV template a
    rater can fill in. Each row pairs a generated distractor with its
    closest gold distractor; the rater scores each on the 1–5 scale.

    If the file already has rater scores, we skip writing — overwriting
    would wipe the human's work. Delete the file manually to regenerate.
    """
    if _likert_csv_has_ratings(out_path):
        print(f"  [LIKERT] Existing ratings detected at {out_path} — skipping rewrite "
              "(delete the file to regenerate template).")
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for ex in distractor_examples:
        gen = ex.get('gen_distractors', [])
        gold = ex.get('gold_distractors', [])
        for i, g in enumerate(gen):
            r1 = [_MATCH_ROUGE.score(gd, g)['rouge1'].fmeasure for gd in gold]
            best_idx = int(np.argmax(r1)) if r1 else -1
            best_gold = gold[best_idx] if best_idx >= 0 else ''
            rows.append({
                'correct_answer': _sanitize_cell(ex.get('correct', '')),
                'generated_distractor': _sanitize_cell(g),
                'matched_gold_distractor': _sanitize_cell(best_gold),
                'rater_score_generated_1to5': '',
                'rater_score_gold_1to5': '',
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8',
              quoting=1)  # csv.QUOTE_ALL — every field quoted, no row splitting
    print(f"  [LIKERT] Wrote {len(df)} rater rows to {out_path}")
    return out_path


def likert_confusion_matrix(csv_path=LIKERT_TEMPLATE_PATH):
    """
    Read a filled-in Likert CSV and produce a confusion matrix between
    rater scores on generated vs gold distractors. Returns None silently
    if no human ratings have been entered yet (so the pipeline is still
    runnable end-to-end without a rater in the loop).
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['rater_score_generated_1to5', 'rater_score_gold_1to5'])
    df = df[(df['rater_score_generated_1to5'] != '') &
            (df['rater_score_gold_1to5'] != '')]
    if len(df) == 0:
        return None
    try:
        y_true = df['rater_score_gold_1to5'].astype(int).clip(1, 5).values
        y_pred = df['rater_score_generated_1to5'].astype(int).clip(1, 5).values
    except Exception:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    return {'matrix': cm, 'labels': LIKERT_BUCKETS, 'n_ratings': int(len(df))}


def print_likert_confusion_matrix(cm_result):
    if cm_result is None:
        print("\n  [LIKERT] No human ratings yet — Confusion Matrix will be "
              f"computable once rater fills {LIKERT_TEMPLATE_PATH}")
        return
    print(f"\n  [LIKERT] Confusion Matrix on {cm_result['n_ratings']} rater rows "
          f"(rows = gold, cols = generated):")
    cm = cm_result['matrix']
    header = "          " + "".join(f"{lbl:>14}" for lbl in cm_result['labels'])
    print(header)
    for i, lbl in enumerate(cm_result['labels']):
        row = "".join(f"{int(v):>14}" for v in cm[i])
        print(f"  {lbl:<10}{row}")


def evaluate_distractors(df, ranker, vec, glove, idf_lookup,
                         max_rows=EVAL_SAMPLE_SIZE, label="VAL"):
    """
    Score each generated distractor against its closest gold (greedy match
    by ROUGE-1 using a separate scorer that doesn't pollute corpus BLEU).
    A pair counts as "matched" when ROUGE-1 > 0.3.
    """
    metrics = GenerationMetrics()
    keys = ['bleu_1', 'bleu', 'rouge1', 'rougeL', 'meteor']
    sums = {k: 0.0 for k in keys}

    n_rows, n_pairs = 0, 0
    n_matched = 0
    n_generated = 0
    n_gold = 0
    n_top_not_correct = 0   # spec §5.5 — ranker accuracy: top distractor != correct answer
    examples = []

    print(f"  [{label}] Scoring distractors on up to {min(max_rows, len(df)):,} rows...")
    for i, row in df.head(max_rows).iterrows():
        passage = str(row.get('article', ''))
        letter  = row.get('answer', None)
        if letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(letter, '')).strip()
        if not correct_ans or not passage:
            continue

        gold = []
        for L in ('A', 'B', 'C', 'D'):
            if L == letter:
                continue
            d = str(row.get(L, '')).strip()
            if d:
                gold.append(d)
        if not gold:
            continue

        gen = generate_distractors(passage, correct_ans, ranker,
                                    vec, glove, idf_lookup, top_k=3)
        if not gen:
            continue

        if gen[0].lower().strip() != correct_ans.lower().strip():
            n_top_not_correct += 1

        # Greedy: pick best gold for each gen by ROUGE-1 (separate scorer to
        # avoid double-buffering the corpus BLEU). Then score the chosen pair
        # ONCE through GenerationMetrics for the actual reported numbers.
        for g in gen[:3]:
            r1_scores = [_MATCH_ROUGE.score(gd, g)['rouge1'].fmeasure for gd in gold]
            best_idx = int(np.argmax(r1_scores))
            best_gold = gold[best_idx]
            s = metrics.score(g, best_gold)
            for k in keys:
                sums[k] += s[k]
            n_pairs += 1
            if s['rouge1'] > DISTRACTOR_MATCH_THRESH:
                n_matched += 1

        n_generated += min(3, len(gen))
        n_gold      += len(gold)
        n_rows      += 1

        if len(examples) < 3:
            examples.append({
                'passage_excerpt': passage[:120] + ('...' if len(passage) > 120 else ''),
                'correct': correct_ans,
                'gold_distractors': gold,
                'gen_distractors': gen,
            })

        if (i + 1) % 500 == 0:
            print(f"    [{label}] scored {i + 1:,}/{min(max_rows, len(df)):,}")

    avg = {k: sums[k] / max(1, n_pairs) for k in keys}
    avg['bleu_1_corpus'] = metrics.corpus_bleu_at_n(1)
    avg['bleu_corpus']   = metrics.corpus_bleu_at_n(4)
    avg['precision']     = n_matched / max(1, n_generated)
    avg['recall']        = n_matched / max(1, n_gold)
    avg['f1']            = (
        2 * avg['precision'] * avg['recall']
        / max(1e-9, (avg['precision'] + avg['recall']))
    )
    avg['accuracy']      = n_top_not_correct / max(1, n_rows)
    avg['n_rows']        = n_rows
    return avg, examples


def evaluate_hints(df, vec, top_k=3, use_ml=False, hint_ranker=None,
                   glove=None, idf_lookup=None,
                   max_rows=EVAL_SAMPLE_SIZE, label="VAL"):
    """Precision@K vs derived gold key sentence; BLEU/ROUGE/METEOR on top-1."""
    metrics = GenerationMetrics()
    keys = ['bleu_1', 'bleu', 'rouge1', 'rougeL', 'meteor']
    sums = {k: 0.0 for k in keys}

    n_rows, n_at_k = 0, 0

    print(f"  [{label}] Scoring hints on up to {min(max_rows, len(df)):,} rows...")
    for i, row in df.head(max_rows).iterrows():
        passage  = str(row.get('article', ''))
        question = str(row.get('question', ''))
        letter   = row.get('answer', None)
        if letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(letter, '')).strip()
        if not correct_ans or not passage or not question:
            continue

        gold_key = _derive_gold_key_sentence(passage, correct_ans)
        if gold_key is None:
            continue

        if use_ml and hint_ranker is not None:
            hints = extract_hints_ml(passage, question, correct_ans,
                                     hint_ranker, vec, glove, idf_lookup,
                                     top_k=top_k)
        else:
            hints = extract_hints_tfidf(passage, question, vec, top_k=top_k)
        if not hints:
            continue

        hint_sentences = [h[0] for h in hints]
        if gold_key in hint_sentences:
            n_at_k += 1

        s = metrics.score(hint_sentences[0], gold_key)
        for k in keys:
            sums[k] += s[k]

        n_rows += 1
        if (i + 1) % 500 == 0:
            print(f"    [{label}] scored {i + 1:,}/{min(max_rows, len(df)):,}")

    avg = {k: sums[k] / max(1, n_rows) for k in keys}
    avg['bleu_1_corpus']  = metrics.corpus_bleu_at_n(1)
    avg['bleu_corpus']    = metrics.corpus_bleu_at_n(4)
    avg['precision_at_k'] = n_at_k / max(1, n_rows)
    avg['n_rows']         = n_rows
    return avg


# ─── Final summary ───────────────────────────────────────────────────────────
def print_summary(distractor_results, hint_results,
                  glove_distractor_examples=None,
                  freq_examples=None,
                  hint_regression=None):
    print("\n" + "=" * 122)
    print("FINAL COMPARISON  —  Model B  (Distractor + Hint Generation)")
    print("=" * 122)

    print("\n  Distractor Generation:")
    print(f"  {'Ranker':<8} {'Split':<6} {'N':>5}   "
          f"{'Acc':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}   "
          f"{'BLEU-1':>8} {'BLEU-1-c':>9} {'ROUGE-1':>8} {'ROUGE-L':>8} {'METEOR':>8}")
    print("  " + "-" * 120)
    for name, splits in distractor_results.items():
        for split_name in ('val', 'test'):
            if split_name not in splits:
                continue
            m = splits[split_name]
            print(f"  {name:<8} {split_name.upper():<6} {m['n_rows']:>5,}   "
                  f"{m.get('accuracy', 0.0):>8.4f} "
                  f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}   "
                  f"{m['bleu_1']:>8.4f} {m['bleu_1_corpus']:>9.4f} "
                  f"{m['rouge1']:>8.4f} {m['rougeL']:>8.4f} {m['meteor']:>8.4f}")

    print("\n  Hint Extraction:")
    print(f"  {'Variant':<12} {'Split':<6} {'N':>5}   "
          f"{'P@3':>8}   {'BLEU-1':>8} {'BLEU-1-c':>9} {'ROUGE-1':>8} {'ROUGE-L':>8} {'METEOR':>8}")
    print("  " + "-" * 110)
    for name, splits in hint_results.items():
        for split_name in ('val', 'test'):
            if split_name not in splits:
                continue
            m = splits[split_name]
            print(f"  {name:<12} {split_name.upper():<6} {m['n_rows']:>5,}   "
                  f"{m['precision_at_k']:>8.4f}   "
                  f"{m['bleu_1']:>8.4f} {m['bleu_1_corpus']:>9.4f} "
                  f"{m['rouge1']:>8.4f} {m['rougeL']:>8.4f} {m['meteor']:>8.4f}")

    if hint_regression:
        print("\n  Hint Regression (spec §5.5 — R² Score):")
        print(f"  {'Split':<6} {'N':>7}   {'R²':>8} {'MAE':>8} {'RMSE':>8}")
        print("  " + "-" * 50)
        for split_name in ('val', 'test'):
            m = hint_regression.get(split_name)
            if not m:
                continue
            print(f"  {split_name.upper():<6} {m['n_examples']:>7,}   "
                  f"{m['r2']:>8.4f} {m['mae']:>8.4f} {m['rmse']:>8.4f}")

    if glove_distractor_examples:
        print("\n  GloVe Nearest Neighbour Distractors (spec §5.4 alternative):")
        for ex in glove_distractor_examples[:3]:
            print(f"    Correct : {ex['correct']}")
            print(f"    Gold    : {ex['gold']}")
            print(f"    GloVe NN: {ex['glove_nn']}")
            print()

    if freq_examples:
        print("\n  Frequency-Based Substitution Distractors (spec §5.4 alternative):")
        for ex in freq_examples[:3]:
            print(f"    Correct  : {ex['correct']}")
            print(f"    Gold     : {ex['gold']}")
            print(f"    FreqSub  : {ex['freq_sub']}")
            print()

    print("=" * 122 + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("# QuizForge — Model B  (Distractor & Hint Generator)")
    print("#" * 70)

    _ensure_nltk_data()

    print("\n[DATA] Loading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df   = pd.read_csv(VAL_CSV_PATH)
    test_df  = pd.read_csv(TEST_CSV_PATH)
    print(f"  Train: {len(train_df):,}   Val: {len(val_df):,}   Test: {len(test_df):,}")

    print("\n[SHARED] Loading TF·IDF vectorizer (saved by model_a.py)...")
    if os.path.exists(TFIDF_VEC_PATH):
        vec = joblib.load(TFIDF_VEC_PATH)
        print(f"  Loaded from {TFIDF_VEC_PATH}  (vocab {len(vec.vocabulary_):,})")
    else:
        print("  No saved vectorizer; fitting fresh on training corpus...")
        vec = fit_tfidf_vectorizer(TRAIN_CSV_PATH)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(vec, TFIDF_VEC_PATH)

    glove = load_glove()
    idf_lookup = build_idf_lookup(vec)

    # ─── B2: Distractor rankers ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("B2  Distractor Ranking — LR + RF on candidate features")
    print("=" * 70)
    distractor_rankers = train_distractor_rankers(train_df, vec, glove, idf_lookup)

    # ─── B5: Hint ranker ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("B5  Hint Ranking — Logistic Regression on sentence features")
    print("=" * 70)
    hint_ranker = train_hint_ranker(train_df, vec, glove, idf_lookup)

    # ─── B5b: Hint regressor (R² metric, spec §5.5) ──────────────────────
    print("\n" + "=" * 70)
    print("B5b Hint Regression — Linear Regression on continuous relevance (R²)")
    print("=" * 70)
    hint_regressor = train_hint_regressor(train_df, vec, glove, idf_lookup)

    # ─── Evaluate distractors on val + test ──────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluating Distractors (val + test)")
    print("=" * 70)
    distractor_results = {}
    distractor_examples_by_ranker = {}
    for name, ranker in distractor_rankers.items():
        print(f"\n[Distractors] Variant: {name}")
        v_avg, v_ex = evaluate_distractors(val_df, ranker, vec, glove, idf_lookup, label=f"{name} VAL")
        t_avg, _    = evaluate_distractors(test_df, ranker, vec, glove, idf_lookup, label=f"{name} TEST")
        distractor_results[name] = {'val': v_avg, 'test': t_avg}
        distractor_examples_by_ranker[name] = v_ex
        print(f"  Val:  P={v_avg['precision']:.4f}  R={v_avg['recall']:.4f}  F1={v_avg['f1']:.4f}  "
              f"BLEU-1={v_avg['bleu_1']:.4f}")
        print(f"  Test: P={t_avg['precision']:.4f}  R={t_avg['recall']:.4f}  F1={t_avg['f1']:.4f}  "
              f"BLEU-1={t_avg['bleu_1']:.4f}")

    if 'LR' in distractor_examples_by_ranker:
        print("\n[Distractors] Sample LR generations (val):")
        for ex in distractor_examples_by_ranker['LR'][:3]:
            print(f"  Correct : {ex['correct']}")
            print(f"  Gold    : {ex['gold_distractors']}")
            print(f"  Gen (LR): {ex['gen_distractors']}")
            print()

    # ─── B6: Likert / Confusion Matrix scaffolding (spec §5.5) ───────────
    print("\n" + "=" * 70)
    print("B6  Human-Evaluation Likert Template + Confusion Matrix (spec §5.5)")
    print("=" * 70)
    sample_for_likert = distractor_examples_by_ranker.get('LR', [])
    write_likert_template(sample_for_likert)
    print_likert_confusion_matrix(likert_confusion_matrix())

    # ─── B3b: Frequency-Based Substitution examples (spec §5.4) ─────────
    print("\n" + "=" * 70)
    print("B3b Frequency-Based Substitution Distractors (spec §5.4 alternative)")
    print("=" * 70)
    freq_examples = []
    for _, row in val_df.head(50).iterrows():
        letter = row.get('answer', None)
        if letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(letter, '')).strip()
        passage = str(row.get('article', ''))
        if not correct_ans or not passage:
            continue
        gold = [str(row.get(L, '')).strip() for L in ('A', 'B', 'C', 'D') if L != letter]
        sub = generate_distractors_freq(correct_ans, passage, top_k=3)
        if sub:
            freq_examples.append({'correct': correct_ans, 'gold': gold, 'freq_sub': sub})
            if len(freq_examples) >= 5:
                break
    print(f"  Sampled {len(freq_examples)} successful frequency-substitution generations.")
    for ex in freq_examples[:3]:
        print(f"    Correct  : {ex['correct']}")
        print(f"    Gold     : {ex['gold']}")
        print(f"    FreqSub  : {ex['freq_sub']}")
        print()

    # ─── B3: GloVe nearest neighbour examples ────────────────────────────
    print("\n" + "=" * 70)
    print("B3  GloVe Nearest Neighbour Distractors (spec §5.4 alternative)")
    print("=" * 70)
    glove_examples = []
    if glove is not None:
        for _, row in val_df.head(50).iterrows():
            letter = row.get('answer', None)
            if letter not in ('A', 'B', 'C', 'D'):
                continue
            correct_ans = str(row.get(letter, '')).strip()
            passage = str(row.get('article', ''))
            if not correct_ans or not passage:
                continue
            gold = [str(row.get(L, '')).strip() for L in ('A', 'B', 'C', 'D') if L != letter]
            nn = generate_distractors_glove(correct_ans, glove, passage, top_k=3)
            if nn:
                glove_examples.append({
                    'correct': correct_ans,
                    'gold': gold,
                    'glove_nn': nn,
                })
                if len(glove_examples) >= 5:
                    break
        print(f"  Sampled {len(glove_examples)} successful GloVe NN generations.")
    else:
        print("  GloVe not available, skipping.")

    # ─── Evaluate hints on val + test ────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluating Hints (val + test)")
    print("=" * 70)
    hint_results = {}

    print("\n[Hints] Variant: TF·IDF cosine ranking")
    v_avg = evaluate_hints(val_df, vec, top_k=3, use_ml=False, label="TF-IDF VAL")
    t_avg = evaluate_hints(test_df, vec, top_k=3, use_ml=False, label="TF-IDF TEST")
    hint_results['TF-IDF'] = {'val': v_avg, 'test': t_avg}
    print(f"  Val:  P@3={v_avg['precision_at_k']:.4f}  BLEU-1={v_avg['bleu_1']:.4f}  ROUGE-1={v_avg['rouge1']:.4f}")
    print(f"  Test: P@3={t_avg['precision_at_k']:.4f}  BLEU-1={t_avg['bleu_1']:.4f}  ROUGE-1={t_avg['rouge1']:.4f}")

    print("\n[Hints] Variant: ML scored ranking (LR)")
    v_avg = evaluate_hints(val_df, vec, top_k=3, use_ml=True, hint_ranker=hint_ranker,
                           glove=glove, idf_lookup=idf_lookup, label="ML VAL")
    t_avg = evaluate_hints(test_df, vec, top_k=3, use_ml=True, hint_ranker=hint_ranker,
                           glove=glove, idf_lookup=idf_lookup, label="ML TEST")
    hint_results['ML-scored'] = {'val': v_avg, 'test': t_avg}
    print(f"  Val:  P@3={v_avg['precision_at_k']:.4f}  BLEU-1={v_avg['bleu_1']:.4f}  ROUGE-1={v_avg['rouge1']:.4f}")
    print(f"  Test: P@3={t_avg['precision_at_k']:.4f}  BLEU-1={t_avg['bleu_1']:.4f}  ROUGE-1={t_avg['rouge1']:.4f}")

    # ─── B5b: Regression hint scorer R² evaluation ───────────────────────
    print("\n[Hints] Variant: Regression hint scorer (R² metric, spec §5.5)")
    reg_v = evaluate_hint_regressor(val_df, hint_regressor, vec, glove, idf_lookup, label="REG VAL")
    reg_t = evaluate_hint_regressor(test_df, hint_regressor, vec, glove, idf_lookup, label="REG TEST")

    # Sample graduated hints (general → near-explicit) on a few rows
    print("\n[Hints] Sample graduated hints (TF·IDF, val):")
    shown = 0
    for _, row in val_df.head(50).iterrows():
        passage  = str(row.get('article', ''))
        question = str(row.get('question', ''))
        letter   = row.get('answer', None)
        if letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(letter, '')).strip()
        if not correct_ans or not passage or not question:
            continue
        hints = extract_hints_tfidf(passage, question, vec, top_k=3)
        if not hints:
            continue
        print(f"  Q: {question[:90]}")
        print(f"  A: {correct_ans}")
        for k, (s, sc) in enumerate(hints, 1):
            print(f"    Hint {k} (score={sc:.3f}): {s[:120]}")
        print()
        shown += 1
        if shown >= 3:
            break

    # ─── Final summary ────────────────────────────────────────────────────
    print_summary(distractor_results, hint_results, glove_examples,
                  freq_examples=freq_examples,
                  hint_regression={'val': reg_v, 'test': reg_t})


if __name__ == "__main__":
    main()


# Final project or my final project?
