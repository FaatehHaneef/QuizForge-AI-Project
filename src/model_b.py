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

Evaluation metrics (per spec §5.5 + professor's BLEU/ROUGE/METEOR decree):
  Distractors : Precision / Recall / F1 (vs gold A/B/C/D distractors;
                a generated distractor is judged a match when ROUGE 1 against
                the closest gold exceeds 0.3); plus BLEU 1, BLEU 4, ROUGE 1,
                ROUGE L, METEOR averaged across all matched pairs (sentence
                level) and pooled across the split (corpus level).
  Hints       : Precision @ K against the derived gold key sentence (the
                passage sentence with maximum word overlap to the correct
                answer), plus BLEU / ROUGE / METEOR of the top one hint vs
                the gold key sentence.

Reuses infrastructure from model_a (no neural code leaks):
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TEST_CSV_PATH,
    ANSWER_MAP,
    MODELS_DIR,
)
from utils import save_model

# Reuse classical utilities from Model A. No neural code is imported here.
from model_a import (
    split_sentences,
    gen_tokens_lower,
    row_cosine,
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
MODEL_B_HINT_PATH    = os.path.join(MODELS_DIR, "model_b_hint_lr.pkl")

EPS                   = 1e-9
TRAIN_SAMPLE_SIZE     = 15_000   # rows used for ranker training
EVAL_SAMPLE_SIZE      = 4_000    # rows used for evaluation (val / test)
MAX_CANDIDATES        = 25       # candidate phrases per passage
DISTRACTOR_MATCH_THRESH = 0.3    # ROUGE 1 above this counts as match
TOK_RE                = re.compile(r'\b[a-zA-Z]+\b')


# ─── B1: Distractor candidate extraction ─────────────────────────────────────
def extract_candidate_phrases(passage, max_candidates=MAX_CANDIDATES):
    """
    Spec §5.3.1 Step 1 — extract candidate phrases from the passage using
    simple string matching and frequency based selection. No NLP tools.

    Three candidate sources:
      - Whole sentences of moderate length (3 to 25 tokens)
      - Capitalised noun phrase like spans (proper nouns and named entities)
      - High frequency content words from the passage
    """
    sentences = split_sentences(passage)
    candidates = []

    # 1. Whole sentences of moderate length
    for sent in sentences:
        clean = sent.strip().rstrip('.!?,;:"\'')
        wc = len(clean.split())
        if 3 <= wc <= 25 and clean:
            candidates.append(clean)

    # 2. Capitalised noun phrase like spans (sequences of Title cased words,
    #    optionally with lowercase connectors in between)
    np_pat = re.compile(r'\b[A-Z][a-z]+(?:\s+(?:of|the|a|an|in|on))?\s*(?:[A-Z][a-z]+)?\b')
    for sent in sentences:
        for m in np_pat.findall(sent):
            wc = len(m.split())
            if 1 <= wc <= 6 and m.strip():
                candidates.append(m.strip())

    # 3. High frequency content words (length >= 4, top by passage count)
    word_counts = Counter()
    for sent in sentences:
        for w in TOK_RE.findall(sent.lower()):
            if len(w) >= 4:
                word_counts[w] += 1
    for w, _ in word_counts.most_common(10):
        candidates.append(w)

    # Deduplicate (case insensitive) preserving order
    seen, out = set(), []
    for c in candidates:
        norm = c.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(c)

    return out[:max_candidates]


# ─── B2: Distractor features ─────────────────────────────────────────────────
def _char_bigrams(s):
    s = re.sub(r'\s+', ' ', s.lower())
    return set(s[i:i + 2] for i in range(len(s) - 1)) if len(s) >= 2 else set()


def distractor_features(candidate, correct_answer, passage,
                        vec, glove=None, idf_lookup=None):
    """
    Per spec §5.3.1 Step 2 — feature engineering for a distractor candidate.

    Returns a 7 dimensional feature vector:
      0  cos_tfidf      : TF·IDF cosine similarity to the correct answer
      1  char_match     : Jaccard on character bigrams vs correct answer
      2  passage_freq   : occurrences of candidate in passage / passage length
      3  length_diff    : abs token length difference vs correct answer
      4  length_ratio   : token length ratio (candidate / answer)
      5  cos_glove      : GloVe cosine similarity to correct answer (or 0)
      6  word_overlap   : fraction of answer tokens present in candidate
    """
    cand_str = str(candidate)
    ans_str  = str(correct_answer)
    psg_str  = str(passage).lower()

    cand_lower = cand_str.lower().strip()
    ans_lower  = ans_str.lower().strip()

    # 1) TF·IDF cosine
    cand_vec = vec.transform([cand_str])
    ans_vec  = vec.transform([ans_str])
    cos_tfidf = float(row_cosine(cand_vec, ans_vec)[0])

    # 2) Character bigram Jaccard
    bg_c, bg_a = _char_bigrams(cand_lower), _char_bigrams(ans_lower)
    char_match = (len(bg_c & bg_a) / max(1, len(bg_c | bg_a))) if bg_c or bg_a else 0.0

    # 3) Passage frequency
    psg_freq = psg_str.count(cand_lower) / max(1, len(psg_str.split()))

    # 4–5) Length features
    lc = max(1, len(cand_str.split()))
    la = max(1, len(ans_str.split()))
    length_diff  = float(abs(lc - la))
    length_ratio = float(lc) / float(la)

    # 6) GloVe cosine
    if glove is not None:
        dim = glove.vector_size
        cg = text_to_glove_vec(cand_str, glove, dim, idf_lookup)
        ag = text_to_glove_vec(ans_str,  glove, dim, idf_lookup)
        cn = float(np.linalg.norm(cg)) + EPS
        an = float(np.linalg.norm(ag)) + EPS
        cos_glove = float((cg @ ag) / (cn * an))
    else:
        cos_glove = 0.0

    # 7) Word overlap with answer
    cw = set(TOK_RE.findall(cand_lower))
    aw = set(TOK_RE.findall(ans_lower))
    word_overlap = (len(cw & aw) / max(1, len(aw))) if aw else 0.0

    return np.array([
        cos_tfidf, char_match, psg_freq,
        length_diff, length_ratio,
        cos_glove, word_overlap,
    ], dtype=np.float32)


def build_distractor_train_data(df, vec, glove, idf_lookup,
                                 max_rows=TRAIN_SAMPLE_SIZE):
    """
    Mine training labels from the dataset itself: for every row, the three
    non correct options are positive distractor examples; three randomly
    sampled passage candidates not matching the correct answer are negatives.
    """
    X, y = [], []
    rng = np.random.default_rng(42)

    for _, row in df.head(max_rows).iterrows():
        passage = str(row.get('article', ''))
        correct_letter = row.get('answer', None)
        if correct_letter not in ('A', 'B', 'C', 'D'):
            continue
        correct_ans = str(row.get(correct_letter, '')).strip()
        if not correct_ans or not passage:
            continue

        # Positives: dataset's own A / B / C / D minus the correct one
        gold_distractors = []
        for letter in ('A', 'B', 'C', 'D'):
            if letter == correct_letter:
                continue
            d = str(row.get(letter, '')).strip()
            if d:
                gold_distractors.append(d)

        # Negatives: random passage candidates that are not the correct answer
        cands = extract_candidate_phrases(passage)
        cands = [c for c in cands if c.lower().strip() != correct_ans.lower().strip()]
        if len(cands) > 3:
            neg_indices = rng.choice(len(cands), size=3, replace=False)
            negatives = [cands[i] for i in neg_indices]
        else:
            negatives = cands

        for d in gold_distractors:
            X.append(distractor_features(d, correct_ans, passage, vec, glove, idf_lookup))
            y.append(1)
        for n in negatives:
            X.append(distractor_features(n, correct_ans, passage, vec, glove, idf_lookup))
            y.append(0)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def train_distractor_rankers(train_df, vec, glove, idf_lookup):
    """Train Logistic Regression and Random Forest distractor rankers."""
    print("\n[B2-RANKER] Building distractor training data "
          f"(up to {TRAIN_SAMPLE_SIZE:,} rows)...")
    X, y = build_distractor_train_data(train_df, vec, glove, idf_lookup)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"  Examples: {len(X):,}  ({n_pos:,} distractors, {n_neg:,} non distractors)")

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
    Score all candidates and pick top K with a Jaccard diversity filter
    (skip near duplicates). The correct answer itself is excluded.
    """
    cands = extract_candidate_phrases(passage)
    correct_lower = correct_answer.lower().strip()
    cands = [c for c in cands if c.lower().strip() != correct_lower]
    if not cands:
        return []

    feats = np.stack([
        distractor_features(c, correct_answer, passage, vec, glove, idf_lookup)
        for c in cands
    ])
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


# ─── B3: GloVe nearest neighbour distractor alternative (spec §5.4) ────────
def generate_distractors_glove(correct_answer, glove, passage="", top_k=3):
    """
    Pre trained GloVe nearest neighbours of the correct answer's content
    words. Filters out words that appear in the passage (per spec: must be
    NOT extractable from the passage) and stop word noise.
    """
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
    Spec §5.3.2 — extractive hint generation. Each passage sentence is
    scored by TF·IDF cosine similarity to the question; top K are surfaced
    as graduated hints (most relevant first).
    """
    sentences = split_sentences(passage)
    if not sentences:
        return []
    sent_vecs = vec.transform(sentences)
    q_vec     = vec.transform([question])
    q_norm    = float(np.sqrt(q_vec.multiply(q_vec).sum())) + EPS
    s_norms   = np.sqrt(np.asarray(sent_vecs.multiply(sent_vecs).sum(axis=1)).ravel()) + EPS
    sims      = np.asarray((sent_vecs @ q_vec.T).todense()).ravel() / (s_norms * q_norm)
    order     = np.argsort(-sims)[:top_k]
    return [(sentences[int(i)], float(sims[int(i)])) for i in order]


# ─── B5: Hint extraction — ML scored variant ───────────────────────────────
def hint_features(sentence, position, question, correct_answer,
                  vec, glove=None, idf_lookup=None):
    """
    Five sentence features for the hint ranker:
      0  cos_q      : TF·IDF cosine vs question
      1  cos_a      : TF·IDF cosine vs correct answer
      2  position   : normalised sentence index in passage [0, 1]
      3  n_tokens   : sentence length in tokens
      4  cos_q_glv  : GloVe cosine vs question (0 if GloVe unavailable)
    """
    s_v = vec.transform([sentence])
    q_v = vec.transform([question])
    a_v = vec.transform([correct_answer])

    s_n = float(np.sqrt(s_v.multiply(s_v).sum())) + EPS
    q_n = float(np.sqrt(q_v.multiply(q_v).sum())) + EPS
    a_n = float(np.sqrt(a_v.multiply(a_v).sum())) + EPS

    cos_q = float(np.asarray((s_v @ q_v.T).todense()).ravel()[0]) / (s_n * q_n)
    cos_a = float(np.asarray((s_v @ a_v.T).todense()).ravel()[0]) / (s_n * a_n)

    n_tokens = float(len(sentence.split()))

    if glove is not None:
        dim = glove.vector_size
        sg = text_to_glove_vec(sentence, glove, dim, idf_lookup)
        qg = text_to_glove_vec(question, glove, dim, idf_lookup)
        sgn = float(np.linalg.norm(sg)) + EPS
        qgn = float(np.linalg.norm(qg)) + EPS
        cos_q_glv = float((sg @ qg) / (sgn * qgn))
    else:
        cos_q_glv = 0.0

    return np.array([cos_q, cos_a, position, n_tokens, cos_q_glv], dtype=np.float32)


def build_hint_train_data(df, vec, glove, idf_lookup,
                          max_rows=TRAIN_SAMPLE_SIZE):
    """
    Train the hint ranker on the most overlap with answer sentence as
    positive (the "gold key sentence"); 3 random other sentences as
    negatives per row.
    """
    X, y = [], []
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

        n = len(sentences)
        for idx, label in [(key_idx, 1)] + [(int(i), 0) for i in negs]:
            pos_norm = idx / max(1, n - 1)
            X.append(hint_features(
                sentences[idx], pos_norm, question, correct_ans,
                vec, glove, idf_lookup,
            ))
            y.append(label)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def train_hint_ranker(train_df, vec, glove, idf_lookup):
    print("\n[B5-RANKER] Building hint training data "
          f"(up to {TRAIN_SAMPLE_SIZE:,} rows)...")
    X, y = build_hint_train_data(train_df, vec, glove, idf_lookup)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    print(f"  Examples: {len(X):,}  ({n_pos:,} key sentences, {n_neg:,} non key)")
    print("[B5-RANKER] Training Logistic Regression...")
    model = LogisticRegression(C=1.0, max_iter=400, n_jobs=-1, random_state=42).fit(X, y)
    save_model(model, MODEL_B_HINT_PATH)
    return model


def extract_hints_ml(passage, question, correct_answer, ranker,
                     vec, glove, idf_lookup, top_k=3):
    sentences = split_sentences(passage)
    if not sentences:
        return []
    n = len(sentences)
    feats = np.stack([
        hint_features(s, i / max(1, n - 1), question, correct_answer,
                      vec, glove, idf_lookup)
        for i, s in enumerate(sentences)
    ])
    if hasattr(ranker, 'predict_proba'):
        scores = ranker.predict_proba(feats)[:, 1]
    else:
        scores = ranker.decision_function(feats)
    order = np.argsort(-scores)[:top_k]
    return [(sentences[int(i)], float(scores[int(i)])) for i in order]


# ─── Evaluation ──────────────────────────────────────────────────────────────
def _derive_gold_key_sentence(passage, correct_answer):
    """Heuristic gold key sentence: passage sentence with max overlap with correct answer."""
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


def evaluate_distractors(df, ranker, vec, glove, idf_lookup,
                         max_rows=EVAL_SAMPLE_SIZE, label="VAL"):
    """
    Score each generated distractor against its closest gold (greedy match
    by ROUGE 1). A pair counts as a "match" when ROUGE 1 exceeds 0.3.
    Returns averaged BLEU/ROUGE/METEOR + Precision/Recall/F1.
    """
    metrics = GenerationMetrics()
    keys = ['bleu_1', 'bleu', 'rouge1', 'rougeL', 'meteor']
    sums = {k: 0.0 for k in keys}

    n_rows, n_pairs = 0, 0
    n_matched = 0
    n_generated = 0
    n_gold = 0
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

        # Greedy: each generated paired with closest gold by ROUGE 1
        for g in gen[:3]:
            best_gold = max(gold, key=lambda x: metrics.score(g, x)['rouge1'])
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

        if (i + 1) % 1000 == 0:
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
    avg['n_rows']        = n_rows
    return avg, examples


def evaluate_hints(df, vec, top_k=3, use_ml=False, hint_ranker=None,
                   glove=None, idf_lookup=None,
                   max_rows=EVAL_SAMPLE_SIZE, label="VAL"):
    """
    Precision @ K against the derived gold key sentence; BLEU / ROUGE /
    METEOR comparing the top one hint to the gold key sentence.
    """
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
        if (i + 1) % 1000 == 0:
            print(f"    [{label}] scored {i + 1:,}/{min(max_rows, len(df)):,}")

    avg = {k: sums[k] / max(1, n_rows) for k in keys}
    avg['bleu_1_corpus']  = metrics.corpus_bleu_at_n(1)
    avg['bleu_corpus']    = metrics.corpus_bleu_at_n(4)
    avg['precision_at_k'] = n_at_k / max(1, n_rows)
    avg['n_rows']         = n_rows
    return avg


# ─── Final summary ───────────────────────────────────────────────────────────
def print_summary(distractor_results, hint_results, glove_distractor_examples=None):
    print("\n" + "=" * 122)
    print("FINAL COMPARISON  —  Model B  (Distractor + Hint Generation)")
    print("=" * 122)

    print("\n  Distractor Generation:")
    print(f"  {'Ranker':<8} {'Split':<6} {'N':>5}   "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}   "
          f"{'BLEU-1':>8} {'BLEU-1-c':>9} {'ROUGE-1':>8} {'ROUGE-L':>8} {'METEOR':>8}")
    print("  " + "-" * 110)
    for name, splits in distractor_results.items():
        for split_name in ('val', 'test'):
            if split_name not in splits:
                continue
            m = splits[split_name]
            print(f"  {name:<8} {split_name.upper():<6} {m['n_rows']:>5,}   "
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

    if glove_distractor_examples:
        print("\n  GloVe Nearest Neighbour Distractors (spec §5.4 alternative):")
        for ex in glove_distractor_examples[:3]:
            print(f"    Correct : {ex['correct']}")
            print(f"    Gold    : {ex['gold']}")
            print(f"    GloVe NN: {ex['glove_nn']}")
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

    # Sample LR distractors for eyeballing
    if 'LR' in distractor_examples_by_ranker:
        print("\n[Distractors] Sample LR generations (val):")
        for ex in distractor_examples_by_ranker['LR'][:3]:
            print(f"  Correct : {ex['correct']}")
            print(f"  Gold    : {ex['gold_distractors']}")
            print(f"  Gen (LR): {ex['gen_distractors']}")
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

    # ─── Final summary ────────────────────────────────────────────────────
    print_summary(distractor_results, hint_results, glove_examples)


if __name__ == "__main__":
    main()
