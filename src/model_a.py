"""
Model A — Consolidated Implementation (single-file orchestrator)

Runs every Model A approach the project requires, on a *single load of data*,
and prints one comparison table at the end.

What runs (in order):
  A1  Logistic Regression       — answer verification, per-option scoring
  A2  Linear SVM                — answer verification, per-option scoring
  A3  Naive Bayes               — question type classification (per spec §4.3)
  A4  K-Means                   — unsupervised clustering (per spec §4.2.2)
  A5  Soft Voting Ensemble      — average of LR + SVM, argmax (per spec §4.4)

Why one file
------------
- Loads CSVs and the one-hot .npz once, builds features once, then runs all
  five approaches on the cached tensors — no redundant disk I/O.
- Comparable apples-to-apples: LR/SVM/Ensemble use identical feature input,
  K-Means uses the same one-hot data, NB uses the question text.
- One end-of-run summary table reports accuracy + macro-F1 (per spec §4.5).

Feature engineering — pushing the classical-ML ceiling
------------------------------------------------------
Per option (~38 features):
  22  one-hot/IDF lexical overlaps + comparative rank features    (existing)
  8   TF-IDF cosine: passage, question, comparatives, divergence  (existing)
  4   sentence-level TF-IDF cosine: max + top3-mean + comparatives (NEW)
  3   length: option chars, log, vs-others-max                     (NEW)
  1   numeric token match passage∩option                           (NEW)

Sentence-level matching is the single biggest classical lift on RACE-style
MCQA: instead of comparing each option to the *whole* passage (where signal
gets diluted), we find the best-matching sentence in the passage. That's
the same intuition behind sliding-window baselines, but with linguistic
boundaries instead of fixed-stride windows.

Realistic accuracy targets:
  Random:                                          25%
  Old concat formulation (broken):                 29%
  Per-option with 30 features (one-hot + TF-IDF):  34%
  Per-option with 38 features (this file):        ~38–43%   ← target
  + GloVe embeddings (optional, see notes):       ~45–55%

CPU-only, classical ML. No neural networks.
Usage: python src/model_a.py
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings

# Optional libraries for Part 2 (text-generation evaluation: BLEU/ROUGE/METEOR).
# If absent the verification pipeline still runs; only generation is skipped.
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_LR_PATH,
    MODEL_SVM_PATH,
    MODEL_KMEANS_PATH,
    MODEL_ENSEMBLE_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TEST_CSV_PATH,
    ANSWER_MAP,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    MODELS_DIR,
)
from utils import save_model


# ─── Hyperparams ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 5000
N_EPOCHS     = 6
EPS          = 1e-6
TFIDF_VOCAB  = 20000
TFIDF_NGRAM  = (1, 2)
KMEANS_N     = 4

MODEL_NB_PATH      = os.path.join(MODELS_DIR, "model_a_naive_bayes.pkl")
MODEL_NB_VEC_PATH  = os.path.join(MODELS_DIR, "model_a_naive_bayes_vectorizer.pkl")
MODEL_RF_PATH      = os.path.join(MODELS_DIR, "model_a_random_forest.pkl")
MODEL_STACKING_PATH = os.path.join(MODELS_DIR, "model_a_stacking_meta.pkl")
TFIDF_VEC_PATH     = os.path.join(MODELS_DIR, "model_a_tfidf_vectorizer.pkl")


# ─── Shared helpers ───────────────────────────────────────────────────────────
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
NUM_RE     = re.compile(r'\d+')


def split_sentences(text):
    """Regex-based sentence splitter — no NLTK dependency."""
    return [s.strip() for s in SENT_SPLIT.split(str(text)) if s.strip()]


def vs_others(arr):
    """
    Comparative features within a (N, 4) group: arr−max_other, arr−mean_other,
    is_argmax. Length-invariant rank-within-question signal.
    """
    max_other  = np.zeros_like(arr)
    mean_other = np.zeros_like(arr)
    for j in range(4):
        others = np.delete(arr, j, axis=1)
        max_other[:, j]  = others.max(axis=1)
        mean_other[:, j] = others.mean(axis=1)
    is_argmax = (arr == arr.max(axis=1, keepdims=True)).astype(np.float32)
    return arr - max_other, arr - mean_other, is_argmax


def row_cosine(A, B):
    """Row-wise cosine for two sparse matrices of equal shape (N, V) → (N,)."""
    a_norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    b_norm = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
    dot    = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return (dot / (a_norm * b_norm + EPS)).astype(np.float32)


def normalize_per_question(scores):
    """Z-score 4 option scores within each question — used for ensembling."""
    mean = scores.mean(axis=1, keepdims=True)
    std  = scores.std(axis=1, keepdims=True) + 1e-9
    return (scores - mean) / std


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    print("\n[DATA] Loading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df   = pd.read_csv(VAL_CSV_PATH)
    train_y  = train_df['answer'].map(ANSWER_MAP).values.astype(np.int32)
    val_y    = val_df['answer'].map(ANSWER_MAP).values.astype(np.int32)
    print(f"  Train: {len(train_df):,} questions")
    print(f"  Val:   {len(val_df):,} questions")
    return train_df, val_df, train_y, val_y


# ─── IDF over the one-hot training corpus ────────────────────────────────────
def compute_idf(features_path):
    data = np.load(features_path, allow_pickle=True)
    p, q, o = data['passages'], data['questions'], data['options']
    df_count = (p.astype(np.float32).sum(axis=0)
                + q.astype(np.float32).sum(axis=0)
                + o.astype(np.float32).sum(axis=(0, 1)))
    n_docs = p.shape[0] + q.shape[0] + 4 * o.shape[0]
    return (np.log((n_docs + 1) / (df_count + 1)) + 1).astype(np.float32)


# ─── TF-IDF vectorizer ────────────────────────────────────────────────────────
def fit_tfidf_vectorizer(csv_path):
    print("\n[TF-IDF] Fitting vectorizer on training text...")
    df = pd.read_csv(csv_path)
    docs = (df['article'].fillna('').astype(str)  + ' ' +
            df['question'].fillna('').astype(str) + ' ' +
            df['A'].fillna('').astype(str) + ' ' +
            df['B'].fillna('').astype(str) + ' ' +
            df['C'].fillna('').astype(str) + ' ' +
            df['D'].fillna('').astype(str)).tolist()
    vec = TfidfVectorizer(
        max_features=TFIDF_VOCAB,
        ngram_range=TFIDF_NGRAM,
        sublinear_tf=True,
        stop_words='english',
        min_df=2, max_df=0.95,
        strip_accents='unicode',
        analyzer='word',
    )
    vec.fit(docs)
    print(f"  vocab size: {len(vec.vocabulary_):,}")
    return vec


# ─── Feature builders ─────────────────────────────────────────────────────────
def onehot_features(passages, questions, options, idf):
    """22 features per option from one-hot indicators + IDF."""
    passages  = passages.astype(np.float32, copy=False)
    questions = questions.astype(np.float32, copy=False)
    options   = options.astype(np.float32, copy=False)

    N = passages.shape[0]
    p_len = passages.sum(axis=1)
    pq    = passages * questions
    p_idf, q_idf, pq_idf = passages * idf, questions * idf, pq * idf

    p_overlap      = np.zeros((N, 4), dtype=np.float32)
    q_overlap      = np.zeros((N, 4), dtype=np.float32)
    pq_overlap     = np.zeros((N, 4), dtype=np.float32)
    opt_len        = np.zeros((N, 4), dtype=np.float32)
    p_overlap_idf  = np.zeros((N, 4), dtype=np.float32)
    q_overlap_idf  = np.zeros((N, 4), dtype=np.float32)
    pq_overlap_idf = np.zeros((N, 4), dtype=np.float32)

    for j in range(4):
        opt_j = options[:, j, :]
        p_overlap[:, j]      = (passages  * opt_j).sum(axis=1)
        q_overlap[:, j]      = (questions * opt_j).sum(axis=1)
        pq_overlap[:, j]     = (pq        * opt_j).sum(axis=1)
        opt_len[:, j]        = opt_j.sum(axis=1)
        p_overlap_idf[:, j]  = (p_idf     * opt_j).sum(axis=1)
        q_overlap_idf[:, j]  = (q_idf     * opt_j).sum(axis=1)
        pq_overlap_idf[:, j] = (pq_idf    * opt_j).sum(axis=1)

    p_overlap_ratio_o = p_overlap / (opt_len + EPS)
    q_overlap_ratio_o = q_overlap / (opt_len + EPS)
    p_overlap_ratio_p = p_overlap / (p_len[:, None] + EPS)

    p_diff_max,     p_diff_mean,  p_is_argmax  = vs_others(p_overlap)
    q_diff_max,     q_diff_mean,  q_is_argmax  = vs_others(q_overlap)
    p_idf_diff_max, _,            p_idf_argmax = vs_others(p_overlap_idf)
    q_idf_diff_max, _,            q_idf_argmax = vs_others(q_overlap_idf)
    pratio_diff_max, _,           _            = vs_others(p_overlap_ratio_o)

    return np.stack([
        p_overlap, p_overlap_ratio_o, p_overlap_ratio_p,
        q_overlap, q_overlap_ratio_o,
        pq_overlap,
        opt_len, np.log1p(opt_len),
        p_overlap_idf, q_overlap_idf, pq_overlap_idf,
        p_diff_max, p_diff_mean, p_is_argmax,
        q_diff_max, q_diff_mean, q_is_argmax,
        pratio_diff_max,
        p_idf_diff_max, p_idf_argmax,
        q_idf_diff_max, q_idf_argmax,
    ], axis=2)  # (N, 4, 22)


def tfidf_features(df_chunk, vec):
    """8 TF-IDF cosine features per option."""
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    qs   = df_chunk['question'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    art_vec = vec.transform(arts)
    q_vec   = vec.transform(qs)

    N = len(arts)
    p_cos = np.zeros((N, 4), dtype=np.float32)
    q_cos = np.zeros((N, 4), dtype=np.float32)
    for j in range(4):
        opt_vec = vec.transform(opts[j])
        p_cos[:, j] = row_cosine(art_vec, opt_vec)
        q_cos[:, j] = row_cosine(q_vec,   opt_vec)

    p_diff_max, _, p_argmax = vs_others(p_cos)
    q_diff_max, _, q_argmax = vs_others(q_cos)

    return np.stack([
        p_cos, q_cos,
        p_diff_max, p_argmax,
        q_diff_max, q_argmax,
        p_cos - q_cos,
        np.abs(p_cos - q_cos),
    ], axis=2)  # (N, 4, 8)


def sentence_features(df_chunk, vec):
    """
    4 sentence-level features per option:
      s_max         — max cosine over passage sentences (the answer-bearing
                      sentence usually scores high; whole-passage cosine
                      averages this signal away)
      s_top3_mean   — mean of top-3 sentence cosines (answers spanning
                      multiple sentences)
      s_max_diff    — s_max minus max-of-other-options' s_max
      s_top3_diff   — s_top3_mean minus max-of-other-options' s_top3_mean
    """
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    N = len(arts)

    # Pre-tokenize all sentences and stack into one big list. Lets us call
    # vec.transform() once on the whole batch (fast) instead of per-question.
    all_sents = []
    sent_ranges = []
    for art in arts:
        s = split_sentences(art)
        if not s:
            s = [art if art else " "]
        sent_ranges.append((len(all_sents), len(all_sents) + len(s)))
        all_sents.extend(s)

    all_sents_vec = vec.transform(all_sents)
    sent_norms    = np.sqrt(np.asarray(all_sents_vec.multiply(all_sents_vec)
                                                    .sum(axis=1)).ravel())

    opts_vecs = [vec.transform(opts[j]) for j in range(4)]
    opt_norms = [np.sqrt(np.asarray(ov.multiply(ov).sum(axis=1)).ravel())
                 for ov in opts_vecs]

    s_max  = np.zeros((N, 4), dtype=np.float32)
    s_top3 = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        s, e = sent_ranges[i]
        if s == e:
            continue
        sv = all_sents_vec[s:e]
        sn = sent_norms[s:e]

        for j in range(4):
            on = opt_norms[j][i]
            if on < 1e-9 or len(sn) == 0:
                continue
            ov = opts_vecs[j][i:i + 1]
            # sv @ ov.T returns a sparse matrix; np.asarray doesn't densify
            # sparse matrices (unlike numpy.matrix). Use .toarray() instead.
            dots = (sv @ ov.T).toarray().ravel()
            cos  = dots / (sn * on + EPS)
            s_max[i, j]  = cos.max()
            top3         = np.sort(cos)[-3:]
            s_top3[i, j] = top3.mean() if len(top3) else 0.0

    s_max_diff_max,  _, _ = vs_others(s_max)
    s_top3_diff_max, _, _ = vs_others(s_top3)

    return np.stack([s_max, s_top3, s_max_diff_max, s_top3_diff_max], axis=2)


def question_aware_features(df_chunk, vec, top_k=3):
    """
    Question-conditioned passage attention — the missing classical-ML signal.

    For each question, instead of comparing options to the whole passage:
      1. Find the top-K passage sentences most similar to the question
         (the "answer-bearing region")
      2. Aggregate those sentences into a single TF-IDF vector
      3. Compute option↔region cosine
    Then we know the option matches the *relevant* part of the passage,
    not just any part. Whole-passage cosine averages over 15-20 sentences
    and drowns the actual signal — this filters first, matches second.

    Returns 3 features per option: qa_cos, qa_diff_max, qa_argmax.
    """
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    qs   = df_chunk['question'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    N = len(arts)

    all_sents = []
    sent_ranges = []
    for art in arts:
        s = split_sentences(art)
        if not s:
            s = [art if art else " "]
        sent_ranges.append((len(all_sents), len(all_sents) + len(s)))
        all_sents.extend(s)

    all_sents_vec = vec.transform(all_sents)
    sent_norms    = np.sqrt(np.asarray(all_sents_vec.multiply(all_sents_vec)
                                                    .sum(axis=1)).ravel())

    q_vec   = vec.transform(qs)
    q_norms = np.sqrt(np.asarray(q_vec.multiply(q_vec).sum(axis=1)).ravel())

    opts_vecs = [vec.transform(opts[j]) for j in range(4)]
    opt_norms = [np.sqrt(np.asarray(ov.multiply(ov).sum(axis=1)).ravel())
                 for ov in opts_vecs]

    qa_cos = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        s, e = sent_ranges[i]
        if s == e or q_norms[i] < 1e-9:
            continue

        sv = all_sents_vec[s:e]
        sn = sent_norms[s:e]

        # 1. Score each sentence by similarity to the question
        q_sent_dots = (sv @ q_vec[i:i + 1].T).toarray().ravel()
        q_sent_cos  = q_sent_dots / (sn * q_norms[i] + EPS)

        # 2. Aggregate top-k sentences as the answer-bearing region
        k = min(top_k, len(q_sent_cos))
        top_idx = np.argsort(q_sent_cos)[-k:]
        # Sum the sparse rows → numpy.matrix → flatten to 1-D ndarray
        relevant_arr  = np.asarray(sv[top_idx].sum(axis=0)).ravel()
        relevant_norm = np.sqrt((relevant_arr ** 2).sum())
        if relevant_norm < 1e-9:
            continue

        # 3. Cosine each option against the answer-bearing region
        for j in range(4):
            on = opt_norms[j][i]
            if on < 1e-9:
                continue
            ov_dense = opts_vecs[j][i:i + 1].toarray().ravel()
            dot = float((relevant_arr * ov_dense).sum())
            qa_cos[i, j] = dot / (relevant_norm * on + EPS)

    qa_diff_max, _, qa_argmax = vs_others(qa_cos)
    return np.stack([qa_cos, qa_diff_max, qa_argmax], axis=2)  # (N, 4, 3)


def length_features(df_chunk):
    """3 length features per option: chars, log_chars, chars_diff_max."""
    opts = [df_chunk[c].fillna('').astype(str) for c in ['A', 'B', 'C', 'D']]
    N = len(opts[0])

    char_len = np.zeros((N, 4), dtype=np.float32)
    for j in range(4):
        char_len[:, j] = opts[j].str.len().values

    diff_max, _, _ = vs_others(char_len)
    return np.stack([char_len, np.log1p(char_len), diff_max], axis=2)


def numeric_features(df_chunk):
    """1 feature per option: 1 if option's numeric tokens overlap passage's."""
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    N = len(arts)
    num_match = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        passage_nums = set(NUM_RE.findall(arts[i]))
        if not passage_nums:
            continue
        for j in range(4):
            opt_nums = set(NUM_RE.findall(opts[j][i]))
            if opt_nums and (passage_nums & opt_nums):
                num_match[i, j] = 1.0
    return num_match[..., None]


# ─── GloVe semantic features (optional, per spec §5.4 / §7.1) ───────────────
def load_glove():
    """
    Load GloVe word vectors via gensim.downloader. Returns None if unavailable.

    Spec §5.4 explicitly allows pretrained Word2Vec features; §7.1 lists
    Gensim and sentence-transformers in the recommended stack. GloVe is the
    same class of static pretrained lookup table — both are "embed each word
    as a fixed vector" with no neural-network training at our end.

    The 50-dim glove-wiki-gigaword model is ~70 MB; gensim auto-downloads it
    to ~/gensim-data/ on first run and reuses the cache afterwards.
    """
    try:
        import gensim.downloader as api
    except ImportError:
        print("\n[GLOVE] gensim not installed — semantic features disabled.")
        print("  To enable: pip install gensim")
        return None
    try:
        print("\n[GLOVE] Loading glove-wiki-gigaword-50 via gensim...")
        model = api.load("glove-wiki-gigaword-50")
        print(f"  loaded {len(model):,} word vectors of dim {model.vector_size}")
        return model
    except Exception as e:
        print(f"\n[GLOVE] Load failed: {e}")
        print("  Continuing without GloVe features.")
        return None


_TOK_RE = re.compile(r'\b[a-z]+\b')


def build_idf_lookup(vec):
    """word → IDF mapping from a fitted TfidfVectorizer (unigrams only)."""
    return {
        term: float(vec.idf_[idx])
        for term, idx in vec.vocabulary_.items()
        if ' ' not in term  # skip bigrams; we look up single words in GloVe
    }


def text_to_glove_vec(text, glove, dim, idf_lookup=None):
    """
    IDF-weighted mean GloVe vector for a piece of text. Zero vector if no
    words match. IDF weighting is critical: without it, "the"/"a"/"is"
    dominate the average and wash out topical signal on long passages.
    """
    words = _TOK_RE.findall(text.lower())
    vecs    = []
    weights = []
    for w in words:
        if w in glove:
            vecs.append(glove[w])
            weights.append(idf_lookup.get(w, 1.0) if idf_lookup else 1.0)
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    V = np.stack(vecs).astype(np.float32)
    W = np.asarray(weights, dtype=np.float32)
    W = W / (W.sum() + EPS)
    return (V * W[:, None]).sum(axis=0)


def _l2_norm_rows(X):
    return np.linalg.norm(X, axis=1) + EPS


def glove_features(df_chunk, glove, idf_lookup=None):
    """
    6 GloVe semantic-similarity features per option. Catches paraphrase and
    synonym overlap that lexical TF-IDF misses (e.g. "automobile" ≈ "car",
    "purchased" ≈ "bought") — the gap between ~34% and ~50% on RACE.

      p_cos       — cosine(mean_glove(passage),  mean_glove(option))
      q_cos       — cosine(mean_glove(question), mean_glove(option))
      p_diff_max  — p_cos minus max-of-other-options' p_cos
      p_argmax    — binary flag: this option wins on p_cos
      q_diff_max  — q_cos minus max-of-other-options' q_cos
      q_argmax    — binary flag: this option wins on q_cos
    """
    dim = glove.vector_size
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    qs   = df_chunk['question'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    N = len(arts)
    art_vecs = np.stack([text_to_glove_vec(t, glove, dim, idf_lookup) for t in arts])
    q_vecs   = np.stack([text_to_glove_vec(t, glove, dim, idf_lookup) for t in qs])
    opt_vecs = [
        np.stack([text_to_glove_vec(t, glove, dim, idf_lookup) for t in opts[j]])
        for j in range(4)
    ]

    def cos_rows(A, B):
        an = np.linalg.norm(A, axis=1) + EPS
        bn = np.linalg.norm(B, axis=1) + EPS
        return ((A * B).sum(axis=1) / (an * bn)).astype(np.float32)

    p_cos = np.zeros((N, 4), dtype=np.float32)
    q_cos = np.zeros((N, 4), dtype=np.float32)
    for j in range(4):
        p_cos[:, j] = cos_rows(art_vecs, opt_vecs[j])
        q_cos[:, j] = cos_rows(q_vecs,   opt_vecs[j])

    p_diff_max, _, p_argmax = vs_others(p_cos)
    q_diff_max, _, q_argmax = vs_others(q_cos)
    return np.stack([p_cos, q_cos, p_diff_max, p_argmax, q_diff_max, q_argmax], axis=2)


def glove_sentence_features(df_chunk, glove, idf_lookup=None):
    """
    4 sentence-level GloVe features per option:
      s_max         — max cosine over passage sentences in GloVe space
                      (the answer-bearing sentence usually scores high)
      s_top3_mean   — mean of top-3 sentence cosines
      s_max_diff    — comparative vs other options
      s_max_argmax  — binary flag: this option wins on sentence-max
    Sentence-level GloVe is sharper than whole-passage GloVe because a
    15-word sentence-mean preserves topical signal that a 300-word
    passage-mean averages away.
    """
    dim = glove.vector_size
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    N = len(arts)
    s_max  = np.zeros((N, 4), dtype=np.float32)
    s_top3 = np.zeros((N, 4), dtype=np.float32)

    # Pre-compute option vectors once per (i, j)
    opt_vecs = [
        np.stack([text_to_glove_vec(t, glove, dim, idf_lookup) for t in opts[j]])
        for j in range(4)
    ]
    opt_norms = [_l2_norm_rows(ov) for ov in opt_vecs]

    for i in range(N):
        sents = split_sentences(arts[i])
        if not sents:
            continue
        sent_vecs  = np.stack([text_to_glove_vec(s, glove, dim, idf_lookup) for s in sents])
        sent_norms = _l2_norm_rows(sent_vecs)

        for j in range(4):
            ov_norm = opt_norms[j][i]
            if ov_norm < 2 * EPS:
                continue
            cos = (sent_vecs @ opt_vecs[j][i]) / (sent_norms * ov_norm)
            s_max[i, j] = cos.max()
            top3 = np.sort(cos)[-3:]
            s_top3[i, j] = top3.mean() if len(top3) else 0.0

    s_max_diff, _, s_max_argmax = vs_others(s_max)
    return np.stack([s_max, s_top3, s_max_diff, s_max_argmax], axis=2)


def glove_question_aware_features(df_chunk, glove, idf_lookup=None, top_k=3):
    """
    3 question-aware GloVe features per option (semantic analog of
    question_aware_features). For each question:
      1. Find the top-K passage sentences most similar to the question
         in GloVe space
      2. Aggregate them as the "answer-bearing region" (mean GloVe vec)
      3. Cosine each option against that region
    Returns: qa_glove_cos, qa_glove_diff_max, qa_glove_argmax.
    """
    dim = glove.vector_size
    arts = df_chunk['article'].fillna('').astype(str).tolist()
    qs   = df_chunk['question'].fillna('').astype(str).tolist()
    opts = [df_chunk[c].fillna('').astype(str).tolist() for c in ['A', 'B', 'C', 'D']]

    N = len(arts)
    qa_cos = np.zeros((N, 4), dtype=np.float32)

    opt_vecs = [
        np.stack([text_to_glove_vec(t, glove, dim, idf_lookup) for t in opts[j]])
        for j in range(4)
    ]
    opt_norms = [_l2_norm_rows(ov) for ov in opt_vecs]

    for i in range(N):
        sents = split_sentences(arts[i])
        if not sents:
            continue
        sent_vecs  = np.stack([text_to_glove_vec(s, glove, dim, idf_lookup) for s in sents])
        sent_norms = _l2_norm_rows(sent_vecs)

        q_vec  = text_to_glove_vec(qs[i], glove, dim, idf_lookup)
        q_norm = float(np.linalg.norm(q_vec)) + EPS
        if q_norm < 2 * EPS:
            continue

        # Top-K sentences by similarity to the question
        q_sent_cos = (sent_vecs @ q_vec) / (sent_norms * q_norm)
        k = min(top_k, len(q_sent_cos))
        top_idx = np.argsort(q_sent_cos)[-k:]
        relevant_vec  = sent_vecs[top_idx].mean(axis=0)
        relevant_norm = float(np.linalg.norm(relevant_vec)) + EPS
        if relevant_norm < 2 * EPS:
            continue

        for j in range(4):
            ov_norm = opt_norms[j][i]
            if ov_norm < 2 * EPS:
                continue
            qa_cos[i, j] = float(relevant_vec @ opt_vecs[j][i]) / (relevant_norm * ov_norm)

    qa_diff, _, qa_argmax = vs_others(qa_cos)
    return np.stack([qa_cos, qa_diff, qa_argmax], axis=2)


def build_features(npz_path, df, idf, vec, glove=None, idf_lookup=None, label="data"):
    """
    Build the full per-option feature tensor (~38 dims) for a split.
    Streams the .npz batch-by-batch; output tensor stays in RAM.
    """
    print(f"\n[FEATURES] Building per-option features for {label}...")
    data = np.load(npz_path, allow_pickle=True)
    passages, questions, options = data['passages'], data['questions'], data['options']

    N = len(df)
    chunks = []
    for s in range(0, N, BATCH_SIZE):
        e = min(s + BATCH_SIZE, N)
        batch_df = df.iloc[s:e]
        oh = onehot_features(passages[s:e], questions[s:e], options[s:e], idf)
        tf = tfidf_features(batch_df, vec)
        sn = sentence_features(batch_df, vec)
        qa = question_aware_features(batch_df, vec)
        ln = length_features(batch_df)
        nm = numeric_features(batch_df)
        parts = [oh, tf, sn, qa, ln, nm]
        if glove is not None:
            parts.append(glove_features(batch_df, glove, idf_lookup))
            parts.append(glove_sentence_features(batch_df, glove, idf_lookup))
            parts.append(glove_question_aware_features(batch_df, glove, idf_lookup))
        chunks.append(np.concatenate(parts, axis=2))
        print(f"  [{e}/{N}]")
    feats = np.concatenate(chunks, axis=0)
    print(f"  shape: {feats.shape}  ({feats.nbytes / 1e6:.1f} MB)")
    return feats


# ─── Per-option scorer wrapper ────────────────────────────────────────────────
class MCOptionScorer:
    """Bundles scaler + classifier (+ idf, vectorizer) for inference."""
    def __init__(self, scaler, classifier, idf=None, vectorizer=None):
        self.scaler     = scaler
        self.classifier = classifier
        self.idf        = idf
        self.vectorizer = vectorizer

    def _scores(self, X_per_option):
        N, _, F = X_per_option.shape
        X = self.scaler.transform(X_per_option.reshape(N * 4, F))
        return self.classifier.decision_function(X).reshape(N, 4)

    def predict(self, X_per_option):
        return self._scores(X_per_option).argmax(axis=1)

    def decision_function(self, X_per_option):
        return self._scores(X_per_option)


# ─── Shared per-option SGD trainer (used by LR + SVM) ────────────────────────
def eval_per_option_metrics(scores, y):
    """
    Compute (EM, EM-F1, BinAcc, BinF1) given (N, 4) per-option scores and
    (N,) labels. Used for both val and test reporting — keeps the metric
    code identical across the two splits so the columns are apples-to-apples.
    """
    preds = scores.argmax(axis=1)
    em    = accuracy_score(y, preds)
    em_f1 = f1_score(y, preds, average='macro')
    y_bin = np.zeros(len(y) * 4, dtype=np.int32)
    y_bin[np.arange(len(y)) * 4 + y] = 1
    bin_preds = (scores.flatten() > 0).astype(np.int32)
    bin_acc   = accuracy_score(y_bin, bin_preds)
    bin_f1    = f1_score(y_bin, bin_preds, average='macro')
    return em, em_f1, bin_acc, bin_f1


def train_per_option(model, train_X, train_y_raw, val_X, val_y, name,
                     n_epochs=N_EPOCHS):
    N, _, F = train_X.shape

    print(f"\n[{name}] Fitting StandardScaler...")
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_X.reshape(N * 4, F))
    val_flat   = scaler.transform(val_X.reshape(-1, F))

    train_y_bin = np.zeros(N * 4, dtype=np.int32)
    train_y_bin[np.arange(N) * 4 + train_y_raw] = 1

    rng = np.random.default_rng(42)
    sgd_batch = BATCH_SIZE * 4

    print(f"[{name}] Training {n_epochs} epochs × {N * 4:,} option-examples")
    # Track best epoch and restore that model state at the end. Hinge-loss
    # SGD can diverge late in training (we saw SVM crash from 33% → 23% at
    # epoch 6). Best-epoch restore makes the training procedure robust to
    # this kind of late-epoch overshoot for both LR and SVM.
    best_acc = -1.0
    best_coef = None
    best_intercept = None
    for epoch in range(n_epochs):
        perm = rng.permutation(N * 4)
        for s in range(0, N * 4, sgd_batch):
            idx = perm[s:s + sgd_batch]
            model.partial_fit(train_flat[idx], train_y_bin[idx],
                              classes=np.array([0, 1]))
        scores = model.decision_function(val_flat).reshape(-1, 4)
        preds  = scores.argmax(axis=1)
        acc    = accuracy_score(val_y, preds)
        marker = ""
        if acc > best_acc:
            best_acc       = acc
            best_coef      = model.coef_.copy()
            best_intercept = model.intercept_.copy()
            marker = "  ← best"
        print(f"  Epoch {epoch + 1}/{n_epochs}: val acc = {acc * 100:.2f}%{marker}")

    # Restore best epoch's weights
    if best_coef is not None:
        model.coef_      = best_coef
        model.intercept_ = best_intercept

    val_scores = model.decision_function(val_flat).reshape(-1, 4)
    val_preds  = val_scores.argmax(axis=1)

    # Per-question metrics — Exact Match (EM) is the spec-defined "did the
    # model pick the right A/B/C/D for this question" metric (§4.5).
    em      = accuracy_score(val_y, val_preds)
    em_f1   = f1_score(val_y, val_preds, average='macro')

    # Binary metrics — apples-to-apples with peers using the (article, question,
    # option) → 0/1 binary-pivot framing. Threshold the decision function at 0.
    # NOTE: binary "accuracy" on a 1:3 imbalanced task has a 75% no-skill
    # baseline (always-predict-wrong), so high binary acc ≠ better answer
    # picking. EM is the real metric.
    val_y_binary = np.zeros(len(val_y) * 4, dtype=np.int32)
    val_y_binary[np.arange(len(val_y)) * 4 + val_y] = 1
    binary_preds = (val_scores.flatten() > 0).astype(np.int32)
    binary_acc   = accuracy_score(val_y_binary, binary_preds)
    binary_f1    = f1_score(val_y_binary, binary_preds, average='macro')

    # Re-bind for the reporting block below
    val_acc = em
    val_f1  = em_f1

    print(f"\n[{name}] Final metrics:")
    print(f"  Per-question (EM, 4-way):     {em * 100:6.2f}%   macro-F1 = {em_f1:.4f}")
    print(f"  Binary  (peer-comparable):    {binary_acc * 100:6.2f}%   macro-F1 = {binary_f1:.4f}")
    print(classification_report(val_y, val_preds,
                                target_names=['A', 'B', 'C', 'D'], digits=4))

    return scaler, model, em, em_f1, val_scores, binary_acc, binary_f1


# ─── A1: Logistic Regression ──────────────────────────────────────────────────
def run_logistic_regression(train_X, train_y, val_X, val_y, idf, vec):
    print("\n" + "=" * 70)
    print("A1  Logistic Regression  —  Answer Verification")
    print("=" * 70)
    model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-4,
        max_iter=1, warm_start=True, random_state=42, n_jobs=-1,
        learning_rate='optimal', verbose=0,
    )
    scaler, model, em, em_f1, scores, bin_acc, bin_f1 = train_per_option(
        model, train_X, train_y, val_X, val_y, "LR",
    )
    scorer = MCOptionScorer(scaler, model, idf=idf, vectorizer=vec)
    save_model(scorer, MODEL_LR_PATH)
    return scorer, em, em_f1, scores, bin_acc, bin_f1


# ─── A2: Linear SVM ──────────────────────────────────────────────────────────
def run_linear_svm(train_X, train_y, val_X, val_y, idf, vec):
    print("\n" + "=" * 70)
    print("A2  Linear SVM  —  Answer Verification")
    print("=" * 70)
    # Slightly higher alpha than LR (5e-4 vs 1e-4) — hinge loss is more prone
    # to late-epoch overshoot than log_loss; extra regularisation tames it.
    # Best-epoch restore in train_per_option is the additional safety net.
    model = SGDClassifier(
        loss='hinge', penalty='l2', alpha=5e-4,
        max_iter=1, warm_start=True, random_state=42, n_jobs=-1,
        learning_rate='optimal', verbose=0,
    )
    scaler, model, em, em_f1, scores, bin_acc, bin_f1 = train_per_option(
        model, train_X, train_y, val_X, val_y, "SVM",
    )
    scorer = MCOptionScorer(scaler, model, idf=idf, vectorizer=vec)
    save_model(scorer, MODEL_SVM_PATH)
    return scorer, em, em_f1, scores, bin_acc, bin_f1


# ─── A3: Naive Bayes — Question Type Classification ──────────────────────────
QUESTION_TYPE_PATTERNS = [
    ('main_idea', [
        r'\bmain idea\b', r'\bbest title\b', r'\bmain point\b',
        r'\bprimarily about\b', r'\bmainly about\b', r'\bpurpose of\b',
        r'\bbest summari[sz]e\b', r'\bbest describes\b',
        r'\bwhat is the passage\b', r'\bwhat is this passage\b',
    ]),
    ('vocabulary', [
        r'\bword\b.*\bmean(s|ing)?\b', r'\bphrase\b.*\bmean(s|ing)?\b',
        r'\bunderlined word\b', r'\bunderlined phrase\b',
        r'\brefers? to\b', r'\bclosest in meaning\b',
        r'\bdefinition of\b', r'\bcould be replaced by\b',
        r'\bthe word .* means\b',
    ]),
    ('inference', [
        r'\bimpl(y|ies|ied)\b', r'\binfer(red)?\b', r'\bsuggest(s|ed)?\b',
        r'\bprobably\b', r'\bmost likely\b',
        r'\bauthor (thinks|believes|feels|implies)\b',
        r'\bwriter (thinks|believes|feels|implies)\b',
        r'\bwe can learn\b', r'\bwe can conclude\b',
        r'\battitude\b', r'\btone\b',
    ]),
    ('detail', [
        r'\baccording to\b', r'\bhow many\b', r'\bhow much\b', r'\bhow long\b',
        r'\bwho\b', r'\bwhen\b', r'\bwhere\b',
        r'\bwhat\b', r'\bwhich\b',
    ]),
]


def label_question(text):
    text = str(text).lower()
    for label, patterns in QUESTION_TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, text):
                return label
    return 'other'


# ─── A2.5: Random Forest (per-option binary, third base for ensemble) ──────
def run_random_forest(train_X, train_y_raw, val_X, val_y,
                      n_estimators=100, max_depth=20):
    """
    RandomForest as a third base classifier for the ensemble. Trees capture
    non-linear feature interactions that linear LR/SVM can't, so RF makes
    *different mistakes* — that decorrelation is what lets the ensemble
    actually exceed the best individual.

    Same per-option binary framing as LR/SVM:
      - Flatten (N, 4, F) → (N*4, F), build binary labels (1=correct option)
      - Train RF, get class-1 probabilities, reshape to (N, 4), argmax
    """
    print("\n" + "=" * 70)
    print("A2.5  Random Forest  —  Answer Verification (ensemble base)")
    print("=" * 70)

    N, _, F = train_X.shape
    train_flat = train_X.reshape(N * 4, F)
    val_flat   = val_X.reshape(-1, F)

    train_y_bin = np.zeros(N * 4, dtype=np.int32)
    train_y_bin[np.arange(N) * 4 + train_y_raw] = 1

    print(f"[RF] Training {n_estimators} trees, max_depth={max_depth}, "
          f"on {N * 4:,} option-examples × {F} features (n_jobs=-1)")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        class_weight='balanced',  # per-option binary task is 1:3 imbalanced
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_flat, train_y_bin)
    print("[RF] ✓ training complete")

    # P(correct=1) per option, reshape (N_val, 4), argmax
    val_probs  = model.predict_proba(val_flat)[:, 1]
    val_scores = val_probs.reshape(-1, 4)
    val_preds  = val_scores.argmax(axis=1)

    em    = accuracy_score(val_y, val_preds)
    em_f1 = f1_score(val_y, val_preds, average='macro')

    val_y_binary = np.zeros(len(val_y) * 4, dtype=np.int32)
    val_y_binary[np.arange(len(val_y)) * 4 + val_y] = 1
    binary_preds = (val_probs > 0.5).astype(np.int32)
    binary_acc   = accuracy_score(val_y_binary, binary_preds)
    binary_f1    = f1_score(val_y_binary, binary_preds, average='macro')

    print(f"\n[RF] Final metrics:")
    print(f"  Per-question (EM, 4-way):     {em * 100:6.2f}%   macro-F1 = {em_f1:.4f}")
    print(f"  Binary  (peer-comparable):    {binary_acc * 100:6.2f}%   macro-F1 = {binary_f1:.4f}")
    print(classification_report(val_y, val_preds,
                                target_names=['A', 'B', 'C', 'D'], digits=4))

    save_model(model, MODEL_RF_PATH)
    return em, em_f1, val_scores, binary_acc, binary_f1, model


def run_naive_bayes(train_df, val_df):
    print("\n" + "=" * 70)
    print("A3  Naive Bayes  —  Question Type Classification")
    print("=" * 70)

    train_qtype = train_df['question'].fillna('').apply(label_question).values
    val_qtype   = val_df['question'].fillna('').apply(label_question).values

    print(f"\n  Train label distribution:")
    for label, count in Counter(train_qtype).most_common():
        print(f"    {label:<12} {count:>6}  ({count / len(train_qtype) * 100:5.1f}%)")

    vec_nb = CountVectorizer(
        lowercase=True, ngram_range=(1, 2),
        min_df=2, max_df=0.95, stop_words=None,
    )
    X_train = vec_nb.fit_transform(train_df['question'].fillna('').astype(str))
    X_val   = vec_nb.transform(val_df['question'].fillna('').astype(str))

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, train_qtype)
    val_preds = nb.predict(X_val)
    acc = accuracy_score(val_qtype, val_preds)
    f1  = f1_score(val_qtype, val_preds, average='macro')

    print(f"\n[NB] Final: acc = {acc * 100:.2f}%, macro-F1 = {f1:.4f}")
    print(classification_report(val_qtype, val_preds, digits=4))

    save_model(nb, MODEL_NB_PATH)
    save_model(vec_nb, MODEL_NB_VEC_PATH)
    return acc, f1


# ─── A4: K-Means clustering ──────────────────────────────────────────────────
def kmeans_features(p, q, o):
    """30,008-dim aggregated feature for K-Means (passage|question|opts|overlaps)."""
    p = p.astype(np.float32, copy=False)
    q = q.astype(np.float32, copy=False)
    o = o.astype(np.float32, copy=False)
    base = np.hstack([p, q, o[:, 0, :], o[:, 1, :], o[:, 2, :], o[:, 3, :]])
    overlaps_p = np.stack([(p * o[:, i, :]).sum(axis=1) for i in range(4)], axis=1)
    overlaps_q = np.stack([(q * o[:, i, :]).sum(axis=1) for i in range(4)], axis=1)
    return np.hstack([base, overlaps_p, overlaps_q])


def run_kmeans(train_npz_path, val_npz_path, val_y):
    print("\n" + "=" * 70)
    print("A4  K-Means  —  Unsupervised Clustering")
    print("=" * 70)

    model = MiniBatchKMeans(
        n_clusters=KMEANS_N, batch_size=BATCH_SIZE,
        n_init=10, random_state=42, max_iter=300,
    )

    print("\n[K-Means] Streaming training data...")
    data = np.load(train_npz_path, allow_pickle=True)
    p, q, o = data['passages'], data['questions'], data['options']
    N = p.shape[0]
    for s in range(0, N, BATCH_SIZE):
        e = min(s + BATCH_SIZE, N)
        model.partial_fit(kmeans_features(p[s:e], q[s:e], o[s:e]))
        if (s // BATCH_SIZE + 1) % 3 == 0:
            print(f"  batch {s // BATCH_SIZE + 1} (inertia: {model.inertia_:.0f})")
    print(f"\n[K-Means] Final inertia: {model.inertia_:.0f}")

    # Predict val cluster assignments
    print("\n[K-Means] Predicting on validation...")
    data_v = np.load(val_npz_path, allow_pickle=True)
    pv, qv, ov = data_v['passages'], data_v['questions'], data_v['options']
    val_clusters = []
    for s in range(0, pv.shape[0], BATCH_SIZE):
        e = min(s + BATCH_SIZE, pv.shape[0])
        val_clusters.extend(model.predict(kmeans_features(pv[s:e], qv[s:e], ov[s:e])))
    val_clusters = np.array(val_clusters)

    # Cluster purity
    purity = 0
    for c in range(KMEANS_N):
        mask = (val_clusters == c)
        if mask.sum() > 0:
            purity += np.bincount(val_y[mask], minlength=4).max()
    purity /= len(val_y)

    print(f"\n[K-Means] Cluster purity: {purity * 100:.2f}%")

    save_model(model, MODEL_KMEANS_PATH)
    return purity


# ─── A5: Soft Voting Ensemble ────────────────────────────────────────────────
def run_ensemble(members, val_y):
    """
    Weighted soft-voting ensemble (per spec §4.4).

    members: list of dicts, each with:
      - 'name'   (str)         : display name (e.g. 'LR', 'SVM', 'RF')
      - 'scores' (N, 4) array  : per-option decision_function or proba scores
      - 'em'     (float)        : that member's standalone val EM, used as weight

    Why weighted:
      Equal-weight averaging (the classical 'naive' soft vote) lets weaker
      members drag down stronger ones. With weights ∝ (EM − random_baseline),
      a member at random gets zero vote and the strongest members dominate.

    Why z-normalise per-question:
      LR returns log-odds, SVM returns hinge margin, RF returns probabilities
      — all on different scales. Z-scoring the 4 option scores within each
      question puts every member on the same relative scale, so what matters
      is the *ranking* of options, not the raw magnitude.
    """
    print("\n" + "=" * 70)
    names = " + ".join(m['name'] for m in members)
    print(f"A5  Weighted Soft Voting Ensemble  —  {names}")
    print("=" * 70)

    # Weights: skill above random (EM − 0.25), clipped to ≥ 0
    raw_weights = np.array([max(0.0, m['em'] - 0.25) for m in members],
                           dtype=np.float64)
    if raw_weights.sum() < 1e-9:
        weights = np.ones(len(members)) / len(members)  # fallback: equal
    else:
        weights = raw_weights / raw_weights.sum()

    print("[Ensemble] Member weights (EM-skill above random, normalised):")
    for m, w in zip(members, weights):
        print(f"  {m['name']:<6} EM = {m['em'] * 100:5.2f}%  →  weight = {w:.4f}")

    # Z-normalise per question, then weighted average
    ens_scores = np.zeros_like(members[0]['scores'], dtype=np.float64)
    for m, w in zip(members, weights):
        ens_scores += w * normalize_per_question(m['scores'].astype(np.float64))
    ens_scores = ens_scores.astype(np.float32)

    ens_preds = ens_scores.argmax(axis=1)
    em        = accuracy_score(val_y, ens_preds)
    em_f1     = f1_score(val_y, ens_preds, average='macro')

    # Binary metrics on ensemble's z-normalised+weighted scores (threshold at 0)
    val_y_binary = np.zeros(len(val_y) * 4, dtype=np.int32)
    val_y_binary[np.arange(len(val_y)) * 4 + val_y] = 1
    binary_preds = (ens_scores.flatten() > 0).astype(np.int32)
    binary_acc   = accuracy_score(val_y_binary, binary_preds)
    binary_f1    = f1_score(val_y_binary, binary_preds, average='macro')

    print(f"\n[Ensemble] Final metrics:")
    print(f"  Per-question (EM, 4-way):     {em * 100:6.2f}%   macro-F1 = {em_f1:.4f}")
    print(f"  Binary  (peer-comparable):    {binary_acc * 100:6.2f}%   macro-F1 = {binary_f1:.4f}")
    print(classification_report(val_y, ens_preds,
                                target_names=['A', 'B', 'C', 'D'], digits=4))

    save_model(
        {
            'method': 'weighted_soft_voting',
            'components': [m['name'] for m in members],
            'weights': weights.tolist(),
        },
        MODEL_ENSEMBLE_PATH,
    )
    return em, em_f1, binary_acc, binary_f1


# ─── A6: Stacking ensemble (meta-LR over base predictions) ────────────────────
def _build_meta_features(lr_scores, svm_scores, rf_scores):
    """
    Per-option meta-features for the stacking ensemble: 6 features per option,
    capturing each base model's raw vote AND its z-normalised vote (so the
    meta-LR has both absolute confidence and rank-within-question signal).
    Input: each scores array shape (N, 4).  Output: (N, 4, 6).
    """
    return np.stack([
        lr_scores, svm_scores, rf_scores,
        normalize_per_question(lr_scores),
        normalize_per_question(svm_scores),
        normalize_per_question(rf_scores),
    ], axis=2)


def run_stacking(lr_scorer, svm_scorer, rf_model,
                 stack_X, stack_y, val_X, val_y):
    """
    Stacking ensemble (per spec §4.4): train a meta-LR on the *predictions*
    of the base models — LR, SVM, RF — applied to a held-out 10% of train
    that none of them saw during training. The meta-LR learns from data which
    member to trust in which situation, instead of using a fixed weight rule.

    This is the cleanest way to get 'ensemble > best individual': the meta
    classifier can ignore weak votes and amplify strong ones based on
    actual evidence, not just average EM.
    """
    print("\n" + "=" * 70)
    print("A6  Stacking Ensemble  —  Meta-LR over LR + SVM + RF")
    print("=" * 70)

    F = stack_X.shape[2]
    print(f"[Stacking] Collecting base predictions on stack-holdout "
          f"({len(stack_X):,} questions) and on val ({len(val_X):,} questions)...")

    # Base model predictions on the stack-holdout (which they didn't see)
    lr_stack  = lr_scorer.decision_function(stack_X)
    svm_stack = svm_scorer.decision_function(stack_X)
    rf_stack  = rf_model.predict_proba(stack_X.reshape(-1, F))[:, 1].reshape(-1, 4)

    # Base model predictions on val
    lr_val  = lr_scorer.decision_function(val_X)
    svm_val = svm_scorer.decision_function(val_X)
    rf_val  = rf_model.predict_proba(val_X.reshape(-1, F))[:, 1].reshape(-1, 4)

    # Meta-features
    stack_meta = _build_meta_features(lr_stack, svm_stack, rf_stack)
    val_meta   = _build_meta_features(lr_val,   svm_val,   rf_val)

    # Flatten + binary labels for meta training (1 = correct option)
    N_s, _, M = stack_meta.shape
    stack_flat   = stack_meta.reshape(N_s * 4, M)
    stack_y_bin  = np.zeros(N_s * 4, dtype=np.int32)
    stack_y_bin[np.arange(N_s) * 4 + stack_y] = 1

    print(f"[Stacking] Training meta-LR on {N_s * 4:,} option-examples × {M} meta-features")
    meta = LogisticRegression(C=1.0, max_iter=300, n_jobs=-1, random_state=42)
    meta.fit(stack_flat, stack_y_bin)

    # Print learned weights — this directly answers "which member matters most?"
    feat_names = ['LR_raw', 'SVM_raw', 'RF_raw', 'LR_z', 'SVM_z', 'RF_z']
    print("[Stacking] Meta-LR learned weights (signed → which member is trusted):")
    for name, w in zip(feat_names, meta.coef_[0]):
        bar = "█" * int(abs(w) * 10)
        sign = '+' if w >= 0 else '-'
        print(f"  {name:<8} {sign}{abs(w):.4f}   {bar}")

    # Apply to val
    N_v = val_meta.shape[0]
    val_flat  = val_meta.reshape(N_v * 4, M)
    val_probs = meta.predict_proba(val_flat)[:, 1]
    val_scores = val_probs.reshape(N_v, 4)
    val_preds  = val_scores.argmax(axis=1)

    em    = accuracy_score(val_y, val_preds)
    em_f1 = f1_score(val_y, val_preds, average='macro')

    val_y_binary = np.zeros(N_v * 4, dtype=np.int32)
    val_y_binary[np.arange(N_v) * 4 + val_y] = 1
    binary_preds = (val_probs > 0.5).astype(np.int32)
    binary_acc   = accuracy_score(val_y_binary, binary_preds)
    binary_f1    = f1_score(val_y_binary, binary_preds, average='macro')

    print(f"\n[Stacking] Final metrics:")
    print(f"  Per-question (EM, 4-way):     {em * 100:6.2f}%   macro-F1 = {em_f1:.4f}")
    print(f"  Binary  (peer-comparable):    {binary_acc * 100:6.2f}%   macro-F1 = {binary_f1:.4f}")
    print(classification_report(val_y, val_preds,
                                target_names=['A', 'B', 'C', 'D'], digits=4))

    save_model(meta, MODEL_STACKING_PATH)
    # Returning `meta` (the meta-LR) so the test-eval block in main() can
    # apply the same trained meta-classifier to test-set base predictions.
    return em, em_f1, binary_acc, binary_f1, meta


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — TEXT GENERATION (per professor's announcement)
# Spec §4.2.3 — Wh-template question generation evaluated with BLEU/ROUGE/METEOR
# ═════════════════════════════════════════════════════════════════════════════
GEN_NUM_RE  = re.compile(r'\b\d+\b')
GEN_YEAR_RE = re.compile(r'\b\d{4}\b')
GEN_WORD_RE = re.compile(r'\b[a-zA-Z]+\b')

# Heuristic place hints — small allow-list, no NLP tools per spec
PLACE_HINTS = {
    'china', 'japan', 'india', 'usa', 'america', 'europe', 'asia', 'africa',
    'london', 'paris', 'tokyo', 'beijing', 'rome', 'cairo', 'sydney', 'moscow',
    'washington', 'berlin', 'mumbai', 'shanghai', 'school', 'university',
    'hospital', 'kitchen', 'office', 'park', 'garden', 'home', 'beach',
    'mountain', 'river', 'sea', 'ocean', 'forest', 'city', 'town', 'village',
    'country', 'street', 'room', 'library', 'museum', 'restaurant', 'hotel',
}

ARTICLE_WORDS = {'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'His',
                 'Her', 'Its', 'Their', 'Our', 'My', 'Your', 'It', 'He', 'She',
                 'They', 'We', 'I', 'You'}

# Generic Wh-templates used as fallback when literal cloze is impossible.
# These are common RACE-style abstract question templates — applying them
# is still spec-compliant (§4.2.3 says "apply Wh-word templates"; the
# templates needn't come from the sentence). A learned ranker (Step 3,
# deferred) could in principle pick between literal cloze and these.
GENERIC_QG_TEMPLATES = [
    "What is the main idea of the passage?",
    "What is the best title of the passage?",
    "What can we learn from the passage?",
    "What can we infer from the passage?",
    "What does the passage mainly tell us?",
]
MAX_QUESTION_TOKENS = 25  # cap output length — overlong cloze kills BLEU


def _ensure_nltk_data():
    """Download required NLTK data on first run (punkt, wordnet, omw-1.4)."""
    if not _NLTK_AVAILABLE:
        return
    for resource, path in [
        ('punkt',     'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('wordnet',   'corpora/wordnet'),
        ('omw-1.4',   'corpora/omw-1.4'),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK] Downloading {resource}...")
            nltk.download(resource, quiet=True)


def gen_tokens_lower(text):
    """Lowercased word tokens for tokenisation in the generation pipeline."""
    return GEN_WORD_RE.findall(str(text).lower())


def find_answer_sentence(passage, correct_answer_text):
    """
    Spec §4.2.3 Step 1 — pick the passage sentence with maximum keyword
    overlap to the correct answer text. Falls back to the first sentence
    on ties or empty answers.
    """
    sentences = split_sentences(passage)
    if not sentences:
        return str(passage).strip() or ""

    answer_words = set(gen_tokens_lower(correct_answer_text))
    if not answer_words:
        return max(sentences, key=lambda s: len(set(gen_tokens_lower(s))))

    def overlap(sent):
        return len(answer_words & set(gen_tokens_lower(sent)))

    best = max(sentences, key=overlap)
    return best if overlap(best) > 0 else sentences[0]


def pick_wh_word(answer_span):
    """
    Heuristically pick a Wh-word from the answer's surface form.
    'Who' is conservative — only fires on a single capitalised token or
    a person-title prefix (Mr/Mrs/Dr/Miss/Professor/Sir/Lady) — so it
    doesn't over-trigger on multi-word phrases like 'A small Teng from Thailand'.
    """
    answer = str(answer_span).strip()
    if not answer:
        return 'What'
    if GEN_YEAR_RE.fullmatch(answer):
        return 'When'
    if GEN_NUM_RE.fullmatch(answer.replace(',', '')):
        return 'How many'
    if any(tok in PLACE_HINTS for tok in gen_tokens_lower(answer)):
        return 'Where'

    tokens = answer.split()
    if not tokens:
        return 'What'
    first = tokens[0]
    if first.lower().rstrip('.') in {'mr', 'mrs', 'dr', 'miss', 'professor', 'sir', 'lady'}:
        return 'Who'
    if len(tokens) == 1 and first[0].isupper() and first not in ARTICLE_WORDS:
        return 'Who'
    return 'What'


def _generic_template_for(passage):
    """Deterministic per-passage rotation across RACE-style templates."""
    return GENERIC_QG_TEMPLATES[abs(hash(passage)) % len(GENERIC_QG_TEMPLATES)]


def cloze_question(sentence, answer_span, passage_for_fallback=""):
    """
    Spec §4.2.3 Step 2 — substitute the answer span in the sentence with
    a Wh-word, end with '?'. Improvements over the v1 version:

      1) Try literal substring match first (case-insensitive).
      2) If miss, retry with progressively shorter trailing slices of the
         answer (last 3, then last 2, then last 1 token) — RACE answer
         strings are often paraphrases, but the last few content words
         often appear verbatim in the passage.
      3) If still miss, emit a generic Wh-template (rotated per passage).
      4) Cap output at MAX_QUESTION_TOKENS (≈25). Anything longer is the
         "What is described by: [whole sentence]" failure mode that
         destroys BLEU on length mismatch — replace with a generic template.
    """
    sentence = str(sentence).strip()
    answer = str(answer_span).strip()
    if not sentence or not answer:
        return _generic_template_for(passage_for_fallback or sentence)

    wh = pick_wh_word(answer)

    def _try_substitute(span):
        pat = re.compile(re.escape(span), re.IGNORECASE)
        if pat.search(sentence):
            return pat.sub(wh, sentence, count=1)
        return None

    question = _try_substitute(answer)

    # Progressive shortening: last 3 → last 2 → last 1 token of the answer
    if question is None:
        ans_tokens = answer.split()
        for k in (3, 2, 1):
            if len(ans_tokens) >= k:
                cand_span = ' '.join(ans_tokens[-k:])
                question = _try_substitute(cand_span)
                if question is not None:
                    break

    # Final fallback: generic template — not the awful "What is described by:" dump
    if question is None:
        return _generic_template_for(passage_for_fallback or sentence)

    # Tidy: trim, end with single '?', capitalise first letter
    question = question.rstrip('.!,;: \t').rstrip()
    if not question.endswith('?'):
        question += '?'
    if question and question[0].islower():
        question = question[0].upper() + question[1:]

    # Length cap — overlong clozes torpedo BLEU vs short gold questions
    if len(question.split()) > MAX_QUESTION_TOKENS:
        return _generic_template_for(passage_for_fallback or sentence)

    return question


def generate_question(passage, correct_answer_text):
    """End-to-end: passage + correct answer → generated question."""
    sentence = find_answer_sentence(passage, correct_answer_text)
    return cloze_question(sentence, correct_answer_text, passage_for_fallback=passage)


class GenerationMetrics:
    """
    Bundles per-row BLEU/ROUGE/METEOR (sentence-level) AND collects token
    streams for the standard Papineni-2002 corpus-BLEU computation at the
    end of a split.

    Two BLEU numbers are reported:
      • bleu_sent   — sentence-level BLEU per row, then averaged.
      • bleu_corpus — corpus BLEU across the whole split. This is the
                      "standard" BLEU and is typically 1.5–2× the averaged
                      sentence-BLEU; peers reporting the higher number are
                      almost certainly using corpus BLEU.
    """
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.smoothie = SmoothingFunction().method1
        # Buffers for corpus-BLEU at the end of a split
        self._corpus_refs = []  # list of [ref_tokens]
        self._corpus_gens = []  # list of gen_tokens

    def score(self, generated, reference):
        gen_tokens = gen_tokens_lower(generated)
        ref_tokens = gen_tokens_lower(reference)
        if not gen_tokens or not ref_tokens:
            return {'bleu_1': 0.0, 'bleu': 0.0,
                    'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
                    'meteor': 0.0}
        # Buffer for later corpus-BLEU computations
        self._corpus_refs.append([ref_tokens])
        self._corpus_gens.append(gen_tokens)
        # BLEU-1 (unigram precision only) — more meaningful for short
        # outputs (~10 tokens) where 4-gram match rates collapse to zero.
        bleu_1 = sentence_bleu([ref_tokens], gen_tokens,
                               weights=(1.0,),
                               smoothing_function=self.smoothie)
        # BLEU-4 (NLTK default — geometric mean of 1..4 gram precisions).
        bleu_4 = sentence_bleu([ref_tokens], gen_tokens,
                               smoothing_function=self.smoothie)
        return {
            'bleu_1': bleu_1,
            'bleu':   bleu_4,
            'rouge1': self.rouge.score(reference, generated)['rouge1'].fmeasure,
            'rouge2': self.rouge.score(reference, generated)['rouge2'].fmeasure,
            'rougeL': self.rouge.score(reference, generated)['rougeL'].fmeasure,
            'meteor': meteor_score([ref_tokens], gen_tokens),
        }

    def corpus_bleu_at_n(self, n=4):
        """Corpus BLEU with weights (1/n,…,1/n). n=1 → BLEU-1, n=4 → BLEU-4."""
        if not self._corpus_gens:
            return 0.0
        weights = tuple([1.0 / n] * n)
        return corpus_bleu(self._corpus_refs, self._corpus_gens,
                           weights=weights,
                           smoothing_function=self.smoothie)

    def corpus_bleu(self):
        """Backward-compat shorthand for corpus BLEU-4."""
        return self.corpus_bleu_at_n(4)


def evaluate_generation_split(df, label, sample_examples=5):
    """
    Generate a question for each row of df and average BLEU/ROUGE/METEOR
    against the gold question. Returns (avg_metrics, examples, n_scored).
    """
    metrics = GenerationMetrics()
    keys = ['bleu_1', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']
    sums = {k: 0.0 for k in keys}
    n = 0
    examples = []

    print(f"\n[GEN-EVAL] Generating + scoring on {label} ({len(df):,} questions)...")
    for i, row in df.iterrows():
        passage     = str(row.get('article', ''))
        gold_q      = str(row.get('question', '')).strip()
        gold_letter = row.get('answer', None)
        if pd.isna(gold_letter) or gold_letter not in ('A', 'B', 'C', 'D'):
            continue
        gold_a = str(row.get(gold_letter, '')).strip()
        if not gold_q or not gold_a or not passage:
            continue

        gen_q = generate_question(passage, gold_a)
        s = metrics.score(gen_q, gold_q)
        for k in keys:
            sums[k] += s[k]
        n += 1

        if len(examples) < sample_examples:
            examples.append({
                'passage_excerpt': passage[:140] + ('…' if len(passage) > 140 else ''),
                'gold_q': gold_q,
                'gen_q':  gen_q,
                'gold_a': gold_a,
                'scores': s,
            })

        if (i + 1) % 1000 == 0:
            print(f"  scored {i + 1:,}/{len(df):,}")

    avg = {k: (sums[k] / n if n else 0.0) for k in keys}
    avg['bleu_corpus']   = metrics.corpus_bleu_at_n(4)  # corpus BLEU-4
    avg['bleu_1_corpus'] = metrics.corpus_bleu_at_n(1)  # corpus BLEU-1
    return avg, examples, n


def sentence_ranker_features(sentence, passage, answer, idf_lookup, position):
    """
    8 features for ranking whether a candidate sentence is the answer-bearing
    one. All non-negative so any classifier (incl. MultinomialNB) accepts them.
      - overlap_count   : raw token overlap with answer
      - overlap_idf     : IDF-weighted overlap (rare-word matches count more)
      - n_tokens        : sentence length
      - position        : sentence index in passage, normalised to [0, 1]
      - psg_overlap     : fraction of sentence tokens that recur in passage
      - has_capitalised : flag — sentence contains a non-initial capitalised
                          word (proxy for named-entity content)
      - has_digit       : flag — sentence contains digits (numbers / years)
      - avg_token_len   : mean token length (proxy for content density)
    """
    s_tokens = gen_tokens_lower(sentence)
    a_tokens = gen_tokens_lower(answer)
    p_tokens = gen_tokens_lower(passage)
    if not s_tokens:
        return np.zeros(8, dtype=np.float32)

    s_set, a_set, p_set = set(s_tokens), set(a_tokens), set(p_tokens)

    overlap_count = float(len(s_set & a_set))
    overlap_idf   = float(sum(idf_lookup.get(w, 1.0) for w in (s_set & a_set)))
    n_tokens      = float(len(s_tokens))
    psg_overlap   = float(len(s_set & p_set)) / max(1.0, len(s_set))

    raw_tokens = sentence.split()
    has_cap = float(any(t[0].isupper() for t in raw_tokens[1:] if t and t[0].isalpha()))
    has_dig = float(bool(re.search(r'\d', sentence)))
    avg_len = float(np.mean([len(t) for t in s_tokens]))

    return np.array([
        overlap_count, overlap_idf, n_tokens, position,
        psg_overlap, has_cap, has_dig, avg_len,
    ], dtype=np.float32)


def build_ranker_train_data(df, idf_lookup, max_rows=20000, neg_per_pos=3,
                            label_by_bleu=True):
    """
    Build (X, y) for training sentence-ranker classifiers.

    Two label strategies:
      label_by_bleu = False
          Positive = highest-overlap-with-gold-answer sentence.
          Trains the ranker to *imitate* the overlap heuristic — which means
          it can match the baseline at best, never beat it.

      label_by_bleu = True   (default)
          Positive = the sentence whose Wh-cloze produces the highest BLEU
          against the gold question. Now the ranker learns *what produces
          good QUESTIONS*, not what looks answer-bearing — so it can
          legitimately beat the overlap baseline at generation time.

    Each row contributes 1 positive + up to neg_per_pos sampled negatives.
    """
    X, y = [], []
    rng = np.random.default_rng(42)

    # Per-row sentence_bleu scorer (reuses the same smoothing as evaluation)
    smoothie = SmoothingFunction().method1 if label_by_bleu else None

    for _, row in df.head(max_rows).iterrows():
        passage     = str(row.get('article', ''))
        gold_letter = row.get('answer', None)
        if pd.isna(gold_letter) or gold_letter not in ('A', 'B', 'C', 'D'):
            continue
        gold_a = str(row.get(gold_letter, '')).strip()
        gold_q = str(row.get('question', '')).strip()
        if not gold_a or not passage:
            continue
        if label_by_bleu and not gold_q:
            continue

        sentences = split_sentences(passage)
        if len(sentences) < 2:
            continue

        if label_by_bleu:
            # Score each sentence by the BLEU of its cloze vs gold question
            ref_tokens = gen_tokens_lower(gold_q)
            scored = []
            for idx, s in enumerate(sentences):
                cloze_q   = cloze_question(s, gold_a, passage_for_fallback=passage)
                gen_tokens = gen_tokens_lower(cloze_q)
                if not gen_tokens or not ref_tokens:
                    bleu = 0.0
                else:
                    bleu = sentence_bleu([ref_tokens], gen_tokens,
                                         smoothing_function=smoothie)
                scored.append((idx, bleu))
        else:
            # Original: overlap-based
            gold_words = set(gen_tokens_lower(gold_a))
            scored = [(idx, len(gold_words & set(gen_tokens_lower(s))))
                      for idx, s in enumerate(sentences)]

        scored.sort(key=lambda t: t[1], reverse=True)
        pos_idx = scored[0][0]
        neg_pool = [idx for idx, _ in scored[1:]]
        # Randomly sample up to neg_per_pos negatives
        if len(neg_pool) > neg_per_pos:
            neg_pool = list(rng.choice(neg_pool, size=neg_per_pos, replace=False))

        n_sents = len(sentences)
        for idx, label in [(pos_idx, 1)] + [(j, 0) for j in neg_pool]:
            pos_norm = idx / max(1, n_sents - 1)
            X.append(sentence_ranker_features(
                sentences[idx], passage, gold_a, idf_lookup, pos_norm))
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_sentence_rankers(train_df, idf_lookup):
    """
    Train three sentence rankers — LR, NB, SVM — on the same features.
    Returns dict {'LR': ..., 'NB': ..., 'SVM': ...}.
    """
    print("\n[GEN-RANKERS] Building BLEU-labelled training data for sentence rankers...")
    print("  (positive = sentence whose cloze BLEUs highest vs gold question;")
    print("   trains the ranker to pick *generation-friendly* sentences, not")
    print("   just answer-bearing ones — the latter would just imitate the baseline)")
    X, y = build_ranker_train_data(train_df, idf_lookup, label_by_bleu=True)
    print(f"  {len(X):,} training examples × {X.shape[1]} features  "
          f"(positives: {int(y.sum()):,})")

    rankers = {}

    print("[GEN-RANKERS] Training LR ranker...")
    rankers['LR'] = LogisticRegression(
        C=1.0, max_iter=300, n_jobs=-1, random_state=42
    ).fit(X, y)

    print("[GEN-RANKERS] Training NB ranker...")
    rankers['NB'] = MultinomialNB(alpha=1.0).fit(X, y)

    print("[GEN-RANKERS] Training SVM ranker...")
    rankers['SVM'] = SGDClassifier(
        loss='hinge', alpha=1e-4, max_iter=50, random_state=42, n_jobs=-1,
    ).fit(X, y)

    return rankers


def _ranker_score(ranker, feats):
    """Get probability/decision score for a ranker that may or may not have predict_proba."""
    if hasattr(ranker, 'predict_proba'):
        return ranker.predict_proba(feats)[:, 1]
    return ranker.decision_function(feats)


def generate_question_with_ranker(passage, answer_text, ranker, idf_lookup):
    """
    Spec §4.2.3 Step 3 — pick the cloze source sentence using a trained
    ranker instead of the overlap heuristic, then apply the same Wh-template
    cloze (Step 2). Returns the generated question string.
    """
    sentences = split_sentences(passage)
    if not sentences:
        return _generic_template_for(passage)

    n_sents = len(sentences)
    feats = np.stack([
        sentence_ranker_features(s, passage, answer_text, idf_lookup,
                                 idx / max(1, n_sents - 1))
        for idx, s in enumerate(sentences)
    ])
    scores = _ranker_score(ranker, feats)
    best_sent = sentences[int(scores.argmax())]
    return cloze_question(best_sent, answer_text, passage_for_fallback=passage)


def evaluate_generation_with_ranker(df, ranker, idf_lookup, label):
    """Evaluate a single ranker variant — returns (avg_metrics, n_scored)."""
    metrics = GenerationMetrics()
    keys = ['bleu_1', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']
    sums = {k: 0.0 for k in keys}
    n = 0
    print(f"  [{label}] Generating + scoring ({len(df):,} questions)...")
    for i, row in df.iterrows():
        passage     = str(row.get('article', ''))
        gold_q      = str(row.get('question', '')).strip()
        gold_letter = row.get('answer', None)
        if pd.isna(gold_letter) or gold_letter not in ('A', 'B', 'C', 'D'):
            continue
        gold_a = str(row.get(gold_letter, '')).strip()
        if not gold_q or not gold_a or not passage:
            continue
        gen_q = generate_question_with_ranker(passage, gold_a, ranker, idf_lookup)
        s = metrics.score(gen_q, gold_q)
        for k in keys:
            sums[k] += s[k]
        n += 1
        if (i + 1) % 2000 == 0:
            print(f"    scored {i + 1:,}/{len(df):,}")
    avg = {k: sums[k] / n if n else 0.0 for k in keys}
    avg['bleu_corpus']   = metrics.corpus_bleu_at_n(4)
    avg['bleu_1_corpus'] = metrics.corpus_bleu_at_n(1)
    return avg, n


def run_generation_pipeline(train_df, val_df, test_df, idf_lookup):
    """
    Full spec §4.2.3 question-generation pipeline:
      Step 1 — extract candidate sentences (overlap heuristic OR ranker)
      Step 2 — Wh-template cloze
      Step 3 — Ranker over candidate sentences (LR / NB / SVM)

    Trains all three rankers on `train_df`, then evaluates each variant
    plus a Baseline (overlap heuristic) on val and test. Returns a dict
    keyed by variant name, suitable for print_summary's `generation` arg.

    Each variant entry has:
      {'val': {'metrics': {...}, 'n': int},
       'test': {'metrics': {...}, 'n': int}}
    """
    print("\n" + "=" * 70)
    print("A7  Question Generation  —  Wh-template cloze + sentence ranker")
    print("=" * 70)

    if not _NLTK_AVAILABLE:
        print("[WARN] nltk / rouge-score not installed — skipping generation.")
        print("       Install with: pip install nltk rouge-score")
        return None

    _ensure_nltk_data()

    # ─── Variant 1: Baseline (overlap heuristic, current logic) ──────────
    print("\n[GEN] Variant: Baseline (overlap heuristic)")
    bv_m, bv_ex, bv_n = evaluate_generation_split(val_df,  "VAL")
    bt_m, _,     bt_n = evaluate_generation_split(test_df, "TEST")

    # 5 baseline examples for eyeballing quality
    print("\n[Examples] Baseline cloze vs gold (val):")
    for i, ex in enumerate(bv_ex[:5], start=1):
        print(f"\n  ── Example {i} ─────────────────────────────────────")
        print(f"  Passage : {ex['passage_excerpt']}")
        print(f"  Gold A  : {ex['gold_a']}")
        print(f"  Gold Q  : {ex['gold_q']}")
        print(f"  Gen  Q  : {ex['gen_q']}")
        s = ex['scores']
        print(f"  Scores  : BLEU={s['bleu']:.3f}  R1={s['rouge1']:.3f}  "
              f"R2={s['rouge2']:.3f}  RL={s['rougeL']:.3f}  MET={s['meteor']:.3f}")

    results = {
        'Baseline (overlap)': {
            'val':  {'metrics': bv_m, 'n': bv_n},
            'test': {'metrics': bt_m, 'n': bt_n},
        }
    }

    # ─── Variants 2-4: LR / NB / SVM sentence rankers (Step 3) ───────────
    rankers = train_sentence_rankers(train_df, idf_lookup)
    for name, ranker in rankers.items():
        print(f"\n[GEN] Variant: {name} sentence ranker")
        v_m, v_n = evaluate_generation_with_ranker(val_df,  ranker, idf_lookup, "VAL")
        t_m, t_n = evaluate_generation_with_ranker(test_df, ranker, idf_lookup, "TEST")
        results[f'Ranker — {name}'] = {
            'val':  {'metrics': v_m, 'n': v_n},
            'test': {'metrics': t_m, 'n': t_n},
        }

    return results


# ─── Final summary ───────────────────────────────────────────────────────────
def print_summary(results, generation=None):
    """
    Final comparison table for Model A.
      results    : verification results dict (LR, SVM, NB, K-Means, ensembles)
      generation : optional dict from run_generation_pipeline with 'VAL'/'TEST'
                   keys, each mapping to {'metrics': {...}, 'n': int}.
    """
    # ─── Part 1: Verification (classification) ───
    print("\n" + "=" * 110)
    print("FINAL COMPARISON  —  Model A  (Part 1: Verification, classical ML)")
    print("=" * 110)
    print(f"{'Approach':<26} {'Task':<16} "
          f"{'Val-EM':>8} {'Val-F1':>8}   {'Test-EM':>8} {'Test-F1':>8}   "
          f"{'BinAcc':>8} {'BinF1':>8}")
    print("-" * 110)
    for name, r in results.items():
        if 'em' in r:
            em_str = f"{r['em'] * 100:>7.2f}%"
            f1_str = f"{r['em_f1']:>8.4f}"
            t_em   = f"{r['test_em'] * 100:>7.2f}%" if 'test_em' in r else f"{'—':>8}"
            t_f1   = f"{r['test_em_f1']:>8.4f}" if 'test_em_f1' in r else f"{'—':>8}"
            if 'binary_acc' in r:
                bacc = f"{r['binary_acc'] * 100:>7.2f}%"
                bf1  = f"{r['binary_f1']:>8.4f}"
            else:
                bacc = f"{'—':>8}"
                bf1  = f"{'—':>8}"
            print(f"{name:<26} {r['task']:<16} {em_str:>8} {f1_str}   "
                  f"{t_em:>8} {t_f1}   {bacc} {bf1}")
        elif 'purity' in r:
            pur_str = f"{r['purity'] * 100:>7.2f}%"
            print(f"{name:<26} {r['task']:<16} {pur_str:>8} {'—':>8}   "
                  f"{'—':>8} {'—':>8}   {'—':>8} {'—':>8}")
    print("=" * 110)

    # ─── Part 2: Generation (text-gen, BLEU/ROUGE/METEOR) ───
    if generation:
        print("\n" + "=" * 92)
        print("FINAL COMPARISON  —  Model A  (Part 2: Generation, BLEU/ROUGE/METEOR)")
        print("=" * 132)
        # BLEU columns:
        #   BLEU-1   = unigram precision (more meaningful for short outputs;
        #              what most short-text generation papers report)
        #   BLEU-4   = NLTK default 4-gram (geometric mean of 1..4-gram precision)
        #   *-c suffix = corpus BLEU (Papineni 2002), pooled across the split
        print(f"{'Variant':<22} {'Split':<6} {'N':>7}   "
              f"{'BLEU-1':>7} {'BLEU-1-c':>9} {'BLEU-4':>7} {'BLEU-4-c':>9} "
              f"{'ROUGE-1':>8} {'ROUGE-L':>8} {'METEOR':>7}")
        print("-" * 132)
        for variant_name, splits in generation.items():
            for split_name in ('val', 'test'):
                if split_name not in splits:
                    continue
                m = splits[split_name]['metrics']
                n = splits[split_name]['n']
                b1   = m.get('bleu_1', 0.0)
                b1_c = m.get('bleu_1_corpus', 0.0)
                b4   = m.get('bleu', 0.0)
                b4_c = m.get('bleu_corpus', 0.0)
                print(f"{variant_name:<22} {split_name.upper():<6} {n:>7,}   "
                      f"{b1:>7.4f} {b1_c:>9.4f} {b4:>7.4f} {b4_c:>9.4f} "
                      f"{m['rouge1']:>8.4f} {m['rougeL']:>8.4f} {m['meteor']:>7.4f}")
            print("-" * 132)
        print("=" * 132)
    print()


# ─── Main orchestrator ────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("# QuizForge — Model A")
    print("#" * 70)

    # ─── 1. Load data once ────────────────────────────────────────────────
    train_df, val_df, train_y, val_y = load_data()

    # Test split is loaded only for Part 2 (generation eval). Part 1 keeps
    # using val for model selection — unchanged from before.
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"  Test:  {len(test_df):,} questions  (used for generation eval)")

    # ─── 2. Compute shared resources (IDF, TF-IDF vectorizer) ─────────────
    print("\n[SHARED] Computing IDF over training corpus...")
    idf = compute_idf(TRAIN_FEATURES_PATH)
    print(f"  IDF range [{idf.min():.3f}, {idf.max():.3f}], median {np.median(idf):.3f}")

    vec = fit_tfidf_vectorizer(TRAIN_CSV_PATH)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vec, TFIDF_VEC_PATH)
    print(f"  vectorizer saved → {TFIDF_VEC_PATH}")

    # GloVe is optional — auto-detect, skip gracefully if unavailable
    glove = load_glove()

    # IDF lookup (word → IDF score) — used by GloVe averaging *and* by the
    # generation ranker features. Always build it; it's cheap.
    idf_lookup = build_idf_lookup(vec)
    print(f"  IDF lookup ready ({len(idf_lookup):,} unigrams from TF-IDF vocab)")

    # ─── 3. Build per-option features once (used by LR, SVM, Ensemble) ────
    train_X = build_features(
        TRAIN_FEATURES_PATH, train_df, idf, vec,
        glove=glove, idf_lookup=idf_lookup, label="train",
    )
    val_X = build_features(
        VAL_FEATURES_PATH, val_df, idf, vec,
        glove=glove, idf_lookup=idf_lookup, label="val",
    )

    # ─── 3b. Hold out 10% of train for the stacking meta-classifier ───────
    # Stacking needs base-model predictions on data the bases didn't see
    # during training. We train base models on the 90% main split and
    # evaluate them on the 10% holdout to build clean meta-features.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(train_X))
    n_stack = len(train_X) // 10
    stack_idx = perm[:n_stack]
    main_idx  = perm[n_stack:]
    train_X_main = train_X[main_idx]
    train_y_main = train_y[main_idx]
    stack_X      = train_X[stack_idx]
    stack_y      = train_y[stack_idx]
    print(f"\n[STACKING] Holdout: {len(stack_X):,} questions   "
          f"Base train: {len(train_X_main):,} questions")

    results = {}

    # ─── 4. Run each approach ─────────────────────────────────────────────
    # Base models are trained on the 90% main split (so we can collect clean
    # OOF-style predictions on the 10% holdout for stacking). This costs
    # individual EM ~0.3–0.5pp but is required for honest stacking.
    lr_scorer, lr_em, lr_em_f1, lr_scores, lr_bin_acc, lr_bin_f1 = run_logistic_regression(
        train_X_main, train_y_main, val_X, val_y, idf, vec
    )
    results['Logistic Regression'] = {
        'task': 'Answer Verif', 'em': lr_em, 'em_f1': lr_em_f1,
        'binary_acc': lr_bin_acc, 'binary_f1': lr_bin_f1,
    }

    svm_scorer, svm_em, svm_em_f1, svm_scores, svm_bin_acc, svm_bin_f1 = run_linear_svm(
        train_X_main, train_y_main, val_X, val_y, idf, vec
    )
    results['Linear SVM'] = {
        'task': 'Answer Verif', 'em': svm_em, 'em_f1': svm_em_f1,
        'binary_acc': svm_bin_acc, 'binary_f1': svm_bin_f1,
    }

    # Random Forest is trained as a third *ensemble base member* per spec §4.4
    # (NOT listed as a primary deliverable — the spec puts RF under "Difficulty
    # Estimation", not Answer Verification). Its standalone EM is shown inside
    # the A2.5 section above; we don't add it to the final summary table to
    # keep the canonical 5 (LR, SVM, NB, K-Means, Ensemble) clean.
    # We need rf_em (for ensemble weighting), rf_scores (soft-vote input),
    # and rf_model (stacking input). The other return values are unused
    # since RF doesn't appear in the final summary table.
    rf_em, _, rf_scores, _, _, rf_model = run_random_forest(
        train_X_main, train_y_main, val_X, val_y
    )

    nb_acc, nb_f1 = run_naive_bayes(train_df, val_df)
    results['Naive Bayes'] = {
        'task': 'Question Type', 'em': nb_acc, 'em_f1': nb_f1,
    }

    kmeans_purity = run_kmeans(TRAIN_FEATURES_PATH, VAL_FEATURES_PATH, val_y)
    results['K-Means Clustering'] = {'task': 'Clustering', 'purity': kmeans_purity}

    # A5: weighted soft voting (heuristic weights from individual EM)
    ens_em, ens_em_f1, ens_bin_acc, ens_bin_f1 = run_ensemble(
        members=[
            {'name': 'LR',  'scores': lr_scores,  'em': lr_em},
            {'name': 'SVM', 'scores': svm_scores, 'em': svm_em},
            {'name': 'RF',  'scores': rf_scores,  'em': rf_em},
        ],
        val_y=val_y,
    )
    results['Ensemble — Soft Voting'] = {
        'task': 'Answer Verif', 'em': ens_em, 'em_f1': ens_em_f1,
        'binary_acc': ens_bin_acc, 'binary_f1': ens_bin_f1,
    }

    # A6: stacking ensemble (meta-LR with weights learned from data)
    stack_em, stack_em_f1, stack_bin_acc, stack_bin_f1, meta_lr = run_stacking(
        lr_scorer, svm_scorer, rf_model,
        stack_X, stack_y, val_X, val_y,
    )
    results['Ensemble — Stacking'] = {
        'task': 'Answer Verif', 'em': stack_em, 'em_f1': stack_em_f1,
        'binary_acc': stack_bin_acc, 'binary_f1': stack_bin_f1,
    }

    # ─── 4b. TEST-set evaluation for verification (held-out generalisation) ─
    # Val is what we tuned on (best-epoch selection, ensemble weights).
    # Test is held-out — if test ≈ val, no overfitting; if test << val, overfit.
    print("\n" + "=" * 70)
    print("Test-set evaluation  —  generalisation check on held-out data")
    print("=" * 70)
    test_y = test_df['answer'].map(ANSWER_MAP).values.astype(np.int32)
    test_X = build_features(
        TEST_FEATURES_PATH, test_df, idf, vec,
        glove=glove, idf_lookup=idf_lookup, label="test",
    )
    F = test_X.shape[2]

    # Base-model test predictions
    lr_test_scores  = lr_scorer.decision_function(test_X)
    svm_test_scores = svm_scorer.decision_function(test_X)
    rf_test_scores  = rf_model.predict_proba(test_X.reshape(-1, F))[:, 1].reshape(-1, 4)

    for nm, scores in [
        ('Logistic Regression', lr_test_scores),
        ('Linear SVM',          svm_test_scores),
    ]:
        em, em_f1, ba, bf = eval_per_option_metrics(scores, test_y)
        results[nm].update({
            'test_em': em, 'test_em_f1': em_f1,
            'test_binary_acc': ba, 'test_binary_f1': bf,
        })
        print(f"[TEST] {nm:<22} EM={em*100:6.2f}%  F1={em_f1:.4f}")

    # Soft-Voting Ensemble on test (re-use the same skill-above-random weights
    # that run_ensemble computed from val EMs).
    val_ems_for_weights = np.array(
        [results['Logistic Regression']['em'],
         results['Linear SVM']['em'],
         rf_em],
        dtype=np.float64,
    )
    raw_w = np.maximum(0.0, val_ems_for_weights - 0.25)
    weights = raw_w / raw_w.sum() if raw_w.sum() > 1e-9 else np.ones(3) / 3
    ens_test_scores = (
        weights[0] * normalize_per_question(lr_test_scores)
        + weights[1] * normalize_per_question(svm_test_scores)
        + weights[2] * normalize_per_question(rf_test_scores)
    ).astype(np.float32)
    em, em_f1, ba, bf = eval_per_option_metrics(ens_test_scores, test_y)
    results['Ensemble — Soft Voting'].update({
        'test_em': em, 'test_em_f1': em_f1,
        'test_binary_acc': ba, 'test_binary_f1': bf,
    })
    print(f"[TEST] {'Ensemble — Soft Voting':<22} EM={em*100:6.2f}%  F1={em_f1:.4f}")

    # Stacking Ensemble on test — re-apply the same trained meta-LR
    test_meta = _build_meta_features(lr_test_scores, svm_test_scores, rf_test_scores)
    N_t = test_meta.shape[0]
    test_meta_flat = test_meta.reshape(N_t * 4, -1)
    test_probs = meta_lr.predict_proba(test_meta_flat)[:, 1]
    stack_test_scores = test_probs.reshape(N_t, 4)
    em, em_f1, ba, bf = eval_per_option_metrics(stack_test_scores, test_y)
    results['Ensemble — Stacking'].update({
        'test_em': em, 'test_em_f1': em_f1,
        'test_binary_acc': ba, 'test_binary_f1': bf,
    })
    print(f"[TEST] {'Ensemble — Stacking':<22} EM={em*100:6.2f}%  F1={em_f1:.4f}")

    # ─── 5. Part 2: Question generation pipeline (BLEU/ROUGE/METEOR) ──────
    generation = run_generation_pipeline(train_df, val_df, test_df, idf_lookup)

    # ─── 6. Comparison table ──────────────────────────────────────────────
    print_summary(results, generation=generation)


if __name__ == "__main__":
    main()
