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
        print("  (first run downloads ~70 MB to ~/gensim-data/, then cached)")
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


# ─── Final summary ───────────────────────────────────────────────────────────
def print_summary(results, generation=None):
    print("\n" + "=" * 92)
    print("FINAL COMPARISON  —  Model A  (Part 1: Verification)")
    print("=" * 92)
    print(f"{'Approach':<30} {'Task':<16} "
          f"{'EM':>8} {'EM-F1':>8}   {'BinAcc':>8} {'BinF1':>8}")
    print("-" * 92)
    for name, r in results.items():
        if 'em' in r:
            em_str = f"{r['em'] * 100:>7.2f}%"
            f1_str = f"{r['em_f1']:>8.4f}"
            if 'binary_acc' in r:
                bacc = f"{r['binary_acc'] * 100:>7.2f}%"
                bf1  = f"{r['binary_f1']:>8.4f}"
            else:
                bacc = f"{'—':>8}"
                bf1  = f"{'—':>8}"
            print(f"{name:<30} {r['task']:<16} {em_str:>8} {f1_str}   {bacc} {bf1}")
    print("=" * 92 + "\n")


# ─── Main orchestrator ────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("# QuizForge — Model A")
    print("#" * 70)

    train_df, val_df, train_y, val_y = load_data()
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"  Test:  {len(test_df):,} questions")

    print("\n[SHARED] Computing IDF over training corpus...")
    idf = compute_idf(TRAIN_FEATURES_PATH)
    print(f"  IDF range [{idf.min():.3f}, {idf.max():.3f}], median {np.median(idf):.3f}")

    vec = fit_tfidf_vectorizer(TRAIN_CSV_PATH)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vec, TFIDF_VEC_PATH)
    print(f"  vectorizer saved → {TFIDF_VEC_PATH}")

    glove = load_glove()
    idf_lookup = build_idf_lookup(vec)
    print(f"  IDF lookup ready ({len(idf_lookup):,} unigrams from TF-IDF vocab)")

    train_X = build_features(TRAIN_FEATURES_PATH, train_df, idf, vec,
                             glove=glove, idf_lookup=idf_lookup, label="train")
    val_X = build_features(VAL_FEATURES_PATH, val_df, idf, vec,
                           glove=glove, idf_lookup=idf_lookup, label="val")

    results = {}

    lr_scorer, lr_em, lr_em_f1, lr_scores, lr_bin_acc, lr_bin_f1 = run_logistic_regression(
        train_X, train_y, val_X, val_y, idf, vec)
    results['Logistic Regression'] = {
        'task': 'Answer Verif', 'em': lr_em, 'em_f1': lr_em_f1,
        'binary_acc': lr_bin_acc, 'binary_f1': lr_bin_f1,
    }

    svm_scorer, svm_em, svm_em_f1, svm_scores, svm_bin_acc, svm_bin_f1 = run_linear_svm(
        train_X, train_y, val_X, val_y, idf, vec)
    results['Linear SVM'] = {
        'task': 'Answer Verif', 'em': svm_em, 'em_f1': svm_em_f1,
        'binary_acc': svm_bin_acc, 'binary_f1': svm_bin_f1,
    }

    rf_em, rf_em_f1, rf_scores, rf_bin_acc, rf_bin_f1, _rf_model = run_random_forest(
        train_X, train_y, val_X, val_y)
    results['Random Forest'] = {
        'task': 'Answer Verif', 'em': rf_em, 'em_f1': rf_em_f1,
        'binary_acc': rf_bin_acc, 'binary_f1': rf_bin_f1,
    }

    print_summary(results)


if __name__ == "__main__":
    main()
