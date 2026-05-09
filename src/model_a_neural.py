"""
Model A — Neural Approach (sentence-transformers based)
========================================================

Spec mapping
------------
§4.2.2 lists "fine-tune bert-base-uncased on RACE train split for 3–5 epochs"
as one of the unsupervised/semi-supervised options. Fine-tuning bert-base on
CPU is impractical (~hours per epoch), so we use sentence-transformers
`all-MiniLM-L6-v2` instead — a distilled, already-fine-tuned BERT variant
that is:
  - 80 MB on disk (vs 440 MB for bert-base-uncased)
  - 384-dim sentence embeddings (vs 768)
  - CPU-feasible (~1000 sentences/sec on a modern laptop)

`sentence-transformers` is explicitly listed in spec §7.1 ("Embeddings —
sentence-transformers — Semantic similarity for hint ranking") as part of
the recommended stack. We use it for two Model-A neural sub-tasks AND
expose `rank_sentences_by_relevance` so Model B can later reuse it for
hint extraction.

Why this file is separate from `model_a.py`
-------------------------------------------
The user's directive: "keep our implementations separate from the neural
ones for comparison, don't wanna end up mixing them." So all classical
work lives in `model_a.py`; this file is the standalone neural module.
We import only utility functions (sentence splitter, cloze logic, metrics)
from `model_a.py` — no neural code leaks back into the classical pipeline.

What this module produces
-------------------------
1. Neural Verification: per-option semantic features (passage↔option,
   question↔option, (passage+question)↔option cosines + comparatives) →
   small Logistic Regression head. Compared against the classical
   Stacking ensemble.

2. Neural Generation re-ranker: sentence-transformer cosine to pick the
   answer-bearing sentence (alternative to overlap heuristic), with the
   same Wh-cloze + length-cap logic from the classical pipeline.

3. Reusable `rank_sentences_by_relevance()` for Model B's hint extractor.

Embeddings are cached to `data/embeddings/*.npy` so subsequent runs
skip the encoding step entirely.

Usage: python src/model_a_neural.py
       (requires `pip install sentence-transformers`)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Sentence-transformers is the only neural dependency; gracefully fall back
# if it's not installed (the rest of the project keeps working).
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TEST_CSV_PATH,
    ANSWER_MAP,
    MODELS_DIR,
    DATA_DIR,
)

# Reuse classical utilities — these contain no neural code.
from model_a import (
    split_sentences,
    cloze_question,
    eval_per_option_metrics,
    vs_others,
    GenerationMetrics,
)


# ─── Constants ───────────────────────────────────────────────────────────────
EMBED_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE   = 64
EMBEDDINGS_DIR     = os.path.join(DATA_DIR, "embeddings")
MODEL_NEURAL_PATH  = os.path.join(MODELS_DIR, "model_a_neural_lr.pkl")
EPS                = 1e-9
TRAIN_SUBSET       = 30_000   # subsample train to keep CPU runtime tractable


# ─── Model loader + caching helpers ───────────────────────────────────────────
def load_neural_model():
    """Load sentence-transformers model on CPU."""
    print(f"\n[NEURAL] Loading {EMBED_MODEL_NAME} (CPU)...")
    print("  (first run downloads ~80 MB to ~/.cache/torch/sentence_transformers/)")
    model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
    print(f"  ✓ Loaded — embedding dim = {model.get_sentence_embedding_dimension()}")
    return model


def encode_or_load(model, texts, cache_path, label="(unnamed)"):
    """
    Encode `texts` with the sentence-transformer if cache absent, else load
    the cached numpy array. Cache is invalidated only on row-count mismatch.
    """
    texts = [t if isinstance(t, str) else "" for t in texts]
    if os.path.exists(cache_path):
        try:
            embs = np.load(cache_path)
            if embs.shape[0] == len(texts):
                print(f"  [CACHE HIT]  {label:<24} shape={embs.shape}")
                return embs
            print(f"  [CACHE MISS] {label}: row count changed, re-encoding")
        except Exception as e:
            print(f"  [CACHE ERR]  {label}: {e}, re-encoding")
    print(f"  [ENCODING]   {label:<24} ({len(texts):,} texts)...")
    embs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # keep raw, normalize at use site
    ).astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embs)
    return embs


def encode_split_columns(model, df, split_name,
                         columns=('article', 'question', 'A', 'B', 'C', 'D')):
    """Encode each text column for a split. Returns dict[col] -> ndarray (N, D)."""
    embs = {}
    for col in columns:
        texts = df[col].fillna('').astype(str).tolist()
        cache_path = os.path.join(EMBEDDINGS_DIR, f"{split_name}_{col}.npy")
        embs[col] = encode_or_load(model, texts, cache_path,
                                    label=f"{split_name}/{col}")
    return embs


# ─── Reusable utility — Model B will import this ────────────────────────────
def rank_sentences_by_relevance(passage, query, model, top_k=None):
    """
    Public API for Model B's hint extractor (spec §5.3.2 — extractive
    sentence ranking by question relevance using sentence embeddings).

    Returns sentences ordered by cosine similarity to `query` (descending).
    If `top_k` is set, returns only the top-K. Each entry is (sentence, score).
    """
    sentences = split_sentences(passage)
    if not sentences:
        return []
    sent_embs = model.encode(sentences, batch_size=32, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
    query_emb = model.encode([query], batch_size=1, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = sent_embs @ query_emb  # already normalised → exact cosine
    order = np.argsort(-sims)
    if top_k:
        order = order[:top_k]
    return [(sentences[i], float(sims[i])) for i in order]


# ─── Neural Verification: features + LR head ─────────────────────────────────
def cos_batch(A, B):
    """Pairwise cosine for matched rows in A and B (both shape (N, D))."""
    a_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    b_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + EPS)
    return (a_n * b_n).sum(axis=1).astype(np.float32)


def build_neural_features(embs):
    """
    Per-option semantic features from sentence-transformer embeddings.

    Returns shape (N, 4, 9) per option:
      [p_cos, q_cos, pq_cos,                  ← absolute cosines
       p_diff_max,  p_argmax,                 ← passage rank features
       q_diff_max,  q_argmax,                 ← question rank features
       pq_diff_max, pq_argmax]                ← combined rank features
    """
    p_e  = embs['article']
    q_e  = embs['question']
    opt_e = [embs['A'], embs['B'], embs['C'], embs['D']]

    N = p_e.shape[0]
    pq_e = (p_e + q_e) / 2.0  # mean-pooled passage+question representation

    p_cos  = np.zeros((N, 4), dtype=np.float32)
    q_cos  = np.zeros((N, 4), dtype=np.float32)
    pq_cos = np.zeros((N, 4), dtype=np.float32)

    for j in range(4):
        p_cos[:, j]  = cos_batch(p_e,  opt_e[j])
        q_cos[:, j]  = cos_batch(q_e,  opt_e[j])
        pq_cos[:, j] = cos_batch(pq_e, opt_e[j])

    p_diff,  _, p_argmax  = vs_others(p_cos)
    q_diff,  _, q_argmax  = vs_others(q_cos)
    pq_diff, _, pq_argmax = vs_others(pq_cos)

    return np.stack([
        p_cos, q_cos, pq_cos,
        p_diff,  p_argmax,
        q_diff,  q_argmax,
        pq_diff, pq_argmax,
    ], axis=2)  # (N, 4, 9)


def train_and_eval_neural_verification(train_X, train_y, val_X, val_y,
                                        test_X, test_y):
    """Train an LR head on neural features, eval on val + test."""
    N, _, F = train_X.shape

    # Flatten to per-option binary
    train_flat   = train_X.reshape(N * 4, F)
    train_y_bin  = np.zeros(N * 4, dtype=np.int32)
    train_y_bin[np.arange(N) * 4 + train_y] = 1

    print("\n[NEURAL-VERIF] Fitting StandardScaler...")
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_flat)
    val_flat   = scaler.transform(val_X.reshape(-1, F))
    test_flat  = scaler.transform(test_X.reshape(-1, F))

    print(f"[NEURAL-VERIF] Training LR head on {len(train_flat):,} option-examples × {F} features...")
    model = LogisticRegression(C=1.0, max_iter=300, n_jobs=-1, random_state=42)
    model.fit(train_flat, train_y_bin)

    val_probs   = model.predict_proba(val_flat)[:, 1]
    val_scores  = val_probs.reshape(-1, 4)
    val_em, val_em_f1, val_ba, val_bf1 = eval_per_option_metrics(val_scores, val_y)

    test_probs  = model.predict_proba(test_flat)[:, 1]
    test_scores = test_probs.reshape(-1, 4)
    test_em, test_em_f1, test_ba, test_bf1 = eval_per_option_metrics(test_scores, test_y)

    print(f"\n[NEURAL-VERIF] Final metrics:")
    print(f"  Val  EM: {val_em*100:6.2f}%   macro-F1 = {val_em_f1:.4f}")
    print(f"  Test EM: {test_em*100:6.2f}%   macro-F1 = {test_em_f1:.4f}")

    return {
        'val_em':           val_em,
        'val_em_f1':        val_em_f1,
        'val_binary_acc':   val_ba,
        'val_binary_f1':    val_bf1,
        'test_em':          test_em,
        'test_em_f1':       test_em_f1,
        'test_binary_acc':  test_ba,
        'test_binary_f1':   test_bf1,
        'model':            model,
        'scaler':           scaler,
    }


# ─── Neural Generation re-ranker ─────────────────────────────────────────────
def evaluate_neural_generation(df, st_model, label):
    """
    For each row: split passage into sentences, encode all sentences in
    one batch (vastly faster than per-row), pick the most similar sentence
    to the gold answer, then run the classical cloze on that sentence.
    Eval with BLEU-1/-4 (sentence + corpus), ROUGE-1/2/L, METEOR.
    """
    metrics = GenerationMetrics()
    keys = ['bleu_1', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor']
    sums = {k: 0.0 for k in keys}
    n = 0

    print(f"\n[NEURAL-GEN] {label}: collecting sentences for batch encoding...")

    all_sents   = []
    sent_ranges = []   # (start, end) into all_sents per row
    answers     = []
    rows_kept   = []

    for _, row in df.iterrows():
        passage     = str(row.get('article', ''))
        gold_q      = str(row.get('question', '')).strip()
        gold_letter = row.get('answer', None)
        if pd.isna(gold_letter) or gold_letter not in ('A', 'B', 'C', 'D'):
            continue
        gold_a = str(row.get(gold_letter, '')).strip()
        if not gold_q or not gold_a or not passage:
            continue
        sents = split_sentences(passage)
        if not sents:
            sents = [passage]
        sent_ranges.append((len(all_sents), len(all_sents) + len(sents)))
        all_sents.extend(sents)
        answers.append(gold_a)
        rows_kept.append({
            'passage': passage, 'gold_q': gold_q, 'gold_a': gold_a,
            'sentences': sents,
        })

    print(f"  Encoding {len(all_sents):,} passage sentences in batch...")
    all_sent_embs = st_model.encode(
        all_sents, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print(f"  Encoding {len(answers):,} gold answers in batch...")
    answer_embs = st_model.encode(
        answers, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    print(f"  Scoring {len(rows_kept):,} rows...")
    for idx, rdata in enumerate(rows_kept):
        s, e = sent_ranges[idx]
        sims = all_sent_embs[s:e] @ answer_embs[idx]
        best_idx = int(sims.argmax())
        best_sent = rdata['sentences'][best_idx]
        gen_q = cloze_question(best_sent, rdata['gold_a'],
                                passage_for_fallback=rdata['passage'])
        s_metrics = metrics.score(gen_q, rdata['gold_q'])
        for k in keys:
            sums[k] += s_metrics[k]
        n += 1
        if (idx + 1) % 2000 == 0:
            print(f"    scored {idx + 1:,}/{len(rows_kept):,}")

    avg = {k: (sums[k] / n if n else 0.0) for k in keys}
    avg['bleu_corpus']   = metrics.corpus_bleu_at_n(4)
    avg['bleu_1_corpus'] = metrics.corpus_bleu_at_n(1)
    return avg, n


# ─── Final summary ───────────────────────────────────────────────────────────
def _fmt_pct(x):
    return f"{x * 100:>7.2f}%"


def print_neural_classical_comparison(verif, val_gen, test_gen):
    """Side-by-side neural vs classical (numbers from last classical run)."""
    # Classical numbers from latest model_a.py run — kept here as constants
    # so this file is standalone and doesn't need to re-run model_a.py.
    CLASSICAL = {
        'verif_val_em':    0.3918,   # Stacking val EM
        'verif_test_em':   0.3869,   # Stacking test EM
        'gen_val_bleu1':   0.1761,   # baseline overlap heuristic
        'gen_val_bleu1c':  0.2075,
        'gen_val_rouge1':  0.2131,
        'gen_val_meteor':  0.1583,
        'gen_test_bleu1':  0.1745,
        'gen_test_bleu1c': 0.2062,
        'gen_test_rouge1': 0.2113,
        'gen_test_meteor': 0.1575,
    }

    print("\n" + "=" * 92)
    print("NEURAL  vs  CLASSICAL  —  Model A side-by-side")
    print("=" * 92)
    print()
    print("VERIFICATION (per-question Exact Match)")
    print("-" * 92)
    print(f"{'Approach':<40}    {'Val-EM':>10}    {'Test-EM':>10}")
    print(f"{'Classical Stacking (LR+SVM+RF)':<40}    "
          f"{_fmt_pct(CLASSICAL['verif_val_em'])}    {_fmt_pct(CLASSICAL['verif_test_em'])}")
    print(f"{'Neural (MiniLM cosines → LR head)':<40}    "
          f"{_fmt_pct(verif['val_em'])}    {_fmt_pct(verif['test_em'])}")
    print()
    print("GENERATION (BLEU-1 / ROUGE-1 / METEOR)")
    print("-" * 92)
    print(f"{'Approach':<40}    {'BLEU-1':>8} {'BLEU-1-c':>9} "
          f"{'ROUGE-1':>8} {'METEOR':>7}   ({'split':<5})")
    print(f"{'Classical Baseline (overlap)':<40}    "
          f"{CLASSICAL['gen_val_bleu1']:>8.4f} {CLASSICAL['gen_val_bleu1c']:>9.4f} "
          f"{CLASSICAL['gen_val_rouge1']:>8.4f} {CLASSICAL['gen_val_meteor']:>7.4f}   (val)")
    print(f"{'Classical Baseline (overlap)':<40}    "
          f"{CLASSICAL['gen_test_bleu1']:>8.4f} {CLASSICAL['gen_test_bleu1c']:>9.4f} "
          f"{CLASSICAL['gen_test_rouge1']:>8.4f} {CLASSICAL['gen_test_meteor']:>7.4f}   (test)")
    print(f"{'Neural ranker (MiniLM cosine)':<40}    "
          f"{val_gen['bleu_1']:>8.4f} {val_gen['bleu_1_corpus']:>9.4f} "
          f"{val_gen['rouge1']:>8.4f} {val_gen['meteor']:>7.4f}   (val)")
    print(f"{'Neural ranker (MiniLM cosine)':<40}    "
          f"{test_gen['bleu_1']:>8.4f} {test_gen['bleu_1_corpus']:>9.4f} "
          f"{test_gen['rouge1']:>8.4f} {test_gen['meteor']:>7.4f}   (test)")
    print("=" * 92 + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 70)
    print("# Model A — Neural Approach (sentence-transformers)")
    print("# Separate from model_a.py classical pipeline for fair comparison")
    print("#" * 70)

    if not _ST_AVAILABLE:
        print("\n[ERR] sentence-transformers not installed.")
        print("      Install with: pip install sentence-transformers")
        print("      (also installs torch + transformers — sizeable download)")
        return

    # ─── Load splits ─────────────────────────────────────────────────────
    print("\n[DATA] Loading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV_PATH).head(TRAIN_SUBSET)
    val_df   = pd.read_csv(VAL_CSV_PATH)
    test_df  = pd.read_csv(TEST_CSV_PATH)
    print(f"  Train: {len(train_df):,} (subsampled to keep CPU runtime tractable)")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    # ─── Load neural model + encode (cached) ─────────────────────────────
    st_model = load_neural_model()

    print("\n[NEURAL] Encoding texts for verification (cached on disk)...")
    train_embs = encode_split_columns(st_model, train_df, 'train_subset')
    val_embs   = encode_split_columns(st_model, val_df,   'val')
    test_embs  = encode_split_columns(st_model, test_df,  'test')

    # ─── Build features for verification ──────────────────────────────────
    print("\n[NEURAL] Building per-option semantic features...")
    train_X = build_neural_features(train_embs)
    val_X   = build_neural_features(val_embs)
    test_X  = build_neural_features(test_embs)
    print(f"  Shapes: train {train_X.shape}, val {val_X.shape}, test {test_X.shape}")

    train_y = train_df['answer'].map(ANSWER_MAP).values.astype(np.int32)
    val_y   = val_df['answer'].map(ANSWER_MAP).values.astype(np.int32)
    test_y  = test_df['answer'].map(ANSWER_MAP).values.astype(np.int32)

    # ─── Neural verification ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Neural Verification — sentence-transformer features → LR head")
    print("=" * 70)
    verif = train_and_eval_neural_verification(
        train_X, train_y, val_X, val_y, test_X, test_y
    )

    print(f"\n[NEURAL] Saving model + scaler to {MODEL_NEURAL_PATH}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({'model': verif['model'], 'scaler': verif['scaler']},
                MODEL_NEURAL_PATH)
    print("  ✓ Saved")

    # ─── Neural generation re-ranker ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Neural Generation — sentence-transformer ranker (cloze on top)")
    print("=" * 70)
    val_gen,  _ = evaluate_neural_generation(val_df,  st_model, "VAL")
    test_gen, _ = evaluate_neural_generation(test_df, st_model, "TEST")

    print(f"\n[NEURAL-GEN] Val:  BLEU-1={val_gen['bleu_1']:.4f}  "
          f"BLEU-1-c={val_gen['bleu_1_corpus']:.4f}  "
          f"ROUGE-1={val_gen['rouge1']:.4f}  METEOR={val_gen['meteor']:.4f}")
    print(f"[NEURAL-GEN] Test: BLEU-1={test_gen['bleu_1']:.4f}  "
          f"BLEU-1-c={test_gen['bleu_1_corpus']:.4f}  "
          f"ROUGE-1={test_gen['rouge1']:.4f}  METEOR={test_gen['meteor']:.4f}")

    # ─── Side-by-side with classical ─────────────────────────────────────
    print_neural_classical_comparison(verif, val_gen, test_gen)


if __name__ == "__main__":
    main()
