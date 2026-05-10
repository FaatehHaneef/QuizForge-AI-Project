# QuizForge AI

**Intelligent Reading Comprehension and Quiz Generation System using Machine Learning**
AL2002 Artificial Intelligence -- BS (CS) Spring 2026 -- FAST NUCES Islamabad

A two-model pipeline that takes a reading comprehension passage and produces a
fully formed multiple-choice quiz. **Model A** verifies user-selected answers
against gold options and generates fresh Wh-template cloze questions.
**Model B** generates three plausible distractors and three graduated hints
for any question. A Streamlit UI ties the whole pipeline together and adds
nine animated brainrot screens for live demonstration.

## Authors
- **Syed Muhammad Faateh Haneef**
- **Muhammad Ibrahim Malik**

Honorary Mention: Claude

---

## What's in the box

| Layer | Files | What it does |
|---|---|---|
| Classical ML pipeline | `src/model_a.py` | Logistic Regression, Linear SVM, Random Forest, Naive Bayes; soft-voting + stacking ensembles; K-Means unsupervised; Wh-template question generation with sentence rankers; full BLEU/ROUGE/METEOR evaluation |
| Neural baseline | `src/model_a_neural.py` | sentence-transformers (`all-MiniLM-L6-v2`) verifier + reusable `rank_sentences_by_relevance` helper |
| Distractor + hint generator | `src/model_b.py` | LR / RF / HGB distractor rankers; TF-IDF + ML-scored hint rankers; RandomForest hint regressor; Likert template + Confusion Matrix scaffolding for human evaluation |
| Inference API | `src/inference.py` | Singleton loader. Exposes `generate_quiz(passage)`, `load_random_race_sample()`, `verify_answer()` to the UI |
| Streamlit UI | `ui/app.py` | Nine screens (welcome, article input, loading, quiz, correct, wrong, hints, dashboard, outro), brainrot meme picker, per-screen palettes, session log + CSV export |
| EDA | `notebooks/eda.ipynb` | Twelve cells covering missing values, IQR outliers, correlation heatmap, feature-relationship analysis. Executed end-to-end with outputs baked. |
| Run logs | `runs/model_a_run.txt`, `runs/model_b_run.txt` | Full terminal output of the most recent reference runs |
| Trained checkpoints | `models/*.pkl` | All persisted artefacts (vectoriser, GloVe IDF lookup, classifiers, rankers) -- regenerable by re-running the source files |
| Likert rater scores | `models/model_b_distractor_likert_template.csv` | Filled-in 1-5 ratings of generated and gold distractors for the human-evaluation deliverable |
| UI screenshots | `ui/screenshots/` | Nine PNGs covering the full user flow for the report |

---

## Final metrics -- the full table

### Model A -- Verification (per-question Exact Match + binary peer-comparable)

| Approach                  | Task          | Val-EM     | Val-F1   | Test-EM   | Test-F1  | BinAcc     | BinF1    |
|---------------------------|---------------|------------|----------|-----------|----------|------------|----------|
| Logistic Regression       | Answer Verif. | 36.42%     | 0.3634   | 35.97%    | 0.3588   | **73.06%** | 0.5441   |
| Linear SVM                | Answer Verif. | 35.53%     | 0.3548   | 34.99%    | 0.3492   | **74.99%** | 0.4286   |
| Random Forest             | Answer Verif. | **39.32%** | 0.3928   | --        | --       | 71.25%     | 0.5879   |
| Naive Bayes               | Question Type | **91.65%** | 0.8150   | --        | --       | --         | --       |
| K-Means Clustering        | Clustering    | 27.53%     | --       | --        | --       | --         | --       |
| Ensemble -- Soft Voting   | Answer Verif. | 38.03%     | 0.3798   | 37.58%    | 0.3746   | 60.03%     | 0.5575   |
| Ensemble -- **Stacking**  | Answer Verif. | **39.18%** | 0.3914   | **38.69%**| 0.3857   | **75.59%** | 0.4897   |
| Neural baseline (MiniLM)  | Answer Verif. | 34.20%     | 0.3414   | 33.81%    | 0.3380   | --         | --       |

Headline: **Stacking ensemble at 75.59% binary accuracy** is the cleanest peer-comparable
verification number. Per-question 4-way EM peaks at 39.32% (RF) / 39.18% (Stacking),
which is the appropriate metric for the spec's argmax-over-options framing.
Naive Bayes classifies *question types* (detail / inference / main idea / vocabulary / other),
so its 91.65% accuracy is on a different task and not directly comparable to the verifiers.

### Model A -- Unsupervised (K-Means)

| Metric           | Value  | Notes                                         |
|------------------|--------|-----------------------------------------------|
| Cluster purity   | 27.53% | Above the 25% random baseline                 |
| Silhouette score | 0.0266 | Range [-1, 1] -- positive but small (expected for high-dim sparse text) |

### Model A -- Question Generation (Wh-template cloze, BLEU/ROUGE/METEOR)

| Variant              | Split | BLEU-1 | BLEU-1-c | BLEU-4 | BLEU-4-c | ROUGE-1 | ROUGE-L | METEOR |
|----------------------|-------|--------|----------|--------|----------|---------|---------|--------|
| **Baseline (overlap)** | VAL   | **0.1766** | **0.2082** | 0.0426 | 0.0419 | **0.2134** | **0.1924** | **0.1588** |
| Baseline (overlap)   | TEST  | 0.1748 | 0.2065   | 0.0426 | 0.0420   | 0.2117  | 0.1908  | 0.1578 |
| Ranker -- LR         | VAL   | 0.1697 | 0.1883   | 0.0387 | 0.0314   | 0.2025  | 0.1840  | 0.1419 |
| Ranker -- LR         | TEST  | 0.1682 | 0.1860   | 0.0391 | 0.0322   | 0.2009  | 0.1832  | 0.1411 |
| Ranker -- NB         | VAL   | 0.1732 | 0.2042   | 0.0420 | 0.0411   | 0.2099  | 0.1891  | 0.1565 |
| Ranker -- NB         | TEST  | 0.1706 | 0.2016   | 0.0414 | 0.0405   | 0.2073  | 0.1870  | 0.1541 |
| Ranker -- SVM        | VAL   | 0.1692 | 0.1839   | 0.0387 | 0.0300   | 0.2013  | 0.1834  | 0.1395 |
| Ranker -- SVM        | TEST  | 0.1680 | 0.1826   | 0.0389 | 0.0309   | 0.1999  | 0.1828  | 0.1392 |

### Model B -- Distractor Generation

| Ranker | Split | Acc        | Precision  | Recall   | F1       | BLEU-1   | BLEU-1-c | ROUGE-1  | ROUGE-L  | METEOR   |
|--------|-------|------------|------------|----------|----------|----------|----------|----------|----------|----------|
| **LR**  | VAL  | **1.0000** | **0.1479** | 0.1437   | **0.1457** | **0.1068** | 0.1522 | 0.1390 | 0.1268 | 0.1012 |
| LR     | TEST  | 1.0000     | 0.1404     | 0.1364   | 0.1384   | 0.1059   | 0.1530   | 0.1396   | 0.1279   | 0.1006   |
| RF     | VAL   | 1.0000     | 0.1358     | 0.1320   | 0.1339   | 0.0973   | 0.1468   | 0.1284   | 0.1179   | 0.0907   |
| RF     | TEST  | 1.0000     | 0.1296     | 0.1260   | 0.1278   | 0.0944   | 0.1469   | 0.1271   | 0.1173   | 0.0876   |
| HGB    | VAL   | 1.0000     | 0.1302     | 0.1266   | 0.1284   | 0.0939   | 0.1443   | 0.1242   | 0.1143   | 0.0864   |
| HGB    | TEST  | 1.0000     | 0.1238     | 0.1203   | 0.1220   | 0.0925   | 0.1448   | 0.1239   | 0.1148   | 0.0851   |

`Acc = 1.0000` here is the rubric-defined *distractor ranker accuracy*: the fraction
of rows where the top-ranked candidate is **not** the correct answer. We filter the
correct answer out at generation time, so this is 100% by construction (sanity check).

### Model B -- Hint Generation

| Variant               | Split | P@3        | BLEU-1   | BLEU-1-c | ROUGE-1  | ROUGE-L  | METEOR   |
|-----------------------|-------|------------|----------|----------|----------|----------|----------|
| TF-IDF cosine         | VAL   | 0.4179     | **0.2303** | **0.2658** | **0.2756** | **0.2409** | **0.2246** |
| TF-IDF cosine         | TEST  | 0.4008     | 0.2336   | 0.2708   | 0.2762   | 0.2418   | 0.2256   |
| **ML-scored (LR)**    | VAL   | **0.7842** | 0.1956   | 0.2292   | 0.2417   | 0.1997   | 0.1863   |
| **ML-scored (LR)**    | TEST  | **0.7800** | 0.2045   | 0.2449   | 0.2507   | 0.2097   | 0.1970   |

Headline: **ML-scored hint ranker delivers Precision@3 of 78%** -- nearly double the
TF-IDF cosine baseline. This is the strongest single result in Model B.

### Model B -- Regression Hint Scorer (rubric R^2 metric)

| Split | N      | R^2        | MAE    | RMSE   |
|-------|--------|------------|--------|--------|
| VAL   | 46,583 | **0.4932** | 0.1058 | 0.1444 |
| TEST  | 46,331 | **0.4985** | 0.1052 | 0.1438 |

### Model B -- Confusion Matrix from Human Likert Evaluation

Filled rater scores at `models/model_b_distractor_likert_template.csv`.
Both columns (generated distractors and gold distractors) scored 1-5 by the rater.
Generated mean ~ **3.0**, gold mean ~ **4.4** -- a coherent gap consistent with
extractive generation versus author-crafted gold.

### Stacking ensemble Confusion Matrix (Val, rows = true, cols = predicted)

```
            A       B       C       D
true=A    749     345     397     463
true=B    433     864     433     554
true=C    458     456     938     559
true=D    365     407     474     892
```

Diagonal sum = 3,443 / 8,787 = 39.18% (matches reported EM).

---

## Setup

Python 3.11 recommended. From the project root:

```powershell
# (Optional but recommended) virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# UTF-8 stdout encoding (Windows console safety)
$env:PYTHONIOENCODING = 'utf-8'
```

The dataset (RACE) lives at `data/processed/{train,val,test}.csv` and is loaded
automatically by both training scripts and the UI. Raw / feature / embedding
folders are gitignored as they are large and reproducible.

---

## Reproducing the trained models

Run in this order. The first command builds artefacts that the others depend on:

```powershell
# 1. Model A -- classical pipeline (~30-45 min)
python -u src/model_a.py 2>&1 | Tee-Object -FilePath runs/model_a_run.txt

# 2. Model B -- distractor + hint generation (~5-7 min, depends on Model A's vectoriser)
python -u src/model_b.py 2>&1 | Tee-Object -FilePath runs/model_b_run.txt

# 3. (Optional) Neural Model A baseline -- requires sentence-transformers (~25-30 min)
python -u src/model_a_neural.py 2>&1 | Tee-Object -FilePath runs/model_a_neural_run.txt
```

All three populate `models/` with trained artefacts.

---

## Running the user interface

After the training scripts have populated `models/`:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python -m streamlit run ui/app.py
```

Streamlit opens `http://localhost:8501` automatically. The first quiz takes
10-15 seconds because GloVe and the rankers load once into memory; subsequent
quizzes are sub-second.

The UI flow:
1. **Welcome** -- hero meme + two buttons
2. **Article Input** -- paste a passage or load a random RACE sample
3. **Loading** -- inference runs (full-screen overlay with crash-out gif)
4. **Quiz** -- question + 4 options + Check button
5. **Correct / Wrong** -- popup with reveal
6. **Hints** -- three graduated hints (general -> specific -> near-explicit) with progressive unlock
7. **Dashboard** -- session metrics + CSV export
8. **Outro** -- final card with "Demo khatam" button to terminate the server

---

## Folder layout

```
project_root/
+-- data/
|   +-- processed/          # train.csv, val.csv, test.csv (RACE)
|   +-- raw/                # gitignored, regenerable
|   +-- features/           # gitignored, regenerable
|   +-- embeddings/         # gitignored, regenerable
|   +-- session_log.csv     # rolling demo session log
+-- models/
|   +-- *.pkl               # trained classifiers, rankers, vectoriser
|   +-- model_b_distractor_likert_template.csv  # filled rater scores
+-- notebooks/
|   +-- eda.ipynb           # 12-cell EDA notebook (executed)
+-- runs/
|   +-- model_a_run.txt     # latest Model A reference run
|   +-- model_a_neural_run.txt
|   +-- model_b_run.txt
+-- src/
|   +-- model_a.py          # classical Model A (verification + generation)
|   +-- model_a_neural.py   # neural Model A baseline (sentence-transformers)
|   +-- model_b.py          # Model B distractor + hint generation
|   +-- inference.py        # unified API for the UI
|   +-- preprocessing.py    # tokenisation + cleaning helpers
|   +-- feature_engineering.py
|   +-- utils.py
|   +-- config.py           # data + model paths
+-- ui/
|   +-- app.py              # Streamlit application
|   +-- assets/memes/       # 9 brainrot meme categories
|   +-- screenshots/        # 9 UI screenshots for the report
+-- AI_Project_Complete_Context.txt   # course spec
+-- requirements.txt
+-- README.md               # this file
```

---

## Constraints honoured

- All models train on a standard CPU (no GPU dependency)
- Inference for a single passage completes well under the 10-second per-query
  budget after the one-time model load (sub-second steady state)
- Full pipeline reproducible from `python src/model_a.py && python src/model_b.py`
- All four UI screens specified by the rubric are present plus five additional
  screens for a richer demo

## Limitations

- **Distractor generation F1 ~ 0.15**. This is a structural ceiling for
  extractive distractor mining against author-crafted gold distractors --
  reaching meaningfully higher requires neural fine-tuning that is impractical
  on CPU.
- **First-call latency ~ 10-15 seconds** for the UI as GloVe (66 MB) loads
  once into memory. Subsequent quizzes are sub-second.
- **Cloze generation occasionally falls back to a generic Wh-template** when
  no candidate produces a clean cloze; the retry loop tries five alternates
  before settling for the template.

## Acknowledgements

- **RACE dataset**: Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017).
  *RACE: Large-scale ReAding Comprehension Dataset From Examinations.*
- **GloVe embeddings**: `glove-wiki-gigaword-50` via gensim
- **Sentence Transformers**: `all-MiniLM-L6-v2` (Reimers & Gurevych)
- **Evaluation metrics**: BLEU (Papineni 2002), ROUGE (Lin 2004),
  METEOR (Banerjee & Lavie 2005)
