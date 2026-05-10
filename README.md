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
- Faateh Haneef
- Ibrahim (Ibbo)

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

## Final headline numbers

**Model A -- Verification (Stacking ensemble, the strongest):**
- Val EM **39.18%** -- macro-F1 **0.3914**
- Test EM **38.69%** -- macro-F1 **0.3857**
- Confusion Matrix: see `runs/model_a_run.txt`

**Model A -- Unsupervised (K-Means):**
- Cluster purity **27.53%** -- silhouette **0.0266**

**Model A -- Generation (Baseline cloze on test):**
- BLEU-1 **0.1748** -- corpus BLEU-1 **0.2065** -- ROUGE-1 **0.2117** -- METEOR **0.1578**

**Model B -- Distractor Generation (LR ranker, the strongest):**
- Val P/R/F1 **0.1479 / 0.1437 / 0.1457**
- Test P/R/F1 **0.1404 / 0.1364 / 0.1384**

**Model B -- Hint Generation:**
- ML-scored ranker: P@3 **0.7842 / 0.7800** (val/test)
- TF-IDF baseline: P@3 0.4179 / 0.4008
- RandomForest hint regressor R^2 **0.4932 / 0.4985**

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
