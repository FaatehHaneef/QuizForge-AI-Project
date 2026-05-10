"""
QuizForge AI -- Unified Inference API
=====================================

Single entry point used by the Streamlit UI. Loads every trained
artefact from the models/ directory once at import time, then exposes
two top-level functions:

  generate_quiz(passage)
      Run the full pipeline on an arbitrary passage:
        1. Pick a salient phrase from the passage as the correct answer
           (Model B candidate extractor).
        2. Generate a cloze question via Model A's Wh-template generator.
        3. Generate three distractors via Model B's LR ranker.
        4. Extract three graduated hints (general -> near-explicit) via
           Model B's TF.IDF cosine sentence ranker.
      Returns a dict with keys: question, correct_answer, options
      (already shuffled), distractors, hints, latency_ms (per stage)

  load_random_race_sample(split='val')
      Returns one randomly drawn row from the RACE split, already
      shaped as a quiz dict (question, correct_answer, options, hints)

  verify_answer(picked, correct)
      String-equality check. Kept as a function so the UI doesn't have
      to know how verification works internally.

Reproducibility: every random choice routes through a single Random
instance seeded by `seed` if passed; otherwise OS entropy.

This module assumes both `model_a.py` and `model_b.py` have been run
end-to-end at least once (their .pkl artefacts must exist in models/)
"""

import os
import re
import sys
import time
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    TEST_CSV_PATH,
    MODELS_DIR,
)

# Reuse heavy classical infrastructure from Model A and Model B
from model_a import (
    load_glove,
    build_idf_lookup,
    generate_question,
    fit_tfidf_vectorizer,
    split_sentences,
    _generic_template_for,
    GENERIC_QG_TEMPLATES,
    MAX_QUESTION_TOKENS,
    TFIDF_VEC_PATH,
)

# Set of generic templates as a fast membership check -- used to detect when
# `generate_question` fell back to a non-cloze template, which means the
# question doesn't actually correspond to the picked answer span.
_GENERIC_TEMPLATE_SET = set(GENERIC_QG_TEMPLATES)
from model_b import (
    extract_candidate_phrases,
    generate_distractors,
    extract_hints_tfidf,
    extract_hints_ml,
    MODEL_B_DIST_LR_PATH,
    MODEL_B_HINT_PATH,
)


# --- Singletons loaded once at import time ----------------------------------
_ARTEFACTS: Dict[str, object] = {}


def _ensure_loaded():
    """Idempotent loader. Streamlit's @st.cache_resource wraps this."""
    if _ARTEFACTS:
        return _ARTEFACTS

    print("[INFERENCE] Loading TF.IDF vectoriser...")
    if os.path.exists(TFIDF_VEC_PATH):
        vec = joblib.load(TFIDF_VEC_PATH)
    else:
        print("[INFERENCE] No saved vectoriser; fitting fresh on training corpus...")
        vec = fit_tfidf_vectorizer(TRAIN_CSV_PATH)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(vec, TFIDF_VEC_PATH)

    print("[INFERENCE] Loading GloVe (one-time, ~10s)...")
    glove = load_glove()
    idf_lookup = build_idf_lookup(vec)

    print("[INFERENCE] Loading Model B distractor ranker (LR)...")
    distractor_ranker = (
        joblib.load(MODEL_B_DIST_LR_PATH)
        if os.path.exists(MODEL_B_DIST_LR_PATH) else None
    )

    print("[INFERENCE] Loading Model B hint ranker (LR)...")
    hint_ranker = (
        joblib.load(MODEL_B_HINT_PATH)
        if os.path.exists(MODEL_B_HINT_PATH) else None
    )

    print("[INFERENCE] Loading RACE val/test for sample-loader...")
    val_df = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    _ARTEFACTS.update({
        'vec': vec,
        'glove': glove,
        'idf_lookup': idf_lookup,
        'distractor_ranker': distractor_ranker,
        'hint_ranker': hint_ranker,
        'val_df': val_df,
        'test_df': test_df,
    })
    print(f"[INFERENCE] Ready -- {len(val_df):,} val + {len(test_df):,} test rows.")
    return _ARTEFACTS


# --- Helpers ----------------------------------------------------------------
_SENTENCE_END = re.compile(r'[.!?]\s*$')

# Words a phrase must NOT start with -- covers determiners, pronouns,
# and prepositions that produce ungrammatical answer options like
# "In September" or "Of London".
_BAD_START = re.compile(
    r'^('
    r'the|a|an|this|these|those|that|they|it|he|she|we|i|you|'
    r'in|on|at|by|from|with|for|of|to|as|but|and|or|nor|so|yet|'
    r'is|are|was|were|be|been|being|has|have|had|do|does|did'
    r')\b',
    re.IGNORECASE,
)


def _sentence_for_phrase(phrase: str, sentences):
    """Return the first sentence containing `phrase` (case-insensitive)."""
    pl = phrase.lower()
    for s in sentences:
        if pl in s.lower():
            return s
    return None


def _is_entity_like(phrase: str) -> bool:
    """
    A 'good answer' is a noun phrase / entity:
      - 2-5 tokens (single-word picks like 'Chain' or 'Fleming' are too
        ambiguous for a cloze quiz; 6+ words usually means a sentence)
      - No sentence-ending punctuation
      - Doesn't start with a stop-word (determiner, pronoun, preposition)
    """
    p = phrase.strip()
    if not p:
        return False
    n = len(p.split())
    if n < 2 or n > 5:
        return False
    if _SENTENCE_END.search(p):
        return False
    if _BAD_START.match(p):
        return False
    return True


def _pick_correct_answer(passage: str, rng: random.Random) -> str:
    """
    Pick a salient candidate phrase from the passage as the 'correct answer'.

    Tiered strategy -- earlier tiers produce cleaner cloze questions:

      1. Entity-like phrase (2-5 words) found in a SHORT sentence (<=22 tokens),
         so substituting the answer with a Wh-word stays within the
         MAX_QUESTION_TOKENS cap. Best path -- clean, unambiguous cloze.
      2. Entity-like phrase from any sentence -- accepts a slightly long
         sentence but still avoids single-word ambiguity.
      3. Any 1-5 word phrase not starting with a stop-word -- last resort
         before falling back to a default.
    """
    cands = extract_candidate_phrases(passage)
    sentences = split_sentences(passage)
    short_sents = [s for s in sentences if len(s.split()) <= MAX_QUESTION_TOKENS - 3]

    if not cands:
        words = [w for w in re.findall(r'\b[A-Z][a-z]+\b', passage) if len(w) >= 4]
        return words[0] if words else "the main idea of the passage"

    # Tier 1: entity-like AND in a sentence short enough for clean cloze
    tier1 = []
    for c in cands:
        if not _is_entity_like(c):
            continue
        host = _sentence_for_phrase(c, sentences)
        if host is not None and len(host.split()) <= MAX_QUESTION_TOKENS - 3:
            tier1.append(c)
    if tier1:
        return rng.choice(tier1)

    # Tier 2: entity-like from any sentence
    tier2 = [c for c in cands if _is_entity_like(c)]
    if tier2:
        return rng.choice(tier2)

    # Tier 3: any 1-5 word phrase that doesn't start with a stop-word
    tier3 = [
        c for c in cands
        if 1 <= len(c.split()) <= 5
        and not _SENTENCE_END.search(c)
        and not _BAD_START.match(c)
    ]
    if tier3:
        return rng.choice(tier3)

    return rng.choice(cands)


def _is_generic_template(question: str) -> bool:
    """True if `generate_question` fell back to a content-free template."""
    return question.strip() in _GENERIC_TEMPLATE_SET


# Phrase-cleanup: candidate extractor's regex sometimes captures fragments
# like "In September", "World War" (truncated before "II"), "Medicine in"
# (trailing preposition). Strip these dangling stop-words on either end.
_BAD_EDGE_WORDS = {
    'in', 'on', 'at', 'by', 'of', 'from', 'with', 'for', 'to', 'as',
    'but', 'and', 'or', 'nor', 'so', 'yet', 'the', 'a', 'an',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did',
}

# Words that often start sentences and look "proper-noun-ish" because
# they're capitalised, but are actually meaningless as MCQ options:
# 'Very', 'Some', 'Also', 'If', 'When'...
_COMMON_OPENERS = {
    'very', 'some', 'also', 'if', 'when', 'where', 'why', 'how',
    'then', 'now', 'here', 'there', 'these', 'those',
    'all', 'most', 'every', 'each', 'both', 'neither', 'either',
    'not', 'only', 'even', 'just', 'well', 'yes', 'no',
    'however', 'although', 'though', 'while', 'since', 'because',
    'after', 'before', 'during', 'until', 'despite',
    'first', 'second', 'third', 'fourth', 'last', 'next', 'previous',
    'many', 'few', 'several', 'much', 'more', 'less',
    'still', 'yet', 'already', 'finally', 'always', 'never', 'often',
    'rather', 'quite', 'almost', 'nearly', 'really',
    'maybe', 'perhaps', 'probably', 'definitely',
    'today', 'tomorrow', 'yesterday',
    'good', 'bad', 'big', 'small', 'long', 'short', 'old', 'new',
}
_PASSAGE_PHRASE_CACHE: Dict[int, List[str]] = {}


def _normalize_passage(passage: str) -> str:
    """RACE passages frequently miss spaces after sentence punctuation
    ('Memory.We only...' instead of 'Memory. We only...'). Inject the missing
    space so `split_sentences` actually splits on real boundaries instead
    of treating whole paragraphs as one 'sentence'."""
    return re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', passage)


def _clean_phrase(phrase: str) -> str:
    """Strip leading/trailing stop-words and stray punctuation/quotes.
    'In September'->'September', 'Medicine in'->'Medicine',
    '"said the teacher'->'said the teacher'."""
    s = phrase.strip().strip(' "\'`()[]{}.,;:!?-')
    tokens = s.split()
    while tokens and tokens[0].lower() in _BAD_EDGE_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in _BAD_EDGE_WORDS:
        tokens.pop()
    return ' '.join(tokens).strip(' "\'`,.;:!?-')


def _try_extend_with_passage(phrase: str, passage: str) -> str:
    """If the phrase appears in the passage with a meaningful continuation
    ('World War' -> 'World War II', 'Penicillium' -> 'Penicillium notatum',
    '1928' -> 'September 1928'), return the extended form."""
    if not phrase or not passage:
        return phrase
    pl_low = passage.lower()
    p_low = phrase.lower()
    idx = pl_low.find(p_low)
    if idx < 0:
        return phrase
    # Look forward up to 3 tokens for an extension (capitalised word, Roman
    # numeral, or 4-digit year).
    tail = passage[idx + len(phrase):][:40]
    m = re.match(r'\s+([A-Z][A-Za-z]+|\b(?:I{1,4}V?|IV|V|VI{1,3}|IX|X)\b|\d{4})',
                 tail)
    if m:
        ext = phrase + ' ' + m.group(1)
        # Sanity: don't blow up to a sentence
        if len(ext.split()) <= 6:
            return ext
    # Look backward for a leading capitalised word (eg. 'September' -> 'In September' is bad,
    # but '1928' -> 'September 1928' is good -- but only if the preceding word is capitalised).
    head = passage[max(0, idx - 30):idx]
    m2 = re.search(r'([A-Z][A-Za-z]+)\s+$', head)
    if m2 and len(phrase.split()) == 1 and phrase[0].isdigit():
        return m2.group(1) + ' ' + phrase
    return phrase


def _is_acceptable_option(phrase: str) -> bool:
    """A distractor option is acceptable if:
      - at least 2 chars,
      - doesn't start/end with a stop-word,
      - 1 to 10 tokens (full sentences are bad MCQ options),
      - if a single token, must be a number OR a capitalised noun
        that ISN'T a common sentence-opener (Very, Some, Also, If...).
    """
    p = phrase.strip()
    if len(p) < 2:
        return False
    if _BAD_START.match(p):
        return False
    tokens = p.split()
    if not tokens:
        return False
    if len(tokens) > 10:  # likely a full sentence
        return False
    if tokens[-1].lower() in _BAD_EDGE_WORDS:
        return False
    # Reject phrases made entirely of stop-words / common openers
    if all(t.lower() in _COMMON_OPENERS or t.lower() in _BAD_EDGE_WORDS
           for t in tokens):
        return False
    if len(tokens) >= 2:
        return True
    # Single token rules
    tok = tokens[0]
    if tok.lower() in _COMMON_OPENERS:
        return False
    if tok[0].isdigit():
        return True
    # Require capitalised AND >=4 chars to skip 1-3 letter junk like 'If', 'No'
    return tok[0].isupper() and len(tok) >= 4


def _phrase_type(phrase: str) -> str:
    """
    Coarse semantic type classification so distractors of mixed types (year /
    proper-noun / common-noun) don't get jumbled together. Returns one of:
      - 'year'        : 4-digit number (1944, 2026)
      - 'date_phrase' : year preceded by a month or 'By' (September 1928)
      - 'proper_noun' : every token starts with capital (Nobel Prize, Howard Florey)
      - 'numeric'     : any other digit-starting phrase
      - 'common'      : everything else
    """
    p = phrase.strip()
    if not p:
        return 'common'
    if re.fullmatch(r'\d{4}', p):
        return 'year'
    if re.search(r'\d{4}', p) and any(t[0].isupper() for t in p.split() if t):
        return 'date_phrase'
    if re.match(r'^\d', p):
        return 'numeric'
    tokens = [t for t in p.split() if t]
    if tokens and all(t[0].isupper() for t in tokens):
        return 'proper_noun'
    return 'common'


def _polish_options(distractors: List[str], correct: str, passage: str) -> List[str]:
    """Run cleanup + extend + acceptance check + same-type filter over distractors.
    Returns up to 3 polished, deduped options whose semantic type matches the
    correct answer's (so 'Nobel Prize' can't sit next to '1944' and 'Medicine')."""
    seen_low = {correct.lower().strip()}
    target_type = _phrase_type(correct)
    out: List[str] = []
    for d in distractors:
        cleaned = _clean_phrase(d)
        cleaned = _try_extend_with_passage(cleaned, passage)
        if not _is_acceptable_option(cleaned):
            continue
        if _phrase_type(cleaned) != target_type:
            continue
        key = cleaned.lower().strip()
        if key in seen_low:
            continue
        seen_low.add(key)
        out.append(cleaned)
        if len(out) >= 3:
            break
    return out


def _shuffle_options(correct: str, distractors: List[str], rng: random.Random) -> List[Dict]:
    """Return [{label, text, is_correct}] in randomised A/B/C/D order."""
    pool = [(correct, True)] + [(d, False) for d in distractors[:3]]
    rng.shuffle(pool)
    return [
        {'label': L, 'text': t, 'is_correct': c}
        for L, (t, c) in zip(['A', 'B', 'C', 'D'], pool)
    ]


# --- Public API -------------------------------------------------------------
def generate_quiz(passage: str, seed: Optional[int] = None) -> Dict:
    """
    Run the full QuizForge pipeline on `passage`.

    Returns:
        {
            'question': str,
            'correct_answer': str,
            'options': [{'label': 'A', 'text': '...', 'is_correct': bool}, ...],
            'distractors': [str, str, str],
            'hints': [str, str, str],   # graduated: general -> near-explicit
            'latency_ms': {'total': int, 'q_gen': int, 'distractors': int, 'hints': int},
        }
    """
    a = _ensure_loaded()
    rng = random.Random(seed)
    t0 = time.perf_counter()

    # Normalise so split_sentences actually splits -- RACE passages often
    # have missing spaces after periods which collapse into mega-sentences.
    passage = _normalize_passage(passage)

    correct = _pick_correct_answer(passage, rng)
    correct = _try_extend_with_passage(_clean_phrase(correct), passage) or correct

    t1 = time.perf_counter()
    question = generate_question(passage, correct)

    # Quality guard: a 'failed' cloze either produces a stub (<=3 words like
    # 'What?') OR a generic template ('What can we infer from the passage?').
    # In both cases the question doesn't match the picked answer span, so the
    # quiz options become nonsense. Retry with up to 5 different candidates
    # before accepting the failure.
    def _is_bad_question(q: str) -> bool:
        return len(q.split()) <= 3 or _is_generic_template(q)

    if _is_bad_question(question):
        tried = {correct.lower().strip()}
        recovered = False
        for _ in range(5):
            alt = _pick_correct_answer(passage, rng)
            key = alt.lower().strip()
            if key in tried:
                continue
            tried.add(key)
            alt_q = generate_question(passage, alt)
            if not _is_bad_question(alt_q):
                correct = alt
                question = alt_q
                recovered = True
                break
        # Final fallback: if every retry also produced a stub question,
        # accept a generic Wh-template (still readable) over a 'What?' stub.
        if not recovered and len(question.split()) <= 3:
            question = _generic_template_for(passage)

    t2 = time.perf_counter()
    distractors = generate_distractors(
        passage, correct,
        ranker=a['distractor_ranker'],
        vec=a['vec'], glove=a['glove'], idf_lookup=a['idf_lookup'],
        top_k=3,
    ) if a['distractor_ranker'] is not None else []

    # Polish: strip dangling prepositions, extend partial phrases ("World War"
    # -> "World War II"), drop options that fail the acceptance check.
    distractors = _polish_options(distractors, correct, passage)

    # Top up -- pull more same-type candidates from the full extraction pool
    # to bring distractor count to 3. This is the main path when the ranker
    # gave us few same-type matches.
    target_type = _phrase_type(correct)
    if len(distractors) < 3:
        for raw in extract_candidate_phrases(passage):
            cleaned = _clean_phrase(raw)
            cleaned = _try_extend_with_passage(cleaned, passage)
            if not _is_acceptable_option(cleaned):
                continue
            if _phrase_type(cleaned) != target_type:
                continue
            if cleaned.lower().strip() == correct.lower().strip():
                continue
            if cleaned in distractors:
                continue
            distractors.append(cleaned)
            if len(distractors) >= 3:
                break

    # Last-resort: relax the type filter if we still don't have 3
    if len(distractors) < 3:
        for raw in extract_candidate_phrases(passage):
            cleaned = _clean_phrase(raw)
            cleaned = _try_extend_with_passage(cleaned, passage)
            if not _is_acceptable_option(cleaned):
                continue
            if cleaned.lower().strip() == correct.lower().strip():
                continue
            if cleaned in distractors:
                continue
            distractors.append(cleaned)
            if len(distractors) >= 3:
                break

    # Final guarantee -- never fewer than 3 options for the radio buttons
    while len(distractors) < 3:
        filler = ['none of the above', 'all of the above',
                  'cannot be determined from the passage'][len(distractors)]
        distractors.append(filler)

    t3 = time.perf_counter()
    hint_pairs = extract_hints_tfidf(passage, question, a['vec'], top_k=3)
    hints = [s for s, _score in hint_pairs]
    while len(hints) < 3:
        hints.append("(no further hint available)")

    t4 = time.perf_counter()

    return {
        'question': question,
        'correct_answer': correct,
        'options': _shuffle_options(correct, distractors, rng),
        'distractors': distractors,
        'hints': hints,
        'latency_ms': {
            'total':       int(round((t4 - t0) * 1000)),
            'pick_answer': int(round((t1 - t0) * 1000)),
            'q_gen':       int(round((t2 - t1) * 1000)),
            'distractors': int(round((t3 - t2) * 1000)),
            'hints':       int(round((t4 - t3) * 1000)),
        },
    }


_RACE_SAMPLE_CACHE: Dict[str, list] = {}


def load_random_race_sample(split: str = 'val', seed: Optional[int] = None) -> Dict:
    """
    Pull a random RACE row from a 25-row cache (taken from the head of the
    split, filtered to rows with a usable answer + decent passage length).
    Caching means the second+ click of 'Random' is effectively instant --
    no dataframe filter / sample on each call.

    The 25-row pool is small enough that variety is preserved without the
    overhead of scanning 8.7k val rows on every press.
    """
    a = _ensure_loaded()
    rng = random.Random(seed)

    cache = _RACE_SAMPLE_CACHE.get(split)
    if cache is None:
        df = a['val_df'] if split == 'val' else a['test_df']
        valid = df[df['answer'].isin(['A', 'B', 'C', 'D'])]
        valid = valid[valid['article'].astype(str).str.len() > 50]
        if len(valid) == 0:
            raise RuntimeError("No usable RACE rows found.")
        cache = valid.head(25).to_dict('records')
        _RACE_SAMPLE_CACHE[split] = cache
        print(f"[INFERENCE] Cached {len(cache)} RACE-{split} samples for the random loader.")

    row = rng.choice(cache)
    # Normalise the passage so hint extraction doesn't get whole paragraphs
    # back as 'sentences' (RACE passages frequently lack spaces after periods).
    row = dict(row)
    row['article'] = _normalize_passage(str(row.get('article', '')))

    passage = str(row['article'])
    question = str(row['question'])
    letter = row['answer']
    correct = str(row[letter]).strip()
    gold_distractors = [
        str(row[L]).strip() for L in ('A', 'B', 'C', 'D')
        if L != letter and str(row[L]).strip()
    ][:3]

    # Hints: still use Model B's TF-IDF ranker over THIS question
    hint_pairs = extract_hints_tfidf(passage, question, a['vec'], top_k=3)
    hints = [s for s, _score in hint_pairs]
    while len(hints) < 3:
        hints.append("(no further hint available)")

    return {
        'passage': passage,
        'question': question,
        'correct_answer': correct,
        'options': _shuffle_options(correct, gold_distractors, rng),
        'distractors': gold_distractors,
        'hints': hints,
        'latency_ms': {'total': 0, 'pick_answer': 0, 'q_gen': 0,
                       'distractors': 0, 'hints': 0},
        'source': 'RACE-gold',
    }


def verify_answer(picked: str, correct: str) -> bool:
    """Plain string equality -- both passed through .strip().lower()."""
    return (picked or '').strip().lower() == (correct or '').strip().lower()


if __name__ == "__main__":
    # Smoke test
    print("Smoke test: generate_quiz on a tiny passage")
    sample = (
        "The Sun is the star at the center of the Solar System. It is a "
        "nearly perfect ball of hot plasma, heated to incandescence by "
        "nuclear fusion reactions in its core, radiating energy mainly "
        "as visible light, ultraviolet light, and infrared radiation. "
        "It is by far the most important source of energy for life on Earth."
    )
    out = generate_quiz(sample, seed=42)
    print("\nQuestion:        ", out['question'])
    print("Correct answer:  ", out['correct_answer'])
    print("\nOptions:")
    for opt in out['options']:
        marker = '->' if opt['is_correct'] else ' '
        print(f"  {marker} {opt['label']}) {opt['text']}")
    print("\nHints (general -> near-explicit):")
    for i, h in enumerate(out['hints'], 1):
        print(f"  Hint {i}: {h}")
    print(f"\nLatency: {out['latency_ms']}")
