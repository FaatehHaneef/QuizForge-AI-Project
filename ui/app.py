"""
QuizForge AI — Streamlit UI
============================

Brainrot-themed multi-screen quiz app per spec §6.

Run:
    pip install streamlit
    streamlit run ui/app.py
"""

import os
import sys
import time
import random
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from inference import generate_quiz, load_random_race_sample, verify_answer


# ─── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuizForge AI · pls give us marks",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── Constants ──────────────────────────────────────────────────────────────
ASSET_ROOT = PROJECT_ROOT / 'ui' / 'assets' / 'memes'
SESSION_LOG_PATH = PROJECT_ROOT / 'data' / 'session_log.csv'
SUPPORTED_EXTS = {'.gif', '.png', '.jpg', '.jpeg', '.webp'}

BRAINROT_PHRASES = [
    "rizz", "gyatt", "sigma", "delulu", "mewing", "skibidi", "ohio",
    "fanum tax", "edging", "cooked", "no cap", "bussin", "sus",
    "yapping", "looksmaxxing", "GOATed", "based", "L+ratio",
]

# Per-screen palette (bg, accent, text)
THEME = {
    'welcome':       {'bg': '#1a0f1f', 'accent': '#ffd6a5', 'text': '#f0e8ff'},
    'input':         {'bg': '#1f1810', 'accent': '#ff9b3d', 'text': '#fff5e0'},
    'loading':       {'bg': '#0f1929', 'accent': '#ff2d92', 'text': '#e0f2ff'},
    'quiz':          {'bg': '#1a2030', 'accent': '#7eb3d9', 'text': '#f0f4f8'},
    'correct':       {'bg': '#2a1a3f', 'accent': '#ffb3e6', 'text': '#ffffff'},
    'wrong':         {'bg': '#1a2230', 'accent': '#9bb5cc', 'text': '#e8eef4'},
    'hints':         {'bg': '#1f2a1a', 'accent': '#d4a574', 'text': '#f4f0e8'},
    'dashboard':     {'bg': '#2a0e2a', 'accent': '#ff5cb3', 'text': '#ffe8f5'},
    'outro':         {'bg': '#1f1518', 'accent': '#ff5c5c', 'text': '#fff0f0'},
}


# ─── Meme picker ────────────────────────────────────────────────────────────
def list_memes(category: str):
    folder = ASSET_ROOT / category
    if not folder.exists():
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def _meme_natural_dims(path: Path):
    """Return (width, height) of the image, or None if we can't read it."""
    if path is None:
        return None
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def pick_meme(category: str):
    """Random meme from category folder. Returns Path or None."""
    options = list_memes(category)
    if not options:
        return None
    return random.choice(options)


def meme_data_uri(path: Path) -> str:
    """Inline a meme as a data URI so it can be used in custom HTML."""
    if path is None or not path.exists():
        return ""
    mime = {
        '.gif':  'image/gif', '.png':  'image/png',
        '.jpg':  'image/jpeg', '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
    }.get(path.suffix.lower(), 'image/png')
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def meme_html(category: str, *, height: int = 240, width: str = "auto",
              radius: int = 20, fit: str = "contain", caption: str = "",
              cls: str = "meme", tight: bool = False) -> str:
    """
    Render a meme as a styled HTML <div> with image + optional caption.

    Three render modes:
      fit='cover'             — banner with fixed-height container, image
                                cropped to fill (the wide hint banner).
      tight=True              — container HUGS the image's natural aspect.
                                `max-height` caps it; no empty bands above
                                or below. Use this for screens where the
                                meme should display at its real proportions
                                (welcome, input).
      tight=False (default)   — image scaled (upscaled if needed) to fill
                                a fixed-height container. Use when you want
                                the meme to occupy a guaranteed amount of
                                screen space regardless of source resolution
                                (correct, wrong, dashboard, etc.).
    """
    p = pick_meme(category)
    if p is None:
        # Placeholder if folder empty
        return f"""
        <div class="{cls} meme-placeholder"
             style="height:{height}px;width:{width};
                    border-radius:{radius}px;
                    display:flex;align-items:center;justify-content:center;
                    background:rgba(255,255,255,0.05);
                    border:2px dashed rgba(255,255,255,0.2);
                    color:rgba(255,255,255,0.5);font-style:italic;">
          drop a meme into ui/assets/memes/{category}/
        </div>
        """
    uri = meme_data_uri(p)
    cap_html = (
        f'<div class="meme-caption" style="margin-top:12px;font-style:italic;'
        f'opacity:0.85;text-align:center;">{caption}</div>'
        if caption else ""
    )

    if fit == 'cover':
        return f"""
        <div class="{cls}" style="text-align:center;">
          <div style="height:{height}px;width:{width};border-radius:{radius}px;
                      overflow:hidden;background:rgba(0,0,0,0.25);
                      box-shadow:0 12px 40px rgba(0,0,0,0.4);
                      display:flex;align-items:center;justify-content:center;
                      padding:8px;margin:0 auto;">
            <img src="{uri}" style="width:100%;height:100%;
                                    object-fit:cover;
                                    border-radius:{max(0,radius-8)}px;" />
          </div>
          {cap_html}
        </div>
        """

    if tight:
        # Use the image's natural aspect ratio to size a wrapper that hugs
        # the upscaled image. If the source is small (e.g. 217×150), the
        # wrapper still inflates so the meme occupies the requested visual
        # space; aspect_ratio CSS keeps the frame proportional and lets it
        # shrink gracefully on narrow screens.
        dims = _meme_natural_dims(p)
        if dims and dims[1] > 0:
            nat_w, nat_h = dims
            aspect = nat_w / nat_h
            target_w = int(height * aspect)
            return f"""
            <div class="{cls}" style="text-align:center;">
              <div style="display:inline-block;
                          width:100%;
                          max-width:{target_w}px;
                          aspect-ratio:{aspect};
                          border-radius:{radius}px;
                          background:rgba(0,0,0,0.25);
                          box-shadow:0 12px 40px rgba(0,0,0,0.4);
                          padding:8px;
                          box-sizing:border-box;">
                <img src="{uri}" style="display:block;
                                        width:100%;
                                        height:100%;
                                        object-fit:contain;
                                        border-radius:{max(0,radius-8)}px;" />
              </div>
              {cap_html}
            </div>
            """
        # PIL unavailable / read failed — fall back to fixed-fill
        return f"""
        <div class="{cls}" style="text-align:center;">
          <div style="height:{height}px;width:{width};border-radius:{radius}px;
                      overflow:hidden;background:rgba(0,0,0,0.25);
                      box-shadow:0 12px 40px rgba(0,0,0,0.4);
                      display:flex;align-items:center;justify-content:center;
                      padding:8px;margin:0 auto;">
            <img src="{uri}" style="width:100%;height:100%;
                                    object-fit:{fit};
                                    border-radius:{max(0,radius-8)}px;" />
          </div>
          {cap_html}
        </div>
        """

    # Default: image upscales (if needed) to fill the fixed-height container,
    # whole image visible (object-fit: contain). Slight pixelation on upscale
    # is fine — fits the brainrot vibe.
    return f"""
    <div class="{cls}" style="text-align:center;">
      <div style="height:{height}px;width:{width};border-radius:{radius}px;
                  overflow:hidden;background:rgba(0,0,0,0.25);
                  box-shadow:0 12px 40px rgba(0,0,0,0.4);
                  display:flex;align-items:center;justify-content:center;
                  padding:8px;margin:0 auto;">
        <img src="{uri}" style="width:100%;height:100%;
                                object-fit:{fit};
                                border-radius:{max(0,radius-8)}px;
                                image-rendering:auto;" />
      </div>
      {cap_html}
    </div>
    """


# ─── CSS injection ──────────────────────────────────────────────────────────
def inject_css(screen: str):
    palette = THEME.get(screen, THEME['welcome'])
    bg, accent, text = palette['bg'], palette['accent'], palette['text']
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Lilita+One&family=Inter:wght@400;600;800&display=swap');

      :root {{
        --bg:     {bg};
        --accent: {accent};
        --text:   {text};
      }}

      .stApp {{
        background:
          radial-gradient(circle at 20% 10%, color-mix(in srgb, var(--accent) 18%, transparent) 0%, transparent 40%),
          radial-gradient(circle at 80% 90%, color-mix(in srgb, var(--accent) 12%, transparent) 0%, transparent 50%),
          var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
        animation: fade-in 0.4s ease;
      }}
      @keyframes fade-in {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}

      h1, h2, h3, .qf-title {{
        font-family: 'Bangers', 'Lilita One', sans-serif;
        letter-spacing: 0.02em;
        color: var(--accent) !important;
        text-shadow: 0 0 24px color-mix(in srgb, var(--accent) 40%, transparent);
      }}
      h1, .qf-title-xl {{ font-size: 4rem !important; line-height: 1.1; }}
      h2 {{ font-size: 2.4rem !important; }}
      h3 {{ font-size: 1.6rem !important; }}

      .qf-subtitle {{
        font-family: 'Lilita One', sans-serif;
        font-size: 1.2rem;
        opacity: 0.85;
        letter-spacing: 0.5px;
      }}

      /* Marquee */
      .qf-marquee {{
        overflow: hidden; white-space: nowrap;
        background: rgba(0,0,0,0.3);
        border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
        border-radius: 12px;
        padding: 8px 0; margin-bottom: 24px;
        font-family: 'Bangers', sans-serif;
        font-size: 1.2rem;
        color: var(--accent);
      }}
      .qf-marquee-track {{
        display: inline-block;
        animation: scroll 30s linear infinite;
        padding-left: 100%;
      }}
      @keyframes scroll {{
        from {{ transform: translateX(0); }}
        to   {{ transform: translateX(-100%); }}
      }}

      /* Streamlit's button — make it brainrot. Fixed min-height so wrapped
         labels don't break button-row alignment across the screen. */
      .stButton > button {{
        background: linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 60%, white)) !important;
        color: #1a0f1f !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 24px !important;
        min-height: 64px !important;
        height: 64px !important;
        font-family: 'Bangers', sans-serif !important;
        font-size: 1.3rem !important;
        letter-spacing: 0.04em !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: 0 8px 24px color-mix(in srgb, var(--accent) 40%, transparent) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
      }}
      .stButton > button:hover {{
        transform: scale(1.04) translateY(-2px);
        box-shadow: 0 12px 36px color-mix(in srgb, var(--accent) 60%, transparent) !important;
        filter: brightness(1.1);
      }}
      .stButton > button:active {{
        transform: scale(0.98);
      }}

      /* Text area + inputs */
      .stTextArea textarea, .stTextInput input {{
        background: rgba(0,0,0,0.3) !important;
        color: var(--text) !important;
        border: 2px solid color-mix(in srgb, var(--accent) 30%, transparent) !important;
        border-radius: 16px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 16px !important;
        transition: all 0.25s ease;
      }}
      .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 25%, transparent) !important;
      }}

      /* Radio (quiz options) */
      .stRadio > div {{ gap: 12px; }}
      .stRadio label {{
        background: rgba(0,0,0,0.25) !important;
        border: 2px solid color-mix(in srgb, var(--accent) 25%, transparent) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        font-family: 'Lilita One', sans-serif !important;
        font-size: 1.1rem !important;
        transition: all 0.2s ease;
        cursor: pointer;
      }}
      .stRadio label:hover {{
        border-color: var(--accent) !important;
        transform: translateX(4px);
        background: rgba(0,0,0,0.4) !important;
      }}

      /* Cards */
      .qf-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        margin-bottom: 16px;
      }}

      /* Metric tiles */
      [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.04);
        border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
        border-radius: 16px;
        padding: 16px;
      }}
      [data-testid="stMetricLabel"] {{
        font-family: 'Lilita One', sans-serif;
        color: var(--accent) !important;
      }}
      [data-testid="stMetricValue"] {{
        font-family: 'Bangers', sans-serif;
        color: var(--text) !important;
        font-size: 2.2rem !important;
      }}

      /* Expanders for hints */
      .streamlit-expanderHeader {{
        background: rgba(0,0,0,0.3) !important;
        border-radius: 14px !important;
        font-family: 'Lilita One', sans-serif !important;
        color: var(--accent) !important;
      }}

      /* Hide Streamlit chrome we don't want */
      #MainMenu, footer, header {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1100px; }}

      /* Loading overlay — fixed full-viewport so nothing from a prior screen
         peeks through during Streamlit's rerun transition. */
      .qf-loading-overlay {{
        position: fixed;
        inset: 0;
        z-index: 9999;
        background:
          radial-gradient(circle at 20% 10%, color-mix(in srgb, var(--accent) 18%, transparent) 0%, transparent 40%),
          radial-gradient(circle at 80% 90%, color-mix(in srgb, var(--accent) 12%, transparent) 0%, transparent 50%),
          var(--bg);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 28px;
        padding: 48px;
      }}
      .qf-loading-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        flex-wrap: nowrap;
      }}
      .qf-loading-title {{
        font-family: 'Bangers', sans-serif;
        font-size: 3rem;
        color: var(--accent);
        letter-spacing: 0.04em;
        text-shadow: 0 0 24px color-mix(in srgb, var(--accent) 40%, transparent);
        white-space: nowrap;
      }}
      .qf-mini-spinner {{
        display: inline-block;
        width: 32px;
        height: 32px;
        border: 4px solid color-mix(in srgb, var(--accent) 40%, transparent);
        border-top-color: var(--accent);
        border-radius: 50%;
        animation: qf-spin 0.8s linear infinite;
        flex-shrink: 0;
      }}
      @keyframes qf-spin {{ to {{ transform: rotate(360deg); }} }}
      .qf-loading-meme {{
        width: min(420px, 80vw);
        max-height: 60vh;
        border-radius: 28px;
        overflow: hidden;
        background: rgba(0,0,0,0.25);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        padding: 8px;
      }}
      .qf-loading-meme img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 20px;
      }}
    </style>
    """, unsafe_allow_html=True)


def marquee():
    track = " · ".join(BRAINROT_PHRASES * 4)
    st.markdown(
        f'<div class="qf-marquee"><span class="qf-marquee-track">{track}</span></div>',
        unsafe_allow_html=True,
    )


# ─── Session state ──────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'screen':       'welcome',
        'passage':      '',
        'quiz':         None,
        'picked':       None,
        'is_correct':   None,
        'hints_seen':   0,
        'session_log':  [],
        'started_at':   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go(screen: str):
    st.session_state.screen = screen
    st.rerun()


def log_session_event(quiz: dict, picked_text: str, correct: bool, latency_ms: int):
    row = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'question': quiz.get('question', '')[:200],
        'correct_answer': quiz.get('correct_answer', '')[:200],
        'picked': (picked_text or '')[:200],
        'correct': bool(correct),
        'latency_ms': int(latency_ms),
        'hints_used': st.session_state.get('hints_seen', 0),
    }
    st.session_state.session_log.append(row)
    # Persist to disk too
    SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if SESSION_LOG_PATH.exists():
        df.to_csv(SESSION_LOG_PATH, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df.to_csv(SESSION_LOG_PATH, index=False, encoding='utf-8')


# ─── Screens ────────────────────────────────────────────────────────────────
def screen_welcome():
    inject_css('welcome')
    # Welcome-only: vertically centre the whole page in the viewport. The
    # block-container's parents in Streamlit's layout don't always inherit
    # 100vh, so we also push the .stApp ancestor and add a small top spacer
    # for browsers where the flex centring still measures short.
    st.markdown("""
    <style>
      .stApp > div:first-child > div:first-child > div:first-child {
        min-height: 100vh;
      }
      .block-container {
        min-height: calc(100vh - 3rem) !important;
        display: flex !important;
        flex-direction: column;
        justify-content: center;
      }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:6vh;'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 4], gap="large")
    with left:
        st.markdown(meme_html('welcome', height=720, radius=32,
                              caption="why r u even here 🤨",
                              tight=True),
                    unsafe_allow_html=True)
    with right:
        st.markdown('<div class="qf-title qf-title-xl">QUIZFORGE AI</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="qf-subtitle">'
            'Welcome to QuizForger… but eh… why are you here? '
            'chalo niklo? aur koi kaam nahi hai? 🙄<br><br>'
            'paste a passage. get a quiz. cry. repeat.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        if st.button("Dont press this", key="welcome_enter",
                     use_container_width=True):
            st.session_state.started_at = time.time()
            go('input')
        if st.button("Dashy", key="welcome_dash",
                     use_container_width=True):
            go('dashboard')


def screen_input():
    inject_css('input')
    st.markdown('<div class="qf-title">PASTE A PASSAGE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="qf-subtitle">'
                'or don\'t. nothing matters anymore. but the AI does need '
                'something to chew on, so… try to paste something.'
                '</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    main_col, side_col = st.columns([2, 3], gap="large")
    with main_col:
        passage = st.text_area(
            "Reading passage",
            value=st.session_state.passage,
            height=420,
            placeholder="paste a comprehension passage here. minimum like 200 chars or you're cooked.",
            label_visibility='collapsed',
        )
        st.session_state.passage = passage

        # Row 1: ← Back | SUBMIT →
        # Row 2: 🎲 Generate Random (full width across both)
        r1_a, r1_b = st.columns(2, gap="small")
        with r1_a:
            if st.button("← Back", key="input_back",
                         use_container_width=True):
                go('welcome')
        with r1_b:
            submit = st.button("SUBMIT →", key="input_submit",
                               use_container_width=True,
                               disabled=len(passage.strip()) < 50)

        if st.button("🎲 Generate Random", key="input_random",
                     use_container_width=True):
            with st.spinner("loading a random sample…"):
                sample = load_random_race_sample('val')
                st.session_state.passage = sample['passage']
                st.session_state.quiz = sample
                st.session_state.hints_seen = 0
                st.session_state.picked = None
                st.session_state.is_correct = None
            go('quiz')

        if submit:
            go('loading')

    with side_col:
        st.markdown(meme_html('article_input', height=620, radius=28,
                              caption="Ye kesa ganda passage hai, "
                                      "la hawla wala Quwatta 😤",
                              tight=True),
                    unsafe_allow_html=True)


def screen_loading():
    """
    Full-viewport overlay: title + spinner on one row at the top, meme below.
    `position: fixed` covers any stale DOM from the input screen so nothing
    leaks through during Streamlit's rerun transition.
    """
    inject_css('loading')

    meme_p = pick_meme('loading')
    meme_uri = meme_data_uri(meme_p) if meme_p else ""
    meme_block = (
        f'<div class="qf-loading-meme"><img src="{meme_uri}" /></div>'
        if meme_uri
        else '<div class="qf-loading-meme" style="display:flex;'
             'align-items:center;justify-content:center;color:rgba(255,255,255,0.5);'
             'font-style:italic;">drop a meme into ui/assets/memes/loading/</div>'
    )

    st.markdown(
        f'''
        <div class="qf-loading-overlay">
          <div class="qf-loading-header">
            <div class="qf-loading-title">KHANJAR MAAR DO MUJHEY AP</div>
            <div class="qf-mini-spinner"></div>
          </div>
          {meme_block}
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Inference runs synchronously while the overlay is shown. Once it
    # completes we route to the quiz screen.
    quiz = generate_quiz(st.session_state.passage,
                         seed=int(time.time()) % (1 << 30))
    st.session_state.quiz = quiz
    st.session_state.hints_seen = 0
    st.session_state.picked = None
    st.session_state.is_correct = None
    go('quiz')


def screen_quiz():
    inject_css('quiz')
    quiz = st.session_state.quiz
    if quiz is None:
        st.warning("no quiz loaded — going back")
        go('input')
        return

    st.markdown('<div class="qf-title">THE QUIZ</div>', unsafe_allow_html=True)
    main_col, side_col = st.columns([3, 2], gap="large")
    with main_col:
        st.markdown(f'<div class="qf-card">'
                    f'<div style="font-family:\'Lilita One\',sans-serif;'
                    f'font-size:1.3rem;line-height:1.5;">'
                    f'{quiz["question"]}'
                    f'</div></div>', unsafe_allow_html=True)

        labels = [f"{o['label']}) {o['text']}" for o in quiz['options']]
        picked_label = st.radio("Choose an answer:", labels,
                                key="quiz_radio",
                                label_visibility='collapsed',
                                index=None)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("💡 Need a hint", key="quiz_hint",
                         use_container_width=True):
                go('hints')
        with c2:
            check = st.button("CHECK ✓", key="quiz_check",
                              use_container_width=True,
                              disabled=picked_label is None)
        with c3:
            if st.button("🏠 home", key="quiz_home",
                         use_container_width=True):
                go('welcome')

        if check and picked_label:
            picked_text = picked_label.split(') ', 1)[1] if ') ' in picked_label else picked_label
            correct = verify_answer(picked_text, quiz['correct_answer'])
            st.session_state.picked = picked_text
            st.session_state.is_correct = correct
            log_session_event(
                quiz, picked_text, correct,
                latency_ms=quiz.get('latency_ms', {}).get('total', 0),
            )
            go('correct' if correct else 'wrong')

    with side_col:
        st.markdown(meme_html('quiz_view', height=460, radius=24,
                              caption="the AI took 8hrs to write this 😭"),
                    unsafe_allow_html=True)


def screen_correct():
    inject_css('correct')
    quiz = st.session_state.quiz
    st.balloons()
    st.markdown('<div class="qf-title qf-title-xl" style="text-align:center;">'
                'I KNOW DAS RIGHT 🎉</div>', unsafe_allow_html=True)
    st.markdown(meme_html('correct', height=560, radius=28),
                unsafe_allow_html=True)
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🔁 New quiz", key="correct_new",
                     use_container_width=True):
            go('input')
    with c2:
        if st.button("📊 Dashboard", key="correct_dash",
                     use_container_width=True):
            go('dashboard')
    with c3:
        if st.button("🙏 Submit project", key="correct_outro",
                     use_container_width=True):
            go('outro')


def screen_wrong():
    inject_css('wrong')
    quiz = st.session_state.quiz
    st.markdown('<div class="qf-title qf-title-xl" style="text-align:center;">'
                'FITTEY MUU 😤</div>', unsafe_allow_html=True)
    st.markdown(meme_html('wrong', height=540, radius=28),
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="qf-card" style="text-align:center;">'
        f'<div style="opacity:0.7;font-family:\'Lilita One\',sans-serif;">'
        f'the actual answer was:</div>'
        f'<div style="font-family:\'Bangers\',sans-serif;font-size:1.8rem;'
        f'color:var(--accent);margin-top:8px;">'
        f'{quiz["correct_answer"]}</div>'
        f'</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("💡 see the hints", key="wrong_hints",
                     use_container_width=True):
            go('hints')
    with c2:
        if st.button("🔁 try another", key="wrong_new",
                     use_container_width=True):
            go('input')
    with c3:
        if st.button("📊 dashboard", key="wrong_dash",
                     use_container_width=True):
            go('dashboard')


def screen_hints():
    inject_css('hints')
    quiz = st.session_state.quiz
    if quiz is None:
        go('input'); return

    st.markdown('<div class="qf-title qf-title-xl" style="text-align:center;">'
                'TAKE THE DAMN HINTS LOSER</div>', unsafe_allow_html=True)
    st.markdown(meme_html('hints', height=380, radius=24, fit='cover'),
                unsafe_allow_html=True)
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    hints = quiz.get('hints', [])
    seen = st.session_state.hints_seen
    captions = [
        "Hint 1 · the gentlest nudge",
        "Hint 2 · getting warmer fr fr",
        "Hint 3 · BRO I'M LITERALLY TELLING YOU",
    ]
    for i, h in enumerate(hints[:3]):
        if i <= seen:
            with st.expander(captions[i], expanded=(i == seen)):
                st.markdown(f'<div style="font-family:\'Lilita One\',sans-serif;'
                            f'font-size:1.15rem;line-height:1.6;">{h}</div>',
                            unsafe_allow_html=True)
        else:
            with st.expander(f"Hint {i+1} · 🔒 locked"):
                st.write("unlock the previous hint first 😤")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if seen < 2 and st.button("🔓 unlock next hint",
                                  key=f"hint_unlock_{seen}",
                                  use_container_width=True):
            st.session_state.hints_seen += 1
            st.rerun()
    with c2:
        if st.button("← back to quiz", key="hints_back",
                     use_container_width=True):
            go('quiz')
    with c3:
        if seen >= 2:
            if st.button("👁 reveal answer", key="hints_reveal",
                         use_container_width=True):
                st.success(f"the answer is: **{quiz['correct_answer']}**")


def screen_dashboard():
    inject_css('dashboard')

    # Pull the session log (in-memory + persisted on disk)
    log = list(st.session_state.session_log)
    if SESSION_LOG_PATH.exists():
        try:
            disk = pd.read_csv(SESSION_LOG_PATH)
            if not disk.empty:
                log = disk.to_dict('records')
        except Exception:
            pass
    df = pd.DataFrame(log)
    n = len(df)

    # Header: pfp on the left, title + subtitle + 2x2 metrics tiles filling
    # the formerly-empty right side.
    head_l, head_r = st.columns([2, 3], gap="large")
    with head_l:
        st.markdown(meme_html('dashboard', height=480, radius=28),
                    unsafe_allow_html=True)
    with head_r:
        st.markdown('<div class="qf-title qf-title-xl">DEV DASHBOARD</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if n == 0:
            st.markdown(
                '<div class="qf-card" style="text-align:center;">'
                'no sessions yet · go answer a quiz, coward'
                '</div>', unsafe_allow_html=True,
            )
        else:
            # 2x2 grid of metric tiles, filling the right column under the title
            row1_a, row1_b = st.columns(2)
            row2_a, row2_b = st.columns(2)
            row1_a.metric("Sessions", f"{n}")
            if 'correct' in df.columns:
                row1_b.metric("Correct rate",
                              f"{df['correct'].astype(bool).mean() * 100:.1f}%")
            if 'latency_ms' in df.columns:
                row2_a.metric("Avg latency",
                              f"{df['latency_ms'].astype(float).mean() / 1000:.2f}s")
            if 'hints_used' in df.columns:
                row2_b.metric("Avg hints used",
                              f"{df['hints_used'].astype(float).mean():.1f}")

    # Recent sessions table + CSV export below the header row
    if n > 0:
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="qf-card">'
                    '<h3>Recent sessions</h3></div>', unsafe_allow_html=True)
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button(
            "⬇ Export session log to CSV",
            df.to_csv(index=False).encode('utf-8'),
            file_name='quizforge_session_log.csv',
            mime='text/csv',
            use_container_width=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← home", key="dash_home", use_container_width=True):
            go('welcome')
    with c2:
        if st.button("🙏 submit project", key="dash_outro",
                     use_container_width=True):
            go('outro')


def screen_outro():
    inject_css('outro')
    st.markdown('<div class="qf-title qf-title-xl" style="text-align:center;">'
                'THANK U FOR USING QUIZFORGE AI</div>',
                unsafe_allow_html=True)
    st.markdown(meme_html('outro', height=520, radius=28,
                          caption="🙏 marks dedein please 🙏"),
                unsafe_allow_html=True)
    st.markdown(
        '<div class="qf-card" style="text-align:center;">'
        '<div style="font-family:\'Lilita One\',sans-serif;font-size:1.2rem;">'
        'Made with disappointment,<br>'
        'by Yours unsincere,<br>'
        'Faateh and Ibbo'
        '</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 new session", key="outro_new",
                     use_container_width=True):
            for k in ['quiz', 'picked', 'is_correct', 'hints_seen', 'passage']:
                st.session_state[k] = None if k != 'hints_seen' else 0
            st.session_state.passage = ''
            go('welcome')
    with c2:
        if st.button("📊 dashboard", key="outro_dash",
                     use_container_width=True):
            go('dashboard')

    # Full-width "Demo khatam" — kills the Streamlit process and attempts
    # to close the browser tab. Tab-close only works when JS opened the tab;
    # otherwise the browser blocks it and the user sees "site can't be reached".
    if st.button("🛑 Demo khatam", key="outro_kill",
                 use_container_width=True):
        st.markdown(
            '<div class="qf-card" style="text-align:center;">'
            '<div style="font-family:\'Bangers\',sans-serif;font-size:1.6rem;'
            'color:var(--accent);">DEMO KHATAM 🪦<br>'
            '<span style="font-family:\'Lilita One\',sans-serif;font-size:1rem;'
            'opacity:0.8;">closing the tab + killing the server…</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        # JS attempt to close the tab. Falls through silently if blocked.
        st.markdown(
            '<script>setTimeout(function(){window.close();}, 300);</script>',
            unsafe_allow_html=True,
        )
        # Schedule process exit so Streamlit has time to flush the response
        # before the connection drops.
        def _delayed_exit():
            import time as _t
            _t.sleep(1.0)
            os._exit(0)
        import threading
        threading.Thread(target=_delayed_exit, daemon=True).start()


# ─── Router ─────────────────────────────────────────────────────────────────
SCREENS = {
    'welcome':   screen_welcome,
    'input':     screen_input,
    'loading':   screen_loading,
    'quiz':      screen_quiz,
    'correct':   screen_correct,
    'wrong':     screen_wrong,
    'hints':     screen_hints,
    'dashboard': screen_dashboard,
    'outro':     screen_outro,
}


def main():
    init_state()
    SCREENS.get(st.session_state.screen, screen_welcome)()


if __name__ == "__main__":
    main()
