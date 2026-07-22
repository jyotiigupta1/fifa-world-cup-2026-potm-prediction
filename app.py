"""
==================================================================================
FIFA WORLD CUP 2026 — PLAYER OF THE MATCH PREDICTION SYSTEM
==================================================================================
A premium, dark glassmorphism, blue-neon analytics dashboard built with Streamlit.
Powered by a trained Support Vector Classifier (SVC).


Model   : svc_model.pkl
Scaler  : scaler.pkl
Encoders: label_encoder.pkl
==================================================================================
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================================================
# PAGE CONFIGURATION
# ==================================================================================
st.set_page_config(
    page_title="FIFA World Cup 2026 | Player of the Match Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================================
# CONSTANTS — FEATURE SCHEMA (must match training notebook exactly)
# ==================================================================================
MODEL_PATH = "svc_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

TARGET_COLUMN = "player_of_match_awards"

CATEGORICAL_FEATURES = [
    "nationality", "team", "position", "preferred_foot",
    "opponent_team", "tournament_stage", "match_result",
]

# EXACT training feature order
FEATURE_ORDER = [
    "age", "nationality", "team", "position", "height_cm", "weight_kg",
    "preferred_foot", "opponent_team", "tournament_stage", "match_result",
    "minutes_played", "shots", "shots_on_target", "expected_goals_xg",
    "expected_assists_xa", "key_passes", "successful_passes", "total_passes",
    "pass_accuracy", "dribbles_attempted", "successful_dribbles", "crosses",
    "successful_crosses", "tackles", "interceptions", "clearances", "blocks",
    "aerial_duels_won", "aerial_duels_lost", "recoveries", "defensive_actions",
    "fouls_committed", "fouls_suffered", "yellow_cards", "red_cards", "offsides",
    "saves", "save_percentage", "punches", "clean_sheet", "goals_conceded",
    "penalty_saves", "distance_covered_km", "sprint_distance_km", "top_speed_kmh",
    "accelerations", "decelerations", "stamina_score", "offensive_contribution",
    "defensive_contribution", "possession_impact", "pressure_resistance",
    "creativity_score", "consistency_score", "clutch_performance_score",
]

NUMERICAL_FEATURES = [f for f in FEATURE_ORDER if f not in CATEGORICAL_FEATURES]

# Fallback option lists (used only if label_encoders.pkl classes are unavailable)
FALLBACK_OPTIONS = {
    "nationality": [
        "Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal",
        "Netherlands", "Belgium", "Croatia", "Italy", "Uruguay", "USA", "Mexico",
        "Canada", "Morocco", "Japan", "South Korea", "Senegal", "Colombia",
    ],
    "team": [
        "Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal",
        "Netherlands", "Belgium", "Croatia", "Italy", "Uruguay", "USA", "Mexico",
        "Canada", "Morocco", "Japan", "South Korea", "Senegal", "Colombia",
    ],
    "position": ["Goalkeeper", "Defender", "Midfielder", "Forward"],
    "preferred_foot": ["Left", "Right", "Both"],
    "opponent_team": [
        "Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal",
        "Netherlands", "Belgium", "Croatia", "Italy", "Uruguay", "USA", "Mexico",
        "Canada", "Morocco", "Japan", "South Korea", "Senegal", "Colombia",
    ],
    "tournament_stage": [
        "Group Stage", "Round of 32", "Round of 16", "Quarter-Final",
        "Semi-Final", "Third Place Playoff", "Final",
    ],
    "match_result": ["Win", "Draw", "Loss"],
}

# ==================================================================================
# CUSTOM CSS — PREMIUM DARK GLASSMORPHISM / BLUE NEON THEME
# ==================================================================================
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ---------- HIDE STREAMLIT DEFAULT CHROME ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
div[data-testid="stDecoration"] {visibility: hidden;}
div[data-testid="stStatusWidget"] {visibility: hidden;}
#stDecoration {display:none;}

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 20%, #0b1330 0%, #060a1a 45%, #030512 100%);
    background-attachment: fixed;
    color: #e8ecff;
    overflow-x: hidden;
}

/* ---------- ANIMATED FOOTBALL BACKGROUND ---------- */
.football-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.football-bg span {
    position: absolute;
    font-size: 2.4rem;
    opacity: 0.08;
    animation: floatBall linear infinite;
    filter: drop-shadow(0 0 12px #00d4ff);
}
@keyframes floatBall {
    0%   { transform: translateY(110vh) rotate(0deg); opacity: 0; }
    10%  { opacity: 0.10; }
    90%  { opacity: 0.10; }
    100% { transform: translateY(-15vh) rotate(360deg); opacity: 0; }
}
.football-bg span:nth-child(1){ left:5%;  animation-duration:22s; animation-delay:0s;}
.football-bg span:nth-child(2){ left:18%; animation-duration:28s; animation-delay:3s;}
.football-bg span:nth-child(3){ left:32%; animation-duration:19s; animation-delay:6s;}
.football-bg span:nth-child(4){ left:48%; animation-duration:25s; animation-delay:1s;}
.football-bg span:nth-child(5){ left:63%; animation-duration:21s; animation-delay:4s;}
.football-bg span:nth-child(6){ left:77%; animation-duration:30s; animation-delay:2s;}
.football-bg span:nth-child(7){ left:88%; animation-duration:24s; animation-delay:5s;}
.football-bg span:nth-child(8){ left:95%; animation-duration:26s; animation-delay:7s;}

/* ---------- GLASS CARD BASE ---------- */
.glass-card {
    position: relative;
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: all 0.35s ease;
    z-index: 1;
    margin-bottom: 1.2rem;
}
.glass-card:hover {
    transform: translateY(-6px);
    border-color: rgba(0, 212, 255, 0.6);
    box-shadow: 0 16px 44px rgba(0, 212, 255, 0.25), inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ---------- HERO SECTION ---------- */
.hero-wrap {
    text-align: center;
    padding: 3.2rem 1rem 2.4rem 1rem;
    position: relative;
    z-index: 1;
}
.hero-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #4fd8ff;
    font-size: 0.8rem;
    letter-spacing: 2px;
    font-weight: 600;
    margin-bottom: 1.4rem;
    text-transform: uppercase;
    animation: pulseBadge 2.6s ease-in-out infinite;
}
@keyframes pulseBadge {
    0%, 100% { box-shadow: 0 0 0px rgba(0,212,255,0.4); }
    50% { box-shadow: 0 0 18px rgba(0,212,255,0.5); }
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    font-size: 3.4rem;
    line-height: 1.1;
    background: linear-gradient(90deg, #ffffff 0%, #4fd8ff 45%, #0084ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 40px rgba(0, 132, 255, 0.25);
    margin-bottom: 0.6rem;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #9fb4d8;
    font-weight: 300;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-divider {
    width: 120px;
    height: 3px;
    margin: 1.6rem auto;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    border-radius: 4px;
}

/* ---------- SECTION HEADERS ---------- */
.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 1.6rem 0 1rem 0;
    padding-left: 0.8rem;
    border-left: 4px solid #00d4ff;
    text-shadow: 0 0 20px rgba(0,212,255,0.3);
}
.section-sub {
    color: #8fa3c7;
    font-size: 0.92rem;
    margin-bottom: 1.2rem;
    padding-left: 0.9rem;
}

/* ---------- METRIC / STAT CARDS ---------- */
.metric-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 1.3rem;
    text-align: center;
    backdrop-filter: blur(14px);
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease both;
}
.metric-card:hover {
    border-color: rgba(0,212,255,0.55);
    transform: scale(1.04);
    box-shadow: 0 10px 30px rgba(0,212,255,0.22);
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #4fd8ff;
    text-shadow: 0 0 18px rgba(0,212,255,0.4);
}
.metric-label {
    color: #9fb4d8;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- PREDICTION RESULT CARDS ---------- */
.result-success {
    background: linear-gradient(135deg, rgba(0,255,163,0.12), rgba(0,212,255,0.08));
    border: 1px solid rgba(0,255,163,0.5);
    border-radius: 22px;
    padding: 2.2rem;
    text-align: center;
    animation: glowSuccess 2s ease-in-out infinite, popIn 0.5s ease;
    box-shadow: 0 0 50px rgba(0,255,163,0.25);
}
@keyframes glowSuccess {
    0%, 100% { box-shadow: 0 0 30px rgba(0,255,163,0.25); }
    50% { box-shadow: 0 0 60px rgba(0,255,163,0.45); }
}
.result-fail {
    background: linear-gradient(135deg, rgba(255,80,80,0.08), rgba(0,20,50,0.3));
    border: 1px solid rgba(255,120,120,0.35);
    border-radius: 22px;
    padding: 2.2rem;
    text-align: center;
    animation: popIn 0.5s ease;
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.9); }
    100% { opacity: 1; transform: scale(1); }
}
.result-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-icon {
    font-size: 3.6rem;
    margin-bottom: 0.6rem;
    display: block;
}
.confidence-bar-wrap {
    width: 100%;
    height: 14px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    overflow: hidden;
    margin-top: 1rem;
    border: 1px solid rgba(0,212,255,0.25);
}
.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #0084ff, #00e5ff, #00ffa3);
    border-radius: 10px;
    transition: width 1.2s ease;
    box-shadow: 0 0 14px rgba(0,212,255,0.6);
}

/* ---------- BUTTONS ---------- */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0084ff, #00c6ff);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    padding: 0.85rem 1.5rem;
    border-radius: 50px;
    border: none;
    box-shadow: 0 0 25px rgba(0, 132, 255, 0.55);
    transition: all 0.3s ease;
    animation: glowPulse 2.4s ease-in-out infinite;
}
div.stButton > button:hover {
    transform: translateY(-3px) scale(1.015);
    box-shadow: 0 0 45px rgba(0, 212, 255, 0.85);
    color: #ffffff;
}
div.stButton > button:active {
    transform: translateY(0px) scale(0.99);
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(0,132,255,0.45); }
    50% { box-shadow: 0 0 38px rgba(0,212,255,0.8); }
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060a1a 0%, #0a1130 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}
section[data-testid="stSidebar"] .css-1d391kg, section[data-testid="stSidebar"] * {
    color: #dbe6ff;
}
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.4rem 0;
}
.sidebar-logo .icon {
    font-size: 2.6rem;
    filter: drop-shadow(0 0 14px #00d4ff);
}
.sidebar-logo .title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #ffffff;
    margin-top: 0.3rem;
    line-height: 1.3;
}
.sidebar-logo .sub {
    color: #6f89bd;
    font-size: 0.75rem;
    letter-spacing: 1px;
}

/* ---------- TABLE / DATAFRAME ---------- */
.stDataFrame, .stTable {
    border-radius: 14px;
    overflow: hidden;
}

/* ---------- INPUT WIDGETS ---------- */
.stSelectbox, .stNumberInput, .stSlider, .stTextInput {
    background: transparent;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}
input, textarea {
    color: #ffffff !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #0084ff, #00e5ff) !important;
}

/* ---------- BADGES / CHIPS ---------- */
.chip {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.35);
    color: #4fd8ff;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.3rem 0.2rem 0;
}

/* ---------- FOOTER ---------- */
.app-footer {
    text-align: center;
    padding: 2.4rem 1rem 1.6rem 1rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(0,212,255,0.15);
    color: #6f89bd;
    font-size: 0.85rem;
}
.app-footer .brand {
    font-family: 'Orbitron', sans-serif;
    color: #4fd8ff;
    font-weight: 700;
    font-size: 1rem;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #060a1a; }
::-webkit-scrollbar-thumb { background: #0084ff; border-radius: 10px; }

/* ---------- DIVIDER ---------- */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.5), transparent);
    margin: 1.6rem 0;
    border: none;
}
</style>

<div class="football-bg">
<span>⚽</span><span>⚽</span><span>⚽</span><span>⚽</span>
<span>⚽</span><span>⚽</span><span>⚽</span><span>⚽</span>
</div>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================
# MODEL / SCALER / ENCODER LOADING
# ==========================================================
import joblib
import pickle

@st.cache_resource(show_spinner=False)
def load_artifact(path):
    """Load a model/scaler/encoder artifact safely, returning (object, error_message)."""
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        obj = joblib.load(path)
        return obj, None
    except Exception as e:
        return None, f"Failed to load {path}: {e}"


@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    model, model_err = load_artifact(MODEL_PATH)
    scaler, scaler_err = load_artifact(SCALER_PATH)
    encoders, enc_err = load_artifact(ENCODER_PATH)
    errors = [e for e in (model_err, scaler_err, enc_err) if e]
    return model, scaler, encoders, errors


MODEL, SCALER, LABEL_ENCODERS, LOAD_ERRORS = load_all_artifacts()


def get_categorical_options(feature_name):
    """Get dropdown options from the saved label encoder, falling back to defaults."""
    if isinstance(LABEL_ENCODERS, dict) and feature_name in LABEL_ENCODERS:
        try:
            return list(LABEL_ENCODERS[feature_name].classes_)
        except Exception:
            pass
    return FALLBACK_OPTIONS.get(feature_name, ["Unknown"])


def safe_label_encode(feature_name, value):
    """
    Encode a categorical value using the saved LabelEncoder.
    Handles unseen categories gracefully by falling back to the first known class.
    Returns (encoded_value, was_unseen: bool)
    """
    if isinstance(LABEL_ENCODERS, dict) and feature_name in LABEL_ENCODERS:
        encoder = LABEL_ENCODERS[feature_name]
        classes = list(encoder.classes_)
        if value in classes:
            return int(encoder.transform([value])[0]), False
        else:
            # graceful fallback for unseen categorical values
            fallback_value = classes[0]
            return int(encoder.transform([fallback_value])[0]), True
    else:
        # No encoder available — fallback to a simple stable hash-based index
        options = FALLBACK_OPTIONS.get(feature_name, [value])
        idx = options.index(value) if value in options else 0
        return idx, value not in options


# ==================================================================================
# SIDEBAR — NAVIGATION
# ==================================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="icon">🏆</div>
            <div class="title">FIFA WORLD CUP 2026</div>
            <div class="sub">POTM PREDICTION ENGINE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔮 Predict Player of the Match",
            "📊 Dataset Overview",
            "🧠 About the Model",
            "📘 Prediction Guide",
            "🧬 Feature Information"
            
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    st.markdown("##### ⚙️ System Status")
    if MODEL is not None:
        st.markdown('<span class="chip">✅ Model Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip">❌ Model Missing</span>', unsafe_allow_html=True)

    if SCALER is not None:
        st.markdown('<span class="chip">✅ Scaler Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip">❌ Scaler Missing</span>', unsafe_allow_html=True)

    if LABEL_ENCODERS is not None:
        st.markdown('<span class="chip">✅ Encoders Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip">❌ Encoders Missing</span>', unsafe_allow_html=True)

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    st.caption("⚽ Built for the FIFA World Cup 2026 Analytics Suite")


# ==================================================================================
# REUSABLE UI COMPONENTS
# ==================================================================================
def hero_section(badge, title, subtitle):
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-badge">{badge}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-divider"></div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=None):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(col, label, value, icon="📈"):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:1.6rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================================================
# PAGE: HOME
# ==================================================================================
if page == "🏠 Home":
    hero_section(
        "⚽ FIFA WORLD CUP 2026 · AI ANALYTICS",
        "Player of the Match<br>Prediction System",
        "A premium machine-learning powered dashboard that predicts whether a footballer "
        "will be crowned <b>Player of the Match</b>, using real match performance metrics "
        "and a trained Support Vector Classifier (SVC).",
    )

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Input Features", len(FEATURE_ORDER), "🧬")
    metric_card(c2, "Categorical Fields", len(CATEGORICAL_FEATURES), "🏷️")
    metric_card(c3, "Numerical Fields", len(NUMERICAL_FEATURES), "🔢")
    metric_card(c4, "Model Type", "SVC", "🤖")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>⚡ Real-Time Prediction</h3>
                <p style="color:#9fb4d8;">Enter live match statistics and instantly get an
                AI-driven prediction of Player of the Match probability.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🎯 High Precision Model</h3>
                <p style="color:#9fb4d8;">Powered by a tuned Support Vector Classifier trained
                on granular attacking, defensive, and physical performance data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="glass-card">
                <h3>📊 Deep Analytics</h3>
                <p style="color:#9fb4d8;">Explore 50+ performance metrics — from xG and pressing
                actions to sprint speed and clutch performance scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🚀 Get Started", "Head to the Predict tab in the sidebar to run your first prediction.")
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <p style="font-size:1.05rem;color:#c8d7f5;">
            Select <b>🔮 Predict Player of the Match</b> from the sidebar navigation to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# PAGE: PREDICT
# ==================================================================================
elif page == "🔮 Predict Player of the Match":
    hero_section(
        "LIVE PREDICTION ENGINE",
        "Predict Player of the Match",
        "Fill in the player's biometric, match, and performance statistics below, "
        "then run the AI prediction engine.",
    )

    if LOAD_ERRORS:
        st.markdown(
            f"""
            <div class="glass-card" style="border-color:rgba(255,120,120,0.5);">
                <h4 style="color:#ff8a8a;">⚠️ Artifact Loading Warning</h4>
                <p style="color:#e0b5b5;">{'<br>'.join(LOAD_ERRORS)}</p>
                <p style="color:#9fb4d8;">Ensure <code>svc_model.pkl</code>, <code>scaler.pkl</code>,
                and <code>label_encoders.pkl</code> are placed in the same folder as this app.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    user_inputs = {}

    # ---------------- PLAYER PROFILE ----------------
    section_header("Player Profile", "Basic biometric and identity information")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["age"] = st.number_input("🎂 Age", min_value=16, max_value=45, value=26, step=1)
            user_inputs["nationality"] = st.selectbox("🌍 Nationality", get_categorical_options("nationality"))
        with c2:
            user_inputs["height_cm"] = st.number_input("📏 Height (cm)", min_value=150.0, max_value=210.0, value=180.0, step=0.5)
            user_inputs["team"] = st.selectbox("🏳️ Team", get_categorical_options("team"))
        with c3:
            user_inputs["weight_kg"] = st.number_input("⚖️ Weight (kg)", min_value=45.0, max_value=110.0, value=75.0, step=0.5)
            user_inputs["position"] = st.selectbox("🎯 Position", get_categorical_options("position"))
        c4, c5 = st.columns(2)
        with c4:
            user_inputs["preferred_foot"] = st.selectbox("🦶 Preferred Foot", get_categorical_options("preferred_foot"))
        with c5:
            pass
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- MATCH CONTEXT ----------------
    section_header("🏟️ Match Context", "Opponent, tournament stage, and match outcome")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["opponent_team"] = st.selectbox("🆚 Opponent Team", get_categorical_options("opponent_team"))
        with c2:
            user_inputs["tournament_stage"] = st.selectbox("🏆 Tournament Stage", get_categorical_options("tournament_stage"))
        with c3:
            user_inputs["match_result"] = st.selectbox("📣 Match Result", get_categorical_options("match_result"))
        user_inputs["minutes_played"] = st.slider("⏱️ Minutes Played", 0, 120, 90)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- ATTACKING STATS ----------------
    section_header("⚔️ Attacking Statistics", "Shooting, expected goals, and dribbling output")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["shots"] = st.number_input("🎯 Shots", 0, 20, 3)
            user_inputs["dribbles_attempted"] = st.number_input("🌀 Dribbles Attempted", 0, 30, 4)
            user_inputs["crosses"] = st.number_input("🎪 Crosses", 0, 20, 2)
        with c2:
            user_inputs["shots_on_target"] = st.number_input("🥅 Shots on Target", 0, 15, 2)
            user_inputs["successful_dribbles"] = st.number_input("✅ Successful Dribbles", 0, 30, 2)
            user_inputs["successful_crosses"] = st.number_input("✅ Successful Crosses", 0, 20, 1)
        with c3:
            user_inputs["expected_goals_xg"] = st.number_input("📈 Expected Goals (xG)", 0.0, 5.0, 0.3, step=0.01)
            user_inputs["expected_assists_xa"] = st.number_input("🎁 Expected Assists (xA)", 0.0, 5.0, 0.2, step=0.01)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PASSING STATS ----------------
    section_header("🎯 Passing Statistics", "Distribution accuracy and creativity")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["key_passes"] = st.number_input("🔑 Key Passes", 0, 20, 2)
        with c2:
            user_inputs["successful_passes"] = st.number_input("✅ Successful Passes", 0, 150, 40)
        with c3:
            user_inputs["total_passes"] = st.number_input("📊 Total Passes", 0, 200, 50)
        user_inputs["pass_accuracy"] = st.slider("🎯 Pass Accuracy (%)", 0.0, 100.0, 82.0)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- DEFENSIVE STATS ----------------
    section_header("🛡️ Defensive Statistics", "Tackles, interceptions, and aerial duels")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["tackles"] = st.number_input("🦵 Tackles", 0, 15, 2)
            user_inputs["clearances"] = st.number_input("🧹 Clearances", 0, 20, 2)
            user_inputs["recoveries"] = st.number_input("♻️ Recoveries", 0, 20, 4)
        with c2:
            user_inputs["interceptions"] = st.number_input("🛑 Interceptions", 0, 15, 2)
            user_inputs["blocks"] = st.number_input("🚧 Blocks", 0, 15, 1)
            user_inputs["defensive_actions"] = st.number_input("🛡️ Defensive Actions", 0, 30, 5)
        with c3:
            user_inputs["aerial_duels_won"] = st.number_input("🦅 Aerial Duels Won", 0, 15, 2)
            user_inputs["aerial_duels_lost"] = st.number_input("💨 Aerial Duels Lost", 0, 15, 1)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- DISCIPLINE ----------------
    section_header("🟨 Discipline & Fouls", "Cards, fouls, and offsides")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["fouls_committed"] = st.number_input("🟥 Fouls Committed", 0, 10, 1)
        with c2:
            user_inputs["fouls_suffered"] = st.number_input("🤕 Fouls Suffered", 0, 10, 1)
        with c3:
            user_inputs["offsides"] = st.number_input("🚩 Offsides", 0, 10, 0)
        c4, c5 = st.columns(2)
        with c4:
            user_inputs["yellow_cards"] = st.selectbox("🟨 Yellow Cards", [0, 1, 2])
        with c5:
            user_inputs["red_cards"] = st.selectbox("🟥 Red Cards", [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- GOALKEEPING ----------------
    section_header("🧤 Goalkeeping Statistics", "Only relevant for goalkeepers — leave default otherwise")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["saves"] = st.number_input("🧤 Saves", 0, 15, 0)
            user_inputs["clean_sheet"] = st.selectbox("🚫 Clean Sheet", [0, 1])
        with c2:
            user_inputs["save_percentage"] = st.slider("📈 Save Percentage (%)", 0.0, 100.0, 0.0)
            user_inputs["goals_conceded"] = st.number_input("🥅 Goals Conceded", 0, 10, 0)
        with c3:
            user_inputs["punches"] = st.number_input("👊 Punches", 0, 10, 0)
            user_inputs["penalty_saves"] = st.number_input("🎯 Penalty Saves", 0, 5, 0)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PHYSICAL STATS ----------------
    section_header("🏃 Physical Performance", "Distance, speed, and stamina metrics")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["distance_covered_km"] = st.number_input("📍 Distance Covered (km)", 0.0, 15.0, 9.5, step=0.1)
            user_inputs["accelerations"] = st.number_input("⚡ Accelerations", 0, 60, 15)
        with c2:
            user_inputs["sprint_distance_km"] = st.number_input("💨 Sprint Distance (km)", 0.0, 5.0, 1.2, step=0.1)
            user_inputs["decelerations"] = st.number_input("🛑 Decelerations", 0, 60, 14)
        with c3:
            user_inputs["top_speed_kmh"] = st.number_input("🚀 Top Speed (km/h)", 0.0, 40.0, 29.0, step=0.1)
            user_inputs["stamina_score"] = st.slider("🔋 Stamina Score", 0.0, 100.0, 78.0)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- ADVANCED METRICS ----------------
    section_header("🧠 Advanced Performance Metrics", "Composite AI-derived performance indices")
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["offensive_contribution"] = st.slider("⚔️ Offensive Contribution", 0.0, 100.0, 60.0)
            user_inputs["pressure_resistance"] = st.slider("🧊 Pressure Resistance", 0.0, 100.0, 55.0)
        with c2:
            user_inputs["defensive_contribution"] = st.slider("🛡️ Defensive Contribution", 0.0, 100.0, 50.0)
            user_inputs["creativity_score"] = st.slider("🎨 Creativity Score", 0.0, 100.0, 58.0)
        with c3:
            user_inputs["possession_impact"] = st.slider("🔄 Possession Impact", 0.0, 100.0, 52.0)
            user_inputs["consistency_score"] = st.slider("📏 Consistency Score", 0.0, 100.0, 65.0)
        user_inputs["clutch_performance_score"] = st.slider("🔥 Clutch Performance Score", 0.0, 100.0, 60.0)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- PREDICT BUTTON ----------------
    predict_clicked = st.button("🚀 PREDICT PLAYER OF THE MATCH", use_container_width=True)

    if predict_clicked:
        if MODEL is None or SCALER is None or LABEL_ENCODERS is None:
            st.markdown(
                """
                <div class="glass-card" style="border-color:rgba(255,120,120,0.5);">
                    <h4 style="color:#ff8a8a;">❌ Prediction Unavailable</h4>
                    <p style="color:#e0b5b5;">One or more required model artifacts
                    (model / scaler / encoders) could not be loaded. Please verify the
                    files exist in the app directory.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                with st.spinner("⚙️ Running AI inference engine..."):
                    time.sleep(0.6)  # subtle loading animation feel

                    unseen_warnings = []
                    row = {}

                    for feature in FEATURE_ORDER:
                        raw_value = user_inputs.get(feature)

                        if feature in CATEGORICAL_FEATURES:
                            encoded_val, was_unseen = safe_label_encode(feature, raw_value)
                            if was_unseen:
                                unseen_warnings.append(feature)
                            row[feature] = encoded_val
                        else:
                            # basic numeric validation
                            try:
                                row[feature] = float(raw_value)
                            except (TypeError, ValueError):
                                row[feature] = 0.0

                    # Build feature vector in the EXACT training order
                    feature_df = pd.DataFrame([row], columns=FEATURE_ORDER)

                    # Scale the full feature vector using the saved StandardScaler
                    try:
                        
                        scaled_array = SCALER.transform(feature_df.values)
                    except Exception:
                        # Fallback: scaler may have been fit only on numerical columns
                        scaled_numeric = SCALER.transform(feature_df[NUMERICAL_FEATURES].values)
                        cat_array = feature_df[CATEGORICAL_FEATURES].values
                        scaled_array = np.concatenate([cat_array, scaled_numeric], axis=1)

                    # ---- Prediction ----
                    prediction = MODEL.predict(scaled_array)[0]

                    confidence = None
                    if hasattr(MODEL, "predict_proba"):
                        try:
                            proba = MODEL.predict_proba(scaled_array)[0]
                            confidence = float(np.max(proba)) * 100
                        except Exception:
                            confidence = None
                    elif hasattr(MODEL, "decision_function"):
                        try:
                            score = MODEL.decision_function(scaled_array)[0]
                            confidence = float(1 / (1 + np.exp(-score))) * 100
                        except Exception:
                            confidence = None

                if unseen_warnings:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-color:rgba(255,200,0,0.4);">
                            <p style="color:#ffd166;">⚠️ Note: The following categorical inputs were not seen
                            during training and were substituted with a default known category:
                            <b>{', '.join(unseen_warnings)}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                is_potm = int(prediction) == 1

                if is_potm:
                    st.balloons()
                    conf_html = ""
                    if confidence is not None:
                        conf_html = f"""
                        <div style="margin-top:1.2rem;">
                            <p style="color:#9fb4d8;">Prediction Confidence</p>
                            <div class="confidence-bar-wrap">
                                <div class="confidence-bar-fill" style="width:{confidence:.1f}%;"></div>
                            </div>
                            <p style="color:#00ffa3;font-weight:700;margin-top:0.4rem;">{confidence:.1f}%</p>
                        </div>
                        """
                    st.markdown(
                        f"""
                        <div class="result-success">
                            <span class="result-icon">🏆</span>
                            <div class="result-title" style="color:#00ffa3;">PLAYER OF THE MATCH!</div>
                            <p style="color:#c8f5e3;font-size:1.05rem;">
                            The model predicts this player has a strong chance of being named
                            <b>Player of the Match</b> based on the provided performance data.
                            </p>
                            {conf_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    conf_html = ""
                    if confidence is not None:
                        conf_html = f"""
                        <div style="margin-top:1.2rem;">
                            <p style="color:#9fb4d8;">Prediction Confidence</p>
                            <div class="confidence-bar-wrap">
                                <div class="confidence-bar-fill" style="width:{confidence:.1f}%;"></div>
                            </div>
                            <p style="color:#4fd8ff;font-weight:700;margin-top:0.4rem;">{confidence:.1f}%</p>
                        </div>
                        """
                    st.markdown(
                        f"""
                        <div class="result-fail">
                            <span class="result-icon">📋</span>
                            <div class="result-title" style="color:#ff9d9d;">Not Predicted as POTM</div>
                            <p style="color:#d8c8c8;font-size:1.05rem;">
                            Based on the current performance profile, the model does not predict this
                            player as Player of the Match for this fixture. Try adjusting key attacking,
                            defensive, or advanced metric values to explore different scenarios.
                            </p>
                            {conf_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-color:rgba(255,120,120,0.5);">
                        <h4 style="color:#ff8a8a;">❌ Prediction Error</h4>
                        <p style="color:#e0b5b5;">An unexpected error occurred while generating the
                        prediction:</p>
                        <code style="color:#ffb3b3;">{str(e)}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==================================================================================
# PAGE: DATASET OVERVIEW
# ==================================================================================
elif page == "📊 Dataset Overview":
    hero_section(
        "📊 DATA INTELLIGENCE",
        "Dataset Overview",
        "A summary of the structure and composition of the training dataset used to "
        "build the Player of the Match prediction model.",
    )

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total Features", len(FEATURE_ORDER), "🧬")
    metric_card(c2, "Categorical", len(CATEGORICAL_FEATURES), "🏷️")
    metric_card(c3, "Numerical", len(NUMERICAL_FEATURES), "🔢")
    metric_card(c4, "Target Variable", "1", "🎯")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🗂️ Feature Categories")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🏷️ Categorical Features")
        for feat in CATEGORICAL_FEATURES:
            st.markdown(f'<span class="chip">{feat}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔢 Numerical Features")
        chips = "".join(f'<span class="chip">{f}</span>' for f in NUMERICAL_FEATURES)
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("🎯 Target Variable")
    st.markdown(
        f"""
        <div class="glass-card">
            <p style="color:#c8d7f5;">
            <b>{TARGET_COLUMN}</b> — a binary label indicating whether a player was awarded
            <i>Player of the Match</i> (1) or not (0) for a given fixture.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# PAGE: ABOUT THE MODEL
# ==================================================================================
elif page == "🧠 About the Model":
    hero_section(
        "🧠 MODEL INTELLIGENCE",
        "About the Model",
        "Technical details of the machine learning pipeline powering this prediction system.",
    )

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Algorithm", "SVC", "🤖")
    metric_card(c2, "Scaling", "StandardScaler", "📐")
    metric_card(c3, "Encoding", "LabelEncoder", "🏷️")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h4>🔬 Algorithm</h4>
                <p style="color:#9fb4d8;">
                A <b>Support Vector Classifier (SVC)</b> constructs an optimal decision
                boundary (hyperplane) in high-dimensional feature space to separate
                Player-of-the-Match performances from non-award-winning performances.
                Kernel functions allow the model to capture non-linear relationships
                between attacking output, defensive workrate, and physical intensity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="glass-card">
                <h4>📐 Feature Scaling</h4>
                <p style="color:#9fb4d8;">
                All numerical features are standardized using a <b>StandardScaler</b>
                fitted during training, ensuring each feature contributes proportionally
                to the SVC's margin optimization regardless of its original scale.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4>🏷️ Categorical Encoding</h4>
                <p style="color:#9fb4d8;">
                Categorical attributes such as nationality, team, position, and match
                context are transformed into numeric representations using fitted
                <b>LabelEncoders</b> saved from the training notebook, guaranteeing
                identical encoding behaviour at inference time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="glass-card">
                <h4>🧪 Inference Pipeline</h4>
                <p style="color:#9fb4d8;">
                1. Collect raw inputs →
                2. Encode categorical fields →
                3. Assemble feature vector in training order →
                4. Scale using the saved StandardScaler →
                5. Predict using the SVC model →
                6. Derive confidence via <code>predict_proba</code> /
                <code>decision_function</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================================================
# PAGE: PREDICTION GUIDE
# ==================================================================================
elif page == "📘 Prediction Guide":
    hero_section(
        "📘 USER GUIDE",
        "Prediction Guide",
        "Step-by-step instructions for generating accurate predictions.",
    )

    steps = [
        ("1️⃣", "Navigate to Predict", "Open the '🔮 Predict Player of the Match' tab from the sidebar."),
        ("2️⃣", "Enter Player Profile", "Provide age, physical attributes, nationality, team, and position."),
        ("3️⃣", "Set Match Context", "Select opponent, tournament stage, match result, and minutes played."),
        ("4️⃣", "Fill Performance Stats", "Enter attacking, passing, defensive, discipline, and goalkeeping stats."),
        ("5️⃣", "Adjust Physical & Advanced Metrics", "Fine-tune distance, speed, stamina, and composite AI scores."),
        ("6️⃣", "Run Prediction", "Click the glowing 'PREDICT PLAYER OF THE MATCH' button."),
        ("7️⃣", "Interpret Results", "Review the prediction outcome and confidence score displayed in the result card."),
    ]

    for icon, title, desc in steps:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; align-items:flex-start; gap:1rem;">
                <div style="font-size:1.6rem;">{icon}</div>
                <div>
                    <h4 style="margin-bottom:0.2rem;">{title}</h4>
                    <p style="color:#9fb4d8; margin:0;">{desc}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="border-color:rgba(255,200,0,0.35);">
            <h4 style="color:#ffd166;">💡 Tips for Accurate Predictions</h4>
            <ul style="color:#c8d7f5;">
                <li>Use realistic values consistent with actual match statistics.</li>
                <li>Goalkeeping stats should only be populated for goalkeepers.</li>
                <li>Advanced metric sliders (0–100) represent normalized composite scores.</li>
                <li>Ensure the model artifacts are up to date with your latest training run.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# PAGE: FEATURE INFORMATION
# ==================================================================================
elif page == "🧬 Feature Information":
    hero_section(
        "🧬 FEATURE REFERENCE",
        "Feature Information",
        "Detailed reference of every input feature used by the prediction model.",
    )

    feature_groups = {
        "🧍 Player Profile": ["age", "nationality", "team", "position", "height_cm", "weight_kg", "preferred_foot"],
        "🏟️ Match Context": ["opponent_team", "tournament_stage", "match_result", "minutes_played"],
        "⚔️ Attacking": ["shots", "shots_on_target", "expected_goals_xg", "expected_assists_xa",
                          "dribbles_attempted", "successful_dribbles", "crosses", "successful_crosses"],
        "🎯 Passing": ["key_passes", "successful_passes", "total_passes", "pass_accuracy"],
        "🛡️ Defensive": ["tackles", "interceptions", "clearances", "blocks", "aerial_duels_won",
                          "aerial_duels_lost", "recoveries", "defensive_actions"],
        "🟨 Discipline": ["fouls_committed", "fouls_suffered", "yellow_cards", "red_cards", "offsides"],
        "🧤 Goalkeeping": ["saves", "save_percentage", "punches", "clean_sheet", "goals_conceded", "penalty_saves"],
        "🏃 Physical": ["distance_covered_km", "sprint_distance_km", "top_speed_kmh", "accelerations",
                        "decelerations", "stamina_score"],
        "🧠 Advanced Metrics": ["offensive_contribution", "defensive_contribution", "possession_impact",
                                 "pressure_resistance", "creativity_score", "consistency_score",
                                 "clutch_performance_score"],
    }

    for group_name, feats in feature_groups.items():
        st.markdown(f'<div class="section-header">{group_name}</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="chip">{f}</span>' for f in feats)
        st.markdown(f'<div class="glass-card">{chips}</div>', unsafe_allow_html=True)





