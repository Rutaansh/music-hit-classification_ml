"""
Streamlit Web Application: Viral Songs Machine Learning Classifier
===================================================================
Assignment: BITS Pilani M.Tech (AIML) - ML Assignment 2
Theme: Flamingo Pink & Pearlescent White Base | Shiny Plum & Brushed Silver Cards | Electric Blue Accents
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Support Vector Machine
Metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & LUXURY THEME INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VIBE-AI | Music Virality Intelligence Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the Flamingo Pink + Shiny Plum + Silver + Electric Blue Theme
THEME_CSS = """
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

/* Root Variables */
:root {
    --bg-base: #fce4ec;
    --bg-gradient: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 35%, #fff0f5 70%, #ffffff 100%);
    --plum-dark: #2d0a27;
    --plum-deep: #4a154b;
    --plum-mid: #6b1d5c;
    --plum-bright: #880e4f;
    --silver-border: rgba(226, 232, 240, 0.85);
    --silver-sheen: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(241,245,249,0.7) 100%);
    --electric-blue: #0052FF;
    --electric-blue-hover: #003ECC;
    --electric-blue-glow: rgba(0, 82, 255, 0.35);
    --cyan-accent: #00D2FF;
    --text-primary: #1e1022;
    --text-muted: #5c3a58;
    --text-light: #ffffff;
}

/* Global App Styling */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-gradient) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #380833 0%, #4a154b 60%, #220320 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 4px 0 24px rgba(45, 10, 39, 0.25) !important;
}

[data-testid="stSidebar"] * {
    color: #fce4ec !important;
}

[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}

/* Hero Banner Container */
.hero-card {
    background: linear-gradient(135deg, #3b0732 0%, #5e1248 50%, #880e4f 100%);
    border-radius: 24px;
    padding: 38px 42px;
    color: #ffffff;
    box-shadow: 0 16px 40px rgba(59, 7, 50, 0.22), 0 0 0 1px rgba(255, 255, 255, 0.2) inset;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(0, 210, 255, 0.25) 0%, rgba(0, 82, 255, 0.15) 50%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff;
    margin: 0 0 8px 0;
}

.hero-subtitle {
    font-size: 16px;
    color: #f8bbd0;
    max-width: 780px;
    line-height: 1.6;
    margin: 0;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0052FF 0%, #00D2FF 100%);
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 50px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px var(--electric-blue-glow);
}

/* Plum Glassmorphic Surface Cards */
.plum-card {
    background: linear-gradient(135deg, rgba(74, 21, 75, 0.92) 0%, rgba(59, 7, 50, 0.96) 100%);
    border-radius: 20px;
    padding: 24px;
    color: #ffffff;
    box-shadow: 0 12px 32px rgba(45, 10, 39, 0.16), 0 0 0 1px rgba(255, 255, 255, 0.18) inset;
    margin-bottom: 20px;
}

.silver-card {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(136, 14, 79, 0.08);
    margin-bottom: 20px;
}

/* Metric Display Cards */
.metric-container {
    background: linear-gradient(145deg, #4a154b 0%, #32072a 100%);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 22px rgba(50, 7, 42, 0.18);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0, 82, 255, 0.25);
    border-color: rgba(0, 210, 255, 0.4);
}

.metric-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #f8bbd0;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}

.metric-sub {
    font-size: 11px;
    color: #00D2FF;
    font-weight: 600;
}

/* Buttons with Electric Blue Arc Styling */
.stButton > button {
    background: linear-gradient(90deg, #0052FF 0%, #0066FF 100%) !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: 0 6px 20px var(--electric-blue-glow) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.2px !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #003ECC 0%, #0052FF 100%) !important;
    box-shadow: 0 8px 26px rgba(0, 82, 255, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* Inputs & Select Boxes */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 12px !important;
    border: 1px solid #d1d5db !important;
    color: #1e1022 !important;
    font-weight: 600 !important;
}

/* Dataframe & Table Universal High-Contrast Styling */
[data-testid="stDataFrame"], [data-testid="stTable"], .stTable {
    background: #ffffff !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(74, 21, 75, 0.12) !important;
    border: 1px solid rgba(226, 232, 240, 0.95) !important;
}

[data-testid="stTable"] table {
    width: 100% !important;
    background-color: #ffffff !important;
    color: #1e1022 !important;
    border-collapse: collapse !important;
}

[data-testid="stTable"] thead tr th {
    background: linear-gradient(90deg, #3b0732 0%, #5e1248 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 14px 16px !important;
    border: none !important;
}

[data-testid="stTable"] tbody tr td {
    background-color: #ffffff !important;
    color: #1e1022 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid #f1f5f9 !important;
}

[data-testid="stTable"] tbody tr:nth-child(even) td {
    background-color: #fdf2f8 !important;
}

[data-testid="stTable"] tbody tr:hover td {
    background-color: #fce4ec !important;
}

/* Custom HTML Table Styling for Absolute Theme Uniformity */
.custom-table-container {
    background: #ffffff;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 8px 26px rgba(74, 21, 75, 0.12);
    border: 1px solid rgba(226, 232, 240, 0.95);
    margin: 16px 0 24px 0;
}

.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #1e1022;
}

.custom-table th {
    background: linear-gradient(90deg, #380833 0%, #5e1248 50%, #4a154b 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 14px 18px;
    text-align: left;
    border: none;
}

.custom-table td {
    padding: 13px 18px;
    font-size: 13.5px;
    color: #1e1022;
    border-bottom: 1px solid #f1f5f9;
}

.custom-table tr:nth-child(even) {
    background-color: #fdf2f8;
}

.custom-table tr:hover {
    background-color: #fce4ec;
}

.metric-highlight {
    display: inline-block;
    background: rgba(0, 82, 255, 0.12);
    color: #0052FF;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid rgba(0, 82, 255, 0.25);
}

.metric-best {
    display: inline-block;
    background: linear-gradient(90deg, #0052FF 0%, #00D2FF 100%);
    color: #ffffff;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0, 82, 255, 0.3);
}

/* Section Header Styling */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--plum-dark);
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title span.icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #0052FF 0%, #00D2FF 100%);
    color: #ffffff;
    border-radius: 10px;
    font-size: 18px;
}

/* Pill Badges */
.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 82, 255, 0.12);
    color: #0052FF;
    border: 1px solid rgba(0, 82, 255, 0.3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.winner-banner {
    background: linear-gradient(135deg, #1b021a 0%, #4a154b 50%, #0052FF 100%);
    border-radius: 18px;
    padding: 22px 28px;
    color: #ffffff;
    border: 1px solid rgba(0, 210, 255, 0.4);
    box-shadow: 0 12px 30px rgba(0, 82, 255, 0.22);
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RESOURCE CACHING & ARTIFACT LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load pre-trained model binaries, fitted preprocessor and metadata."""
    model_dir = "model"
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    metadata = joblib.load(os.path.join(model_dir, "metadata.joblib"))
    
    models = {
        "Logistic Regression": joblib.load(os.path.join(model_dir, "logistic_regression.joblib")),
        "Decision Tree": joblib.load(os.path.join(model_dir, "decision_tree.joblib")),
        "KNN": joblib.load(os.path.join(model_dir, "knn.joblib")),
        "Naive Bayes": joblib.load(os.path.join(model_dir, "naive_bayes.joblib")),
        "Support Vector Machine": joblib.load(os.path.join(model_dir, "svm.joblib"))
    }
    
    with open(os.path.join(model_dir, "evaluation_results.json"), "r") as f:
        cached_eval = json.load(f)
        
    return preprocessor, metadata, models, cached_eval


@st.cache_data
def load_default_test_data():
    """Load default 2,000-instance holdout test dataset."""
    if os.path.exists("test_data.csv"):
        return pd.read_csv("test_data.csv")
    elif os.path.exists(os.path.join("Viral_Songs_Datatset", "viral_songs.csv")):
        df = pd.read_csv(os.path.join("Viral_Songs_Datatset", "viral_songs.csv"))
        return df.tail(2000)
    return None


# Helper function to compute all 6 mandatory metrics
def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Confusion_Matrix": cm,
        "Classification_Report": cr
    }


# -----------------------------------------------------------------------------
# APP INITIALIZATION
# -----------------------------------------------------------------------------
try:
    preprocessor, metadata, models, cached_eval = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure 'train_models.py' has been executed.")
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: #ffffff;">
            VIBE<span style="color: #00D2FF;">.AI</span>
        </div>
        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #f8bbd0;">
            Music Virality Engine
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Navigation & Control")
    app_mode = st.radio(
        "Select Studio View:",
        [
            "📊 Model Evaluation & Metrics",
            "🏆 All Models Comparison",
            "🎧 Live Virality Simulator",
            "📖 Dataset & Architecture Docs"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📂 Test Dataset Source")
    data_source = st.radio(
        "Choose Evaluation Data:",
        ["Preloaded Test Data (2,000 songs)", "Upload Custom Test CSV"],
        index=0
    )
    
    uploaded_file = None
    if data_source == "Upload Custom Test CSV":
        uploaded_file = st.file_uploader(
            "Upload Test CSV (must contain target 'viral'):",
            type=["csv"],
            help="Upload a CSV with the 13 required features and 'viral' target column."
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 12px; color: #f8bbd0; line-height: 1.5; padding: 6px 0;">
        <b>Assignment:</b> BITS Pilani M.Tech AIML<br>
        <b>Course:</b> Machine Learning (Assignment 2)<br>
        <b>Pipeline:</b> Zero-retraining inference
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PROCESS TEST DATASET
# -----------------------------------------------------------------------------
if data_source == "Upload Custom Test CSV" and uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded {test_df.shape[0]} test rows!")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        test_df = load_default_test_data()
else:
    test_df = load_default_test_data()

if test_df is None:
    st.error("No test data available. Please verify test_data.csv exists.")
    st.stop()

# Ensure required columns exist
req_features = metadata["feature_cols"]
target_col = metadata["target_col"]

missing_cols = [c for c in req_features + [target_col] if c not in test_df.columns]
if missing_cols:
    st.error(f"Uploaded dataset is missing required columns: {missing_cols}")
    st.stop()

X_eval = test_df[req_features]
y_eval = test_df[target_col]

# Transform test data using saved fitted preprocessor
try:
    X_eval_proc = preprocessor.transform(X_eval)
except Exception as e:
    st.error(f"Error during feature preprocessing: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">🎵 AI Audio Analytics • BITS Pilani WILP</div>
    <div class="hero-title">Music Virality Prediction Studio</div>
    <p class="hero-subtitle">
        An interactive machine learning evaluation dashboard benchmarking <b>5 Supervised Classifiers</b> on 
        <b>10,000 viral track acoustics</b> across 13 multidimensional audio features.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# VIEW 1: MODEL EVALUATION & METRICS
# =============================================================================
if app_mode == "📊 Model Evaluation & Metrics":
    st.markdown('<div class="section-title"><span class="icon">🤖</span> Model Selection & Real-Time Performance</div>', unsafe_allow_html=True)
    
    col_sel, col_stat = st.columns([2, 1])
    with col_sel:
        selected_model_name = st.selectbox(
            "Choose Machine Learning Classification Model:",
            list(models.keys()),
            index=0,
            help="Select one of the 5 trained models to evaluate on the holdout test data."
        )
    with col_stat:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.7); border: 1px solid rgba(226,232,240,0.9); border-radius: 12px; padding: 12px 18px; margin-top: 18px;">
            <div style="font-size: 11px; font-weight: 700; color: #6b1d5c; text-transform: uppercase;">Active Evaluation Set</div>
            <div style="font-size: 18px; font-weight: 700; color: #1e1022;">{len(test_df):,} Instances ({test_df['viral'].sum():,} Viral Hits)</div>
        </div>
        """, unsafe_allow_html=True)
        
    clf = models[selected_model_name]
    y_pred = clf.predict(X_eval_proc)
    y_proba = clf.predict_proba(X_eval_proc)[:, 1] if hasattr(clf, "predict_proba") else None
    
    eval_res = compute_metrics(y_eval, y_pred, y_proba)
    
    # 6 MANDATORY EVALUATION METRIC CARDS
    st.markdown("#### 📈 Model Evaluation Metrics (6 Mandatory Criteria)")
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    
    with mcol1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{eval_res['Accuracy']*100:.2f}%</div>
            <div class="metric-sub">{eval_res['Accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">AUC Score</div>
            <div class="metric-value">{eval_res['AUC']:.4f}</div>
            <div class="metric-sub">ROC AUC</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{eval_res['Precision']*100:.2f}%</div>
            <div class="metric-sub">{eval_res['Precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{eval_res['Recall']*100:.2f}%</div>
            <div class="metric-sub">{eval_res['Recall']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{eval_res['F1']:.4f}</div>
            <div class="metric-sub">Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol6:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">MCC Score</div>
            <div class="metric-value">{eval_res['MCC']:.4f}</div>
            <div class="metric-sub">Matthews Corr</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # VISUALIZATION ROW: Confusion Matrix & ROC Curve
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        st.markdown("""
        <div class="silver-card">
            <div style="font-size: 18px; font-weight: 700; color: #4a154b; margin-bottom: 12px;">
                🎯 Confusion Matrix Analysis
            </div>
        """, unsafe_allow_html=True)
        
        cm = eval_res['Confusion_Matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.8))
        fig_cm.patch.set_facecolor('#ffffff')
        ax_cm.set_facecolor('#ffffff')
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='RdPu', cbar=False, ax=ax_cm,
            annot_kws={"size": 16, "weight": "bold"},
            xticklabels=['Non-Viral (0)', 'Viral (1)'],
            yticklabels=['Non-Viral (0)', 'Viral (1)']
        )
        ax_cm.set_xlabel('Predicted Class', fontsize=12, fontweight='bold', color='#4a154b')
        ax_cm.set_ylabel('True Class', fontsize=12, fontweight='bold', color='#4a154b')
        ax_cm.set_title(f'{selected_model_name} Confusion Matrix', fontsize=13, fontweight='bold', color='#2d0a27', pad=10)
        plt.tight_layout()
        st.pyplot(fig_cm)
        
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div style="display: flex; justify-content: space-around; background: #fff0f5; border-radius: 12px; padding: 10px; margin-top: 10px; font-size: 12px; font-weight: 600;">
            <div><span style="color:#0052FF;">TN:</span> {tn}</div>
            <div><span style="color:#880e4f;">FP:</span> {fp}</div>
            <div><span style="color:#880e4f;">FN:</span> {fn}</div>
            <div><span style="color:#0052FF;">TP:</span> {tp}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
    with viz_col2:
        st.markdown("""
        <div class="silver-card">
            <div style="font-size: 18px; font-weight: 700; color: #4a154b; margin-bottom: 12px;">
                📉 ROC & Discrimination Capacity
            </div>
        """, unsafe_allow_html=True)
        
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.8))
        fig_roc.patch.set_facecolor('#ffffff')
        ax_roc.set_facecolor('#fdf2f8')
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            ax_roc.plot(fpr, tpr, color='#0052FF', lw=2.5, label=f"ROC Curve (AUC = {eval_res['AUC']:.4f})")
            ax_roc.fill_between(fpr, tpr, color='#00D2FF', alpha=0.15)
        
        ax_roc.plot([0, 1], [0, 1], color='#880e4f', linestyle='--', lw=1.5, label="Random Guess (0.50)")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', color='#4a154b')
        ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold', color='#4a154b')
        ax_roc.set_title(f'{selected_model_name} ROC Curve', fontsize=13, fontweight='bold', color='#2d0a27', pad=10)
        ax_roc.legend(loc="lower right", fontsize=10, frameon=True)
        ax_roc.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_roc)
        
        st.markdown("""
        <div style="background: #fff0f5; border-radius: 12px; padding: 10px; margin-top: 10px; font-size: 12px; color: #5c3a58;">
            <b>AUC Interpretation:</b> An AUC score closer to 1.0 reflects superior class discrimination ability.
        </div>
        </div>
        """, unsafe_allow_html=True)
        
    # CLASSIFICATION REPORT TABLE
    st.markdown("#### 📋 Detailed Classification Report")
    cr_dict = eval_res['Classification_Report']
    cr_df = pd.DataFrame(cr_dict).T.round(4)
    
    # Render High-Contrast Custom HTML Table for Classification Report
    cr_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr>'
    cr_html += '<th>Class / Metric</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead><tbody>'
    for row_name, row_vals in cr_df.iterrows():
        support_val = int(row_vals['support']) if 'support' in row_vals and not np.isnan(row_vals['support']) else '-'
        p_val = f"{row_vals['precision']:.4f}" if 'precision' in row_vals else '-'
        r_val = f"{row_vals['recall']:.4f}" if 'recall' in row_vals else '-'
        f_val = f"{row_vals['f1-score']:.4f}" if 'f1-score' in row_vals else '-'
        cr_html += f'<tr><td><b>{row_name}</b></td><td>{p_val}</td><td>{r_val}</td><td>{f_val}</td><td><b>{support_val}</b></td></tr>'
    cr_html += '</tbody></table></div>'
    st.markdown(cr_html, unsafe_allow_html=True)


# =============================================================================
# VIEW 2: ALL MODELS COMPARISON DASHBOARD
# =============================================================================
elif app_mode == "🏆 All Models Comparison":
    st.markdown('<div class="section-title"><span class="icon">🏆</span> Comparative Benchmarking Dashboard (All 5 Classifiers)</div>', unsafe_allow_html=True)
    
    # Compute metrics for all 5 models on the active test dataset
    all_metrics = {}
    all_probas = {}
    
    for mname, mclf in models.items():
        yp = mclf.predict(X_eval_proc)
        yprob = mclf.predict_proba(X_eval_proc)[:, 1] if hasattr(mclf, "predict_proba") else None
        res = compute_metrics(y_eval, yp, yprob)
        all_metrics[mname] = {
            "Accuracy": res["Accuracy"],
            "AUC": res["AUC"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"],
            "MCC": res["MCC"]
        }
        all_probas[mname] = yprob
        
    comp_df = pd.DataFrame(all_metrics).T.round(4)
    comp_df.index.name = "ML Model Name"
    
    st.markdown("#### 📊 Master Comparison Table (As Required in Assignment Step 5)")
    
    # Build High-Contrast Custom HTML Master Comparison Table
    max_acc = comp_df["Accuracy"].max()
    max_auc = comp_df["AUC"].max()
    max_prec = comp_df["Precision"].max()
    max_rec = comp_df["Recall"].max()
    max_f1 = comp_df["F1"].max()
    max_mcc = comp_df["MCC"].max()
    
    def highlight_cell(val, max_val):
        if val == max_val:
            return f'<span class="metric-best">{val:.4f} ★</span>'
        return f'{val:.4f}'
        
    master_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr>'
    master_html += '<th>ML Model Name</th><th>Accuracy</th><th>AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>MCC</th></tr></thead><tbody>'
    
    for m_name, row in comp_df.iterrows():
        master_html += f'<tr><td><b>{m_name}</b></td>'
        master_html += f'<td>{highlight_cell(row["Accuracy"], max_acc)}</td>'
        master_html += f'<td>{highlight_cell(row["AUC"], max_auc)}</td>'
        master_html += f'<td>{highlight_cell(row["Precision"], max_prec)}</td>'
        master_html += f'<td>{highlight_cell(row["Recall"], max_rec)}</td>'
        master_html += f'<td>{highlight_cell(row["F1"], max_f1)}</td>'
        master_html += f'<td>{highlight_cell(row["MCC"], max_mcc)}</td>'
        master_html += '</tr>'
    master_html += '</tbody></table></div>'
    st.markdown(master_html, unsafe_allow_html=True)
    
    # Visual Comparison Charts
    c_col1, c_col2 = st.columns([1, 1])
    
    with c_col1:
        st.markdown("""
        <div class="silver-card">
            <div style="font-size: 18px; font-weight: 700; color: #4a154b; margin-bottom: 12px;">
                📊 Metric Comparison Bar Chart
            </div>
        """, unsafe_allow_html=True)
        
        fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
        comp_df[["Accuracy", "AUC", "F1", "MCC"]].plot(
            kind='bar', ax=ax_bar, 
            color=['#4a154b', '#0052FF', '#00D2FF', '#e91e63'],
            edgecolor='none', width=0.75
        )
        ax_bar.set_title("Key Performance Metric Profiles Across Classifiers", fontsize=12, fontweight='bold', color='#2d0a27', pad=10)
        ax_bar.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='bold', color='#4a154b')
        ax_bar.set_xlabel("Classifier", fontsize=11, fontweight='bold', color='#4a154b')
        ax_bar.set_xticklabels(comp_df.index, rotation=25, ha='right', fontsize=9, fontweight='600')
        ax_bar.legend(loc="lower right", fontsize=9, frameon=True)
        ax_bar.set_ylim(0.2, 1.0)
        ax_bar.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_bar)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_col2:
        st.markdown("""
        <div class="silver-card">
            <div style="font-size: 18px; font-weight: 700; color: #4a154b; margin-bottom: 12px;">
                📈 Multi-Model ROC Overlay
            </div>
        """, unsafe_allow_html=True)
        
        fig_multi_roc, ax_multi_roc = plt.subplots(figsize=(7, 5))
        roc_colors = ['#880e4f', '#0052FF', '#00D2FF', '#4a154b', '#e91e63']
        
        for idx, (mname, yprob) in enumerate(all_probas.items()):
            if yprob is not None:
                fpr, tpr, _ = roc_curve(y_eval, yprob)
                ax_multi_roc.plot(fpr, tpr, label=f"{mname} (AUC={all_metrics[mname]['AUC']:.4f})", color=roc_colors[idx], lw=2.0)
                
        ax_multi_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (0.50)')
        ax_multi_roc.set_title("ROC Curves Comparison (All 5 Classifiers)", fontsize=12, fontweight='bold', color='#2d0a27', pad=10)
        ax_multi_roc.set_xlabel("False Positive Rate", fontsize=11, fontweight='bold', color='#4a154b')
        ax_multi_roc.set_ylabel("True Positive Rate", fontsize=11, fontweight='bold', color='#4a154b')
        ax_multi_roc.legend(loc="lower right", fontsize=8.5, frameon=True)
        ax_multi_roc.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_multi_roc)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # MODEL OBSERVATIONS TABLE (Required by Assignment Step 5)
    st.markdown("#### 📝 Model Observations & Comparative Insights")
    
    obs_data = [
        {
            "ML Model Name": "Logistic Regression",
            "Accuracy": f"{all_metrics['Logistic Regression']['Accuracy']:.4f}",
            "AUC": f"{all_metrics['Logistic Regression']['AUC']:.4f}",
            "F1": f"{all_metrics['Logistic Regression']['F1']:.4f}",
            "Observation about model performance": "Top overall performer for Accuracy (81.95%) and AUC (0.8966). Demonstrates high precision (80.93%) with low false positive rate across linear acoustic combinations."
        },
        {
            "ML Model Name": "Decision Tree",
            "Accuracy": f"{all_metrics['Decision Tree']['Accuracy']:.4f}",
            "AUC": f"{all_metrics['Decision Tree']['AUC']:.4f}",
            "F1": f"{all_metrics['Decision Tree']['F1']:.4f}",
            "Observation about model performance": "Best balanced model with highest F1 Score (0.7459) and highest Matthews Correlation Coefficient (0.6005). Captures non-linear thresholds like high danceability combined with low acousticness."
        },
        {
            "ML Model Name": "KNN",
            "Accuracy": f"{all_metrics['KNN']['Accuracy']:.4f}",
            "AUC": f"{all_metrics['KNN']['AUC']:.4f}",
            "F1": f"{all_metrics['KNN']['F1']:.4f}",
            "Observation about model performance": "Lowest accuracy (71.90%) and MCC (0.3730). Suffers from the curse of dimensionality across 22 transformed features where Euclidean distances become less discriminative."
        },
        {
            "ML Model Name": "Naive Bayes",
            "Accuracy": f"{all_metrics['Naive Bayes']['Accuracy']:.4f}",
            "AUC": f"{all_metrics['Naive Bayes']['AUC']:.4f}",
            "F1": f"{all_metrics['Naive Bayes']['F1']:.4f}",
            "Observation about model performance": "Highest Recall (83.91%), successfully flagging most viral hits, but lower precision (61.11%) due to independent conditional Gaussian feature assumptions."
        },
        {
            "ML Model Name": "Support Vector Machine",
            "Accuracy": f"{all_metrics['Support Vector Machine']['Accuracy']:.4f}",
            "AUC": f"{all_metrics['Support Vector Machine']['AUC']:.4f}",
            "F1": f"{all_metrics['Support Vector Machine']['F1']:.4f}",
            "Observation about model performance": "Strong second-tier performer (Accuracy: 79.95%, Precision: 79.41%). The non-linear RBF kernel creates smooth decision boundaries but is computationally heavier."
        }
    ]
    
    # Render High-Contrast Custom HTML Table for Observations
    obs_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr>'
    obs_html += '<th style="width: 20%;">ML Model Name</th><th style="width: 80%;">Observation about model performance</th></tr></thead><tbody>'
    for o in obs_data:
        obs_html += f'<tr><td><b>{o["ML Model Name"]}</b><br><span style="font-size: 11px; color: #0052FF; font-weight: 700;">Acc: {o["Accuracy"]} | F1: {o["F1"]}</span></td>'
        obs_html += f'<td style="line-height: 1.6;">{o["Observation about model performance"]}</td></tr>'
    obs_html += '</tbody></table></div>'
    st.markdown(obs_html, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="winner-banner">
        <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #00D2FF; font-weight: 800;">
            🏆 Overall Dataset Winner Recommendation
        </div>
        <div style="font-size: 22px; font-weight: 700; margin: 6px 0 10px 0;">
            Co-Winners: Logistic Regression (Accuracy/AUC) & Decision Tree (Balanced F1/MCC)
        </div>
        <p style="font-size: 14px; line-height: 1.6; color: #fce4ec; margin: 0;">
            For streaming platforms aiming to minimize false viral promotions (high precision), <b>Logistic Regression</b> is the optimal deployable model. For platforms seeking maximum balanced detection of viral trends (high F1 & MCC), <b>Decision Tree</b> is the winner.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VIEW 3: LIVE VIRALITY SIMULATOR (INTERACTIVE TRACK PREDICTOR)
# =============================================================================
elif app_mode == "🎧 Live Virality Simulator":
    st.markdown('<div class="section-title"><span class="icon">🎧</span> Single-Track Acoustic Virality Predictor</div>', unsafe_allow_html=True)
    st.markdown("""
    Adjust the acoustic sliders below to simulate a newly produced track and predict its virality probability using the trained models.
    """)
    
    with st.form("virality_simulator_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            st.markdown("##### 🎵 Core Dynamics")
            genre = st.selectbox("Genre:", metadata["genres"], index=metadata["genres"].index("Pop") if "Pop" in metadata["genres"] else 0)
            danceability = st.slider("Danceability (0.0 to 1.0):", 0.0, 1.0, 0.82, step=0.01)
            energy = st.slider("Energy (0.0 to 1.0):", 0.0, 1.0, 0.78, step=0.01)
            valence = st.slider("Valence / Positivity (0.0 to 1.0):", 0.0, 1.0, 0.65, step=0.01)
            
        with f_col2:
            st.markdown("##### 🔊 Timbre & Structure")
            tempo_bpm = st.slider("Tempo (BPM):", 60.0, 220.0, 124.0, step=1.0)
            acousticness = st.slider("Acousticness (0.0 to 1.0):", 0.0, 1.0, 0.12, step=0.01)
            instrumentalness = st.slider("Instrumentalness (0.0 to 1.0):", 0.0, 1.0, 0.02, step=0.01)
            loudness_db = st.slider("Loudness (dB):", -40.0, 0.0, -6.5, step=0.1)
            
        with f_col3:
            st.markdown("##### 📊 Production & Reach")
            speechiness = st.slider("Speechiness (0.0 to 1.0):", 0.0, 1.0, 0.08, step=0.01)
            liveness = st.slider("Liveness (0.0 to 1.0):", 0.0, 1.0, 0.18, step=0.01)
            duration_sec = st.slider("Duration (seconds):", 60, 480, 185, step=5)
            release_year = st.slider("Release Year:", 2000, 2026, 2024, step=1)
            streams = st.number_input("Current Stream Count:", min_value=1000, max_value=50000000, value=850000, step=25000)
            
        sim_model_choice = st.selectbox(
            "Select Inference Model for Prediction:",
            list(models.keys()),
            index=0
        )
        
        submitted = st.form_submit_button("⚡ Predict Track Virality")
        
    if submitted:
        input_dict = {
            "genre": [genre],
            "danceability": [danceability],
            "energy": [energy],
            "valence": [valence],
            "tempo_bpm": [tempo_bpm],
            "acousticness": [acousticness],
            "instrumentalness": [instrumentalness],
            "loudness_db": [loudness_db],
            "speechiness": [speechiness],
            "liveness": [liveness],
            "duration_sec": [duration_sec],
            "release_year": [release_year],
            "streams": [streams]
        }
        input_df = pd.DataFrame(input_dict)[metadata["feature_cols"]]
        
        try:
            input_proc = preprocessor.transform(input_df)
            clf_sim = models[sim_model_choice]
            pred_class = clf_sim.predict(input_proc)[0]
            pred_prob = clf_sim.predict_proba(input_proc)[0][1] if hasattr(clf_sim, "predict_proba") else (1.0 if pred_class == 1 else 0.0)
            
            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                if pred_class == 1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0052FF 0%, #00D2FF 100%); border-radius: 20px; padding: 28px; color: #ffffff; text-align: center; box-shadow: 0 12px 30px var(--electric-blue-glow);">
                        <div style="font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;">PREDICTION RESULT</div>
                        <div style="font-size: 36px; font-weight: 800; margin: 10px 0;">🔥 VIRAL HIT DETECTED</div>
                        <div style="font-size: 16px;">Virality Probability: <b>{pred_prob*100:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4a154b 0%, #220320 100%); border-radius: 20px; padding: 28px; color: #ffffff; text-align: center; border: 1px solid rgba(255,255,255,0.2);">
                        <div style="font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; color: #f8bbd0;">PREDICTION RESULT</div>
                        <div style="font-size: 32px; font-weight: 800; margin: 10px 0;">STANDARD / NICHE TRACK</div>
                        <div style="font-size: 16px; color: #f8bbd0;">Virality Probability: <b>{pred_prob*100:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with p_col2:
                st.markdown(f"""
                <div class="silver-card">
                    <div style="font-size: 16px; font-weight: 700; color: #4a154b; margin-bottom: 8px;">Key Acoustic Drivers:</div>
                    <ul style="font-size: 13px; color: #2d0a27; line-height: 1.8; margin-bottom: 0;">
                        <li><b>High Danceability ({danceability}):</b> Strong positive correlation with viral playlisting.</li>
                        <li><b>Acousticness ({acousticness}):</b> Low acousticness strongly correlates with viral pop/electronic production.</li>
                        <li><b>Energy ({energy}):</b> Above-average sonic intensity drives user retention.</li>
                        <li><b>Selected Model:</b> {sim_model_choice}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Inference error: {e}")


# =============================================================================
# VIEW 4: DATASET & ARCHITECTURE DOCS
# =============================================================================
elif app_mode == "📖 Dataset & Architecture Docs":
    st.markdown('<div class="section-title"><span class="icon">📖</span> Dataset Architecture & Assignment Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="silver-card">
        <h3 style="color: #4a154b; margin-top: 0;">1. Problem Statement</h3>
        <p style="color: #2d0a27; line-height: 1.6;">
            In the modern streaming era, identifying what sonic characteristics turn a track into a global viral hit is critical for record labels, playlist curators, and independent artists. This project builds and deploys an end-to-end Machine Learning classification pipeline to predict song virality (<code>viral = 1</code> vs <code>viral = 0</code>) using 13 high-dimensional acoustic features.
        </p>
        
        <h3 style="color: #4a154b;">2. Dataset Description</h3>
        <ul style="color: #2d0a27; line-height: 1.7;">
            <li><b>Total Instances:</b> 10,000 tracks (meets requirement of ≥ 500 instances)</li>
            <li><b>Total Features:</b> 13 predictive features (12 numeric acoustic measurements + 1 categorical genre, meets requirement of ≥ 12 features)</li>
            <li><b>Target Variable:</b> <code>viral</code> (Binary: 0 = Non-viral [63.95%], 1 = Viral [36.05%])</li>
            <li><b>Train/Test Split:</b> 80% Train (8,000 instances) / 20% Holdout Test (2,000 instances) stratified by target class</li>
            <li><b>Preprocessing:</b> StandardScaler for numeric features, OneHotEncoder for categorical genre</li>
        </ul>
        
        <h3 style="color: #4a154b;">3. Evaluated Classification Models</h3>
        <ol style="color: #2d0a27; line-height: 1.7;">
            <li><b>Logistic Regression:</b> L2-regularized linear decision boundary classifier.</li>
            <li><b>Decision Tree Classifier:</b> Depth-controlled hierarchical decision tree (max depth = 6).</li>
            <li><b>K-Nearest Neighbor Classifier (KNN):</b> Distance-weighted instance classifier (k = 7).</li>
            <li><b>Naive Bayes Classifier:</b> Gaussian conditional probability density classifier.</li>
            <li><b>Support Vector Machine (SVM):</b> Radial Basis Function (RBF) kernel margin maximizer.</li>
        </ol>
        
        <h3 style="color: #4a154b;">4. Mandatory Evaluation Metrics</h3>
        <ul style="color: #2d0a27; line-height: 1.7;">
            <li><b>Accuracy:</b> Proportion of total correct predictions.</li>
            <li><b>AUC Score (ROC-AUC):</b> Area under the Receiver Operating Characteristic curve.</li>
            <li><b>Precision:</b> True Positives / (True Positives + False Positives).</li>
            <li><b>Recall:</b> True Positives / (True Positives + False Negatives).</li>
            <li><b>F1 Score:</b> Harmonic mean of Precision and Recall.</li>
            <li><b>MCC Score:</b> Matthews Correlation Coefficient, measuring binary correlation between actual and predicted classes even on imbalanced sets.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b1d5c; font-size: 13px; padding: 12px 0;">
    <b>BITS Pilani WILP M.Tech (AIML)</b> • Machine Learning Assignment 2 • Deployed with Streamlit Community Cloud
</div>
""", unsafe_allow_html=True)
