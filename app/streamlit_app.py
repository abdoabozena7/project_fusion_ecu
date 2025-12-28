import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import load_npz, issparse
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# CRITICAL: define CTRPreprocessor symbol so joblib can unpickle
# ============================================================
class CTRPreprocessor(BaseEstimator, TransformerMixin):
    """
    Placeholder to satisfy joblib/pickle when the saved object expects
    __main__.CTRPreprocessor.

    IMPORTANT:
    - If your preprocessor.joblib requires real logic inside this class,
      paste your original training CTRPreprocessor implementation here.
    """
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


# ============================================================
# Optional visuals (Plotly)
# ============================================================
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ============================================================
# Page config (MUST be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="CTR Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Premium CSS (single unified theme)
# ============================================================
st.markdown(
    """
<style>
.stApp{
  background:
    radial-gradient(1200px 700px at 20% 10%, rgba(0,198,255,.22), transparent 60%),
    radial-gradient(900px 600px at 80% 20%, rgba(0,114,255,.18), transparent 55%),
    radial-gradient(1000px 700px at 60% 90%, rgba(180,0,255,.14), transparent 60%),
    linear-gradient(180deg, #071018 0%, #081724 40%, #06111b 100%);
  color:#eef2ff;
}
header, footer{visibility:hidden;}
section[data-testid="stSidebar"]{
  border-right:1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,0.02);
}
h1,h2,h3{letter-spacing:.2px;}
.muted{opacity:.75;}
.smallnote{font-size:12px; opacity:.72;}
hr{border:none;height:1px;background:rgba(255,255,255,0.10);margin:18px 0;}

.hero{
  position:relative; overflow:hidden;
  border-radius:22px;
  padding:26px;
  background:linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
  border:1px solid rgba(255,255,255,.10);
  box-shadow:0 16px 60px rgba(0,0,0,.45);
  margin-bottom: 12px;
}
.hero:before{
  content:"";
  position:absolute; inset:-120px;
  background:conic-gradient(from 180deg,
    rgba(0,198,255,.22),
    rgba(180,0,255,.18),
    rgba(0,114,255,.20),
    rgba(0,198,255,.22)
  );
  filter:blur(20px);
  animation:spin 10s linear infinite;
  opacity:.55;
}
.hero > div{position:relative; z-index:2;}
@keyframes spin{to{transform:rotate(360deg);} }

.badge{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-size:12px;
  background:rgba(0,198,255,.12);
  border:1px solid rgba(0,198,255,.25);
}

.glass{
  border-radius:18px;
  padding:18px;
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  box-shadow:0 12px 40px rgba(0,0,0,0.35);
  backdrop-filter: blur(8px);
}
.shimmer{position:relative; overflow:hidden;}
.shimmer:after{
  content:"";
  position:absolute;
  top:-40%;
  left:-60%;
  width:45%;
  height:180%;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,.08), transparent);
  transform:skewX(-18deg);
  animation:shimmer 3.6s ease-in-out infinite;
  opacity:.75;
}
@keyframes shimmer{
  0%{left:-70%;}
  55%{left:130%;}
  100%{left:130%;}
}

.pill{
  display:inline-block;
  padding:6px 10px;
  margin:4px 6px 0 0;
  border-radius:999px;
  font-size:12px;
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,0.10);
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  color: #eef2ff !important;
  border-radius: 14px !important;
}
label{opacity:.9;}

.stButton > button{
  width:100%;
  border:0;
  border-radius:14px;
  padding:0.85rem 1rem;
  font-weight:900;
  color:#06111b;
  background:linear-gradient(90deg, #00c6ff, #0072ff, #b400ff);
  background-size:200% 100%;
  animation:glowMove 2.2s ease-in-out infinite;
  box-shadow:0 16px 40px rgba(0,114,255,.25);
}
@keyframes glowMove{
  0%{background-position:0% 50%;}
  50%{background-position:100% 50%;}
  100%{background-position:0% 50%;}
}

.neonbar{
  height:12px;border-radius:999px;
  background:rgba(255,255,255,0.10);
  border:1px solid rgba(255,255,255,0.10);
  overflow:hidden;
}
.neonfill{
  height:100%;
  width:0%;
  border-radius:999px;
  background:linear-gradient(90deg, #00c6ff, #0072ff, #b400ff);
  box-shadow:0 0 20px rgba(0,198,255,.35);
  transition:width 700ms ease;
}

.metricGrid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap:14px;
}
.metricTile{
  border-radius:18px;
  padding:16px;
  background:rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.10);
  box-shadow:0 10px 30px rgba(0,0,0,0.35);
  position:relative;
  overflow:hidden;
}
.metricTile:before{
  content:"";
  position:absolute;
  inset:-80px;
  background: radial-gradient(circle at 10% 10%, rgba(0,198,255,.20), transparent 40%),
              radial-gradient(circle at 90% 30%, rgba(180,0,255,.16), transparent 45%),
              radial-gradient(circle at 60% 90%, rgba(0,114,255,.18), transparent 40%);
  filter: blur(16px);
  opacity:.7;
}
.metricTile > div{position:relative; z-index:2;}
.metricLabel{opacity:.75; font-size:12px; margin-bottom:6px;}
.metricValue{font-size:24px; font-weight:900;}
.metricSub{opacity:.70; font-size:12px; margin-top:6px;}

[data-testid="stDataFrame"]{
  border-radius:16px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.10);
}
@media (max-width: 1200px){
  .metricGrid{grid-template-columns: repeat(2, minmax(0,1fr));}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test_processed.npz")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.csv")

FINAL_EVAL_CSV = os.path.join(PROCESSED_DIR, "final_evaluation_results.csv")
FIG_DIR = os.path.join(PROCESSED_DIR, "figures_eval")
ROC_IMG = os.path.join(FIG_DIR, "roc_curves.png")
PR_IMG = os.path.join(FIG_DIR, "pr_curves.png")

BEST_THR_PATH = os.path.join(PROCESSED_DIR, "best_threshold.json")
FEATURE_COUNT_PATH = os.path.join(PROCESSED_DIR, "feature_count.json")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")


# ============================================================
# Raw schema + ranges
# ============================================================
RAW_FIELDS = [
    "age_level", "final_gender_code", "hour", "price", "pv", "shopping_level",
    "brand", "pvalue_level", "cate_id", "buy", "cart", "fav", "day"
]

RANGES = {
    "age_level": (0, 6),
    "final_gender_code": (1, 2),
    "hour": (0, 23),
    "price": (1.0, 500.0),
    "pv": (0, 20),
    "shopping_level": (0, 5),
    "brand": (0, 5000),
    "pvalue_level": (0, 4),
    "cate_id": (0, 3000),
    "buy": (0, 2),
    "cart": (0, 5),
    "fav": (0, 5),
    "day": (0, 6),
}

def clamp_value(k, v):
    lo, hi = RANGES[k]
    if isinstance(lo, float) or isinstance(hi, float):
        return float(np.clip(float(v), float(lo), float(hi)))
    return int(np.clip(int(v), int(lo), int(hi)))

def clamp_payload(payload: dict) -> dict:
    out = dict(payload)
    for k in RAW_FIELDS:
        if k in out:
            out[k] = clamp_value(k, out[k])
    return out


# ============================================================
# Smart autofill (behavior-aware)
# ============================================================
def weighted_choice(rng, values, probs):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return values[int(rng.choice(len(values), p=probs))]

def smart_autofill_payload(seed=None) -> dict:
    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % (2**32))

    pv = int(weighted_choice(
        rng,
        [0,1,2,3,4,5,6,8,10,12,15,18,20],
        [0.04,0.10,0.16,0.18,0.14,0.10,0.08,0.07,0.05,0.03,0.03,0.01,0.01]
    ))

    hour = int(weighted_choice(
        rng, list(range(24)),
        [0.01,0.01,0.01,0.01,0.01,0.02,0.03,0.05,0.07,0.07,0.06,0.06,
         0.06,0.06,0.06,0.06,0.06,0.06,0.07,0.07,0.06,0.04,0.03,0.02]
    ))

    if pv <= 1:
        cart = int(weighted_choice(rng, [0,1], [0.90,0.10]))
        fav  = int(weighted_choice(rng, [0,1], [0.92,0.08]))
    elif pv <= 4:
        cart = int(weighted_choice(rng, [0,1,2], [0.70,0.25,0.05]))
        fav  = int(weighted_choice(rng, [0,1,2], [0.75,0.22,0.03]))
    elif pv <= 8:
        cart = int(weighted_choice(rng, [0,1,2,3], [0.45,0.40,0.12,0.03]))
        fav  = int(weighted_choice(rng, [0,1,2,3], [0.55,0.33,0.10,0.02]))
    else:
        cart = int(weighted_choice(rng, [0,1,2,3,4], [0.25,0.45,0.20,0.08,0.02]))
        fav  = int(weighted_choice(rng, [0,1,2,3,4], [0.35,0.40,0.18,0.06,0.01]))

    buy_prob = 0.01 + 0.02*(cart >= 2) + 0.01*(fav >= 2) + 0.01*(pv >= 8)
    buy_prob = float(np.clip(buy_prob, 0.0, 0.12))
    buy = int(rng.choice([0,1,2], p=[1-buy_prob, buy_prob*0.95, buy_prob*0.05]))

    price = float(weighted_choice(
        rng,
        [9.0, 19.0, 29.0, 49.0, 79.0, 99.0, 149.0, 199.0, 249.0, 299.0, 399.0],
        [0.03,0.08,0.12,0.17,0.13,0.12,0.10,0.10,0.06,0.05,0.04]
    ))

    payload = dict(
        age_level=int(rng.integers(0, 7)),
        final_gender_code=int(rng.choice([1, 2])),
        hour=hour,
        price=price,
        pv=pv,
        shopping_level=int(weighted_choice(rng, [0,1,2,3,4,5], [0.10,0.18,0.30,0.22,0.15,0.05])),
        brand=int(rng.integers(0, 5001)),
        pvalue_level=int(weighted_choice(rng, [0,1,2,3,4], [0.10,0.30,0.35,0.20,0.05])),
        cate_id=int(rng.integers(0, 3001)),
        buy=buy,
        cart=cart,
        fav=fav,
        day=int(rng.integers(0, 7)),
    )
    return clamp_payload(payload)


# ============================================================
# Reliability indicators
# ============================================================
def input_validity_score(payload: dict) -> tuple[int, list[str]]:
    score = 100.0
    notes = []

    for k in RAW_FIELDS:
        if k not in payload:
            continue
        lo, hi = RANGES[k]
        v = payload[k]
        if v < lo or v > hi:
            score -= 35
            notes.append(f"{k} out of range ({lo}..{hi})")

    pv = int(payload.get("pv", 0))
    cart = int(payload.get("cart", 0))
    fav = int(payload.get("fav", 0))
    buy = int(payload.get("buy", 0))
    age_level = int(payload.get("age_level", 0))

    if pv >= 15:
        score -= 8; notes.append("PV too high (rare case)")
    if cart >= 4:
        score -= 10; notes.append("Cart too high (rare case)")
    if fav >= 4:
        score -= 10; notes.append("Fav too high (rare case)")
    if buy >= 2:
        score -= 10; notes.append("Buy too high (rare case)")
    if age_level > 6:
        score -= 35; notes.append("age_level not logical")

    if buy > 0 and cart == 0 and pv <= 1:
        score -= 12; notes.append("Purchase without interest signals (unusual)")
    if fav >= 3 and pv <= 1:
        score -= 12; notes.append("High Fav with low PV (unusual)")

    score = int(np.clip(score, 0, 100))
    if score >= 85:
        notes.insert(0, "Values close to training distribution")
    elif score >= 60:
        notes.insert(0, "Values acceptable but with slight extremities")
    else:
        notes.insert(0, "Values far from training distribution — result may be unreliable")

    return score, notes

def decision_margin(prob: float, thr: float) -> tuple[float, str]:
    m = float(prob - thr)
    am = abs(m)
    if am >= 0.08:
        label = "Strong decision (far from threshold)"
    elif am >= 0.03:
        label = "Moderate decision"
    else:
        label = "Indecisive (very close to threshold)"
    return m, label

def safe_predict_proba_any(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.reshape(-1)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise AttributeError("Model has no predict_proba or decision_function")

@st.cache_data
def cached_sorted_test_probs(model_name_for_cache: str, X_test_sparse, model):
    if X_test_sparse is None:
        return None
    probs = safe_predict_proba_any(model, X_test_sparse)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    probs.sort()
    return probs

def percentile_rank(prob: float, sorted_probs: np.ndarray) -> float:
    if sorted_probs is None or len(sorted_probs) == 0:
        return np.nan
    idx = np.searchsorted(sorted_probs, prob, side="right")
    return float(idx / len(sorted_probs)) * 100.0

def render_reliability_block(prob: float, thr: float, payload: dict, sorted_test_probs=None):
    vscore, vnotes = input_validity_score(payload)
    m, mlabel = decision_margin(prob, thr)
    perc = percentile_rank(prob, sorted_test_probs) if sorted_test_probs is not None else np.nan
    st.markdown("### ✅ Additional Indicators for Prediction Quality")
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Input Validity Score", f"{vscore}/100")
        st.caption("Are the values close to the training data?")
    with c2:
        st.metric("Decision Margin", f"{m:+.3f}")
        st.caption(mlabel)
    with c3:
        if np.isnan(perc):
            st.metric("Audience Percentile", "N/A")
            st.caption("Activate X_test_processed.npz to calculate ranking against the data")
        else:
            st.metric("Audience Percentile", f"{perc:.1f}%")
            st.caption("How does this case rank against the test samples?")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Quick Notes:**")
    for n in vnotes[:6]:
        st.write(f"- {n}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Core utilities
# ============================================================
def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def list_model_files():
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".pkl")]
    model_like = sorted([f for f in files if f.lower().endswith("_model.pkl")])
    other = sorted([f for f in files if not f.lower().endswith("_model.pkl")])
    return model_like + other

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_preprocessor_safe():
    if not os.path.exists(PREPROCESSOR_PATH):
        return None
    try:
        return joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {str(e)}"}

@st.cache_resource
def load_test_data():
    if not os.path.exists(X_TEST_PATH) or not os.path.exists(Y_TEST_PATH):
        return None, None
    X = load_npz(X_TEST_PATH).tocsr()
    y = pd.read_csv(Y_TEST_PATH).squeeze()
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    return X, y

@st.cache_data
def load_eval_df(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.reshape(-1)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise AttributeError(f"{model.__class__.__name__} has no predict_proba/decision_function")

def model_expected_n_features(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    for attr in ["final_estimator_", "estimator_", "base_estimator_"]:
        if hasattr(model, attr):
            est = getattr(model, attr)
            if hasattr(est, "n_features_in_"):
                return int(est.n_features_in_)
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        try:
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "n_features_in_"):
                return int(base.n_features_in_)
        except Exception:
            pass
    return None

def predict_with_auto_dense(model, X_sparse, X_dense_cache=None):
    try:
        p = safe_predict_proba(model, X_sparse)
        return p, False, X_dense_cache
    except Exception as e:
        msg = str(e).lower()
        dense_required = ("dense data is required" in msg) or ("sparse data was passed" in msg and "dense" in msg)
        if dense_required:
            if X_dense_cache is None:
                X_dense_cache = X_sparse.toarray().astype(np.float32)
            p = safe_predict_proba(model, X_dense_cache)
            return p, True, X_dense_cache
        raise

def ensure_feature_count(X, n_expected: int | None):
    if n_expected is None:
        return X
    n_got = X.shape[1]
    if n_got == n_expected:
        return X
    if n_got > n_expected:
        return X[:, :n_expected]
    missing = n_expected - n_got
    if issparse(X):
        from scipy import sparse as sp
        pad = sp.csr_matrix((X.shape[0], missing), dtype=X.dtype)
        return sp.hstack([X, pad], format="csr")
    X = np.asarray(X)
    pad = np.zeros((X.shape[0], missing), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def decision_text(prob, thr):
    return "CLICK (1)" if prob >= thr else "NO-CLICK (0)"

def confidence_text(prob):
    if prob >= 0.90: return "Very High"
    if prob >= 0.75: return "High"
    if prob >= 0.60: return "Medium"
    return "Low"

def segment_text(prob):
    if prob >= 0.92: return "Elite Audience Match"
    if prob >= 0.80: return "Strong Match"
    if prob >= 0.65: return "Moderate Match"
    return "Needs Optimization"

def render_results(p: float, thr: float, n_expected: int | None, n_used: int, model_name: str, mode_label: str):
    conf = confidence_text(p)
    seg = segment_text(p)
    decision = decision_text(p, thr)

    expected_clicks_per_1k = p * 1000.0
    risk = float(np.clip(1.0 - p, 0.01, 0.99))
    baseline = 0.035
    lift_vs_baseline = (p - baseline) / baseline if baseline > 0 else 0.0

    st.markdown('<div class="glass shimmer">', unsafe_allow_html=True)
    st.markdown(f"### ✅ Prediction ({mode_label})")

    st.markdown(
        f"""
        <div class="metricGrid">
          <div class="metricTile"><div>
            <div class="metricLabel">Predicted CTR</div>
            <div class="metricValue">{p*100:.2f}%</div>
            <div class="metricSub">Confidence: {conf}</div>
          </div></div>

          <div class="metricTile"><div>
            <div class="metricLabel">Decision</div>
            <div class="metricValue">{decision}</div>
            <div class="metricSub">Threshold: {thr:.3f}</div>
          </div></div>

          <div class="metricTile"><div>
            <div class="metricLabel">Expected Clicks / 1K</div>
            <div class="metricValue">{expected_clicks_per_1k:,.0f}</div>
            <div class="metricSub">Segment: {seg}</div>
          </div></div>

          <div class="metricTile"><div>
            <div class="metricLabel">Lift vs Baseline</div>
            <div class="metricValue">{lift_vs_baseline:+.0%}</div>
            <div class="metricSub">Risk: {risk:.2f}</div>
          </div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="margin-top:14px;">
          <div class="neonbar">
            <div class="neonfill" style="width:{p*100:.2f}%"></div>
          </div>
          <div class="muted" style="margin-top:8px;">
            Features used: {n_used}{f" / expected: {n_expected}" if n_expected is not None else ""} · Model: {model_name}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

def plot_eval_dashboard(df: pd.DataFrame):
    st.markdown("## 📊 Evaluation Dashboard")

    if df is None or df.empty:
        st.warning("final_evaluation_results.csv not found (or empty). Generate it from your evaluation step.")
        return

    st.markdown('<div class="glass shimmer">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("If you saved images like roc_curves.png / pr_curves.png, put them in data/processed/figures_eval/")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(ROC_IMG):
            st.image(ROC_IMG, caption="ROC Curves", use_container_width=True)
        else:
            st.info("ROC image not found (roc_curves.png).")
    with c2:
        if os.path.exists(PR_IMG):
            st.image(PR_IMG, caption="Precision–Recall Curves", use_container_width=True)
        else:
            st.info("PR image not found (pr_curves.png).")

    if not PLOTLY_OK:
        return

    candidates = [c for c in ["ROC_AUC", "PR_AUC", "LogLoss", "F1_at_best", "BestThr_F1"] if c in df.columns]
    if not candidates:
        return

    st.markdown("### 📈 Interactive Metric Comparison")
    metric = st.selectbox("Metric", candidates, index=0)
    dplot = df.copy()
    dplot[metric] = pd.to_numeric(dplot[metric], errors="coerce")
    dplot = dplot.dropna(subset=[metric])
    if dplot.empty:
        return

    model_col = "Model" if "Model" in dplot.columns else dplot.columns[0]
    asc = True if metric.lower() == "logloss" else False
    dplot = dplot.sort_values(metric, ascending=asc)

    fig = px.bar(dplot, x=model_col, y=metric, height=420)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eef2ff",
        xaxis_title="Model",
        yaxis_title=metric,
    )
    st.plotly_chart(fig, use_container_width=True)

def logical_defaults():
    return dict(
        age_level=2, final_gender_code=1, hour=13, price=49.0, pv=6,
        shopping_level=2, brand=2700, pvalue_level=1, cate_id=428,
        buy=0, cart=1, fav=0, day=3
    )

def random_safe_values(seed=None):
    rng = np.random.default_rng(seed)
    return dict(
        age_level=int(rng.integers(0, 7)),
        final_gender_code=int(rng.choice([1, 2])),
        hour=int(rng.integers(0, 24)),
        price=float(np.round(rng.uniform(5, 350), 2)),
        pv=int(rng.integers(0, 20)),
        shopping_level=int(rng.integers(0, 6)),
        brand=int(rng.integers(0, 5001)),
        pvalue_level=int(rng.integers(0, 5)),
        cate_id=int(rng.integers(0, 3001)),
        buy=int(rng.integers(0, 3)),
        cart=int(rng.integers(0, 6)),
        fav=int(rng.integers(0, 6)),
        day=int(rng.integers(0, 7)),
    )

def raw_to_payload_df(rp: dict) -> pd.DataFrame:
    out = {k: rp.get(k, 0) for k in RAW_FIELDS}
    for k in out:
        out[k] = pd.to_numeric(out[k], errors="coerce")
        if pd.isna(out[k]):
            out[k] = 0
    out["final_gender_code"] = int(out["final_gender_code"])
    return pd.DataFrame([out])

def transformer_ok(prep) -> bool:
    return (prep is not None) and (not isinstance(prep, dict)) and hasattr(prep, "transform") and callable(getattr(prep, "transform"))

def run_inference(model, preprocessor, payload_df: pd.DataFrame, n_features_from_meta: int | None):
    ok = transformer_ok(preprocessor)
    expected = model_expected_n_features(model) or n_features_from_meta

    if ok:
        X = preprocessor.transform(payload_df)
        X = ensure_feature_count(X, expected)
        if issparse(X):
            probas, _, _ = predict_with_auto_dense(model, X.tocsr(), None)
            prob = clamp01(float(probas[0]))
        else:
            prob = clamp01(float(safe_predict_proba(model, X)[0]))
        return prob, expected, X.shape[1], True

    X = payload_df[RAW_FIELDS].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    X = X.values.astype(np.float32)
    X = ensure_feature_count(X, expected)
    prob = clamp01(float(safe_predict_proba(model, X)[0]))
    return prob, expected, X.shape[1], False


# ============================================================
# REAL stats helpers (per-model)
# ============================================================
def _init_real_stats():
    if "real_stats_by_model" not in st.session_state:
        st.session_state.real_stats_by_model = {}

def get_stats(model_key: str) -> dict:
    _init_real_stats()
    if model_key not in st.session_state.real_stats_by_model:
        st.session_state.real_stats_by_model[model_key] = {"total": 0, "correct": 0, "wrong": 0}
    return st.session_state.real_stats_by_model[model_key]

def reset_stats(model_key: str):
    _init_real_stats()
    st.session_state.real_stats_by_model[model_key] = {"total": 0, "correct": 0, "wrong": 0}

def render_real_result(prob: float, thr: float, model_name: str, true_label: int):
    pred = 1 if prob >= thr else 0
    st.markdown('<div class="glass shimmer">', unsafe_allow_html=True)
    st.markdown("### ✅ REAL Row Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted CTR", f"{prob*100:.2f}%")
    c2.metric("Threshold", f"{thr:.3f}")
    c3.metric("Pred", str(pred))
    c4.metric("True", str(true_label))
    st.caption(f"Model: {model_name} · Decision: {decision_text(prob, thr)}")
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Load artifacts
# ============================================================
model_files = list_model_files()
if not model_files:
    st.error("No .pkl models found in /models.")
    st.stop()

preprocessor = load_preprocessor_safe()
X_test_sparse, y_test = load_test_data()
eval_df = load_eval_df(FINAL_EVAL_CSV)

best_thr_obj = load_json(BEST_THR_PATH, default={})
feature_count_obj = load_json(FEATURE_COUNT_PATH, default={})

n_features_from_meta = None
if isinstance(feature_count_obj, dict):
    try:
        n_features_from_meta = int(feature_count_obj.get("n_features", None))
    except Exception:
        n_features_from_meta = None

thr_map = {}
single_thr = None
single_thr_model = None
if isinstance(best_thr_obj, dict):
    if "best_threshold_f1" in best_thr_obj:
        single_thr = best_thr_obj.get("best_threshold_f1", None)
        single_thr_model = best_thr_obj.get("model", None)
    else:
        thr_map = best_thr_obj

# session state
if "raw_payload" not in st.session_state:
    st.session_state.raw_payload = logical_defaults()
if "test_row_idx" not in st.session_state:
    st.session_state.test_row_idx = 0


# ============================================================
# Header
# ============================================================
st.markdown(
    """
<div class="hero">
  <div>
    <span class="badge"> Premium UI • All Tabs • No Crashes • Raw + REAL Verification</span>
    <h1 style="margin:10px 0 6px 0;">CTR Intelligence Platform</h1>
    <div class="muted">Unified UI and safe inference. REAL tab remains the ground-truth verifier.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    display_names = [f.replace(".pkl", "") for f in model_files]
    selected_display = st.selectbox("Model", display_names, index=0)
    st.session_state.selected_model_display = selected_display

    selected_model_file = f"{selected_display}.pkl"
    model_path = os.path.join(MODELS_DIR, selected_model_file)

    st.markdown("---")
    st.markdown("#### Decision Threshold")
    use_best_thr = st.toggle("Use BestThr_F1 (from evaluation)", value=True)

    selected_thr = 0.50
    if use_best_thr:
        if selected_display in thr_map:
            selected_thr = float(thr_map[selected_display])
            st.caption(f"Loaded from best_threshold.json ({selected_display}) → {selected_thr:.3f}")
        elif isinstance(single_thr, (float, int)):
            selected_thr = float(single_thr)
            st.caption(f"Loaded from best_threshold.json ({single_thr_model}) → {selected_thr:.3f}")
        else:
            st.caption("best_threshold.json missing → default 0.50")
            selected_thr = 0.50
    else:
        selected_thr = st.slider("Manual threshold", 0.05, 0.95, 0.50, 0.01)

    st.markdown("---")
    st.markdown("#### Smart Input Tools")
    cA, cB, cC = st.columns(3)
    if cA.button("Defaults"):
        st.session_state.raw_payload = logical_defaults()
        st.rerun()
    if cB.button("Random Safe"):
        st.session_state.raw_payload = random_safe_values(seed=int(time.time()))
        st.rerun()
    if cC.button("Smart Fill"):
        st.session_state.raw_payload = smart_autofill_payload()
        st.rerun()

    st.markdown("---")
    show_debug = st.toggle("Show Debug Panel", value=False)

    st.markdown("---")
    st.markdown("#### System Status")
    st.success("UI Online")

    if isinstance(preprocessor, dict) and "__error__" in preprocessor:
        st.error("Preprocessor: load error (fallback enabled)")
        st.caption(preprocessor["__error__"][:180])
    elif transformer_ok(preprocessor):
        st.success("Preprocessor: loaded (raw inference enabled)")
    else:
        st.warning("Preprocessor: missing (fallback enabled)")

    if X_test_sparse is not None:
        st.success(f"REAL Test Set Ready (n={X_test_sparse.shape[0]}, d={X_test_sparse.shape[1]})")
    else:
        st.error("REAL Test Set Missing")


# load selected model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {selected_model_file}")
    st.exception(e)
    st.stop()


# sorted test probs for percentile (per selected sidebar model)
sorted_test_probs = None
if X_test_sparse is not None:
    try:
        sorted_test_probs = cached_sorted_test_probs(selected_model_file, X_test_sparse, model)
    except Exception:
        sorted_test_probs = None


# ============================================================
# Tabs
# ============================================================
tabs = st.tabs([
    "🚀 Prediction (Raw)",
    "🧠 Scenario Builder",
    "🧪 Batch Scoring",
    "📊 Evaluation Dashboard",
    "🧾 Test Row Playground (REAL)",
])


# ------------------------------------------------------------
# TAB 1: Prediction (Raw)
# ------------------------------------------------------------
with tabs[0]:
    st.markdown("## 🚀 Prediction & Insights")

    raw_ready = transformer_ok(preprocessor)
    if raw_ready:
        st.markdown('<div class="smallnote">Mode: <b>True raw inference</b> (raw → preprocessor.transform → model)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="smallnote">Mode: <b>Fallback</b> (no preprocessor). App will not crash; REAL tab is the trustworthy verifier.</div>', unsafe_allow_html=True)

    rp = st.session_state.raw_payload

    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.markdown('<div class="glass shimmer">', unsafe_allow_html=True)
        st.markdown("### 🎛 Controls")

        c1, c2, c3 = st.columns(3)
        with c1:
            rp["age_level"] = st.number_input("Age Level", min_value=0, value=int(rp["age_level"]), step=1)
            rp["final_gender_code"] = int(st.selectbox("Gender Code", [1, 2], index=0 if int(rp["final_gender_code"]) == 1 else 1))
            rp["hour"] = st.slider("Hour", 0, 23, int(rp["hour"]))
        with c2:
            rp["price"] = st.number_input("Price", min_value=0.0, value=float(rp["price"]), step=1.0)
            rp["pv"] = st.number_input("Page Views", min_value=0, value=int(rp["pv"]), step=1)
            rp["shopping_level"] = st.number_input("Shopping Level", min_value=0, value=int(rp["shopping_level"]), step=1)
        with c3:
            rp["brand"] = st.number_input("Brand ID", min_value=0, value=int(rp["brand"]), step=1)
            rp["pvalue_level"] = st.number_input("Value Level", min_value=0, value=int(rp["pvalue_level"]), step=1)
            rp["cate_id"] = st.number_input("Category ID", min_value=0, value=int(rp["cate_id"]), step=1)

        st.markdown("<hr/>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            rp["buy"] = st.number_input("Buy", min_value=0, value=int(rp["buy"]), step=1)
        with b2:
            rp["cart"] = st.number_input("Cart", min_value=0, value=int(rp["cart"]), step=1)
        with b3:
            rp["fav"] = st.number_input("Fav", min_value=0, value=int(rp["fav"]), step=1)

        rp["day"] = st.slider("Day (0=Mon ... 6=Sun)", 0, 6, int(rp["day"]))
        st.session_state.raw_payload = rp

        run_raw = st.button("Run CTR Prediction", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass shimmer">', unsafe_allow_html=True)
        st.markdown("### 👁️ Live Snapshot")
        st.markdown(
            f"""
            <span class="pill">Model: {selected_model_file}</span>
            <span class="pill">Threshold: {selected_thr:.3f}</span>
            <span class="pill">Hour: {rp['hour']}</span>
            <span class="pill">PV: {rp['pv']}</span>
            <span class="pill">Price: {float(rp['price']):.2f}</span>
            <span class="pill">Gender: {rp['final_gender_code']}</span>
            <span class="pill">Age: {rp['age_level']}</span>
            <span class="pill">Brand: {rp['brand']}</span>
            <span class="pill">Category: {rp['cate_id']}</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if run_raw:
        payload_df = raw_to_payload_df(rp)
        with st.spinner("Running inference..."):
            prog = st.progress(0)
            for i in range(1, 101, 10):
                time.sleep(0.01)
                prog.progress(i)
            time.sleep(0.05)

            try:
                prob, expected, used, used_transformer = run_inference(
                    model=model,
                    preprocessor=preprocessor,
                    payload_df=payload_df,
                    n_features_from_meta=n_features_from_meta,
                )
                render_results(
                    p=prob,
                    thr=float(selected_thr),
                    n_expected=expected,
                    n_used=used,
                    model_name=selected_model_file,
                    mode_label="Real Raw Inference" if used_transformer else "Fallback (No Preprocessor)",
                )
                render_reliability_block(prob, float(selected_thr), rp, sorted_test_probs=sorted_test_probs)
            except Exception as e:
                st.exception(e)


# ------------------------------------------------------------
# TAB 2: Scenario Builder
# ------------------------------------------------------------
with tabs[1]:
    st.markdown("## 🧠 Scenario Builder")
    st.markdown('<div class="smallnote">Compare scenario variants using the same model.</div>', unsafe_allow_html=True)

    scen = st.selectbox(
        "Scenario",
        [
            "Baseline (Current Inputs)",
            "High Engagement (PV↑)",
            "Discounted Price (Price↓)",
            "Peak Hours (Hour=20)",
            "Cart/Fav Signals (Cart/Fav↑)",
        ],
        index=0,
    )

    base = dict(st.session_state.raw_payload)
    scen_vals = dict(base)
    if scen == "High Engagement (PV↑)":
        scen_vals["pv"] = max(int(scen_vals["pv"]), 12)
    elif scen == "Discounted Price (Price↓)":
        scen_vals["price"] = min(float(scen_vals["price"]), 49.0)
    elif scen == "Peak Hours (Hour=20)":
        scen_vals["hour"] = 20
    elif scen == "Cart/Fav Signals (Cart/Fav↑)":
        scen_vals["cart"] = max(int(scen_vals["cart"]), 3)
        scen_vals["fav"] = max(int(scen_vals["fav"]), 2)

    st.caption("Scenario payload preview")
    st.dataframe(pd.DataFrame([scen_vals])[RAW_FIELDS], use_container_width=True)

    if st.button("Score Scenario", use_container_width=True):
        payload_df = raw_to_payload_df(scen_vals)
        try:
            prob, expected, used, used_transformer = run_inference(
                model=model,
                preprocessor=preprocessor,
                payload_df=payload_df,
                n_features_from_meta=n_features_from_meta,
            )
            render_results(
                p=prob,
                thr=float(selected_thr),
                n_expected=expected,
                n_used=used,
                model_name=selected_model_file,
                mode_label=f"Scenario: {scen}" + ("" if used_transformer else " (Fallback)"),
            )
            render_reliability_block(prob, float(selected_thr), scen_vals, sorted_test_probs=sorted_test_probs)
        except Exception as e:
            st.exception(e)


# ------------------------------------------------------------
# TAB 3: Batch Scoring
# ------------------------------------------------------------
with tabs[2]:
    st.markdown("## 🧪 Batch Scoring")
    st.markdown('<div class="smallnote">Upload CSV with RAW_FIELDS columns. Runs transformer if available; otherwise fallback.</div>', unsafe_allow_html=True)
    st.caption("Required columns:")
    st.code(", ".join(RAW_FIELDS))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to score.")
    else:
        df = pd.read_csv(uploaded)
        missing = [c for c in RAW_FIELDS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Run Batch Scoring", use_container_width=True):
                try:
                    data = df[RAW_FIELDS].copy()
                    for c in RAW_FIELDS:
                        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

                    expected = model_expected_n_features(model) or n_features_from_meta

                    if transformer_ok(preprocessor):
                        X = preprocessor.transform(data)
                    else:
                        X = data.values.astype(np.float32)

                    X = ensure_feature_count(X, expected)

                    if issparse(X):
                        probs, _, _ = predict_with_auto_dense(model, X.tocsr(), None)
                    else:
                        probs = safe_predict_proba(model, X)

                    probs = np.clip(np.asarray(probs).reshape(-1), 0.0, 1.0)

                    out = df.copy()
                    out["pred_ctr"] = probs
                    out["decision"] = (out["pred_ctr"] >= float(selected_thr)).astype(int)

                    st.markdown("### ✅ Batch Results")
                    st.dataframe(out.head(250), use_container_width=True)

                    st.download_button(
                        "Download scored CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="batch_scored.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.exception(e)


# ------------------------------------------------------------
# TAB 4: Evaluation Dashboard
# ------------------------------------------------------------
with tabs[3]:
    plot_eval_dashboard(eval_df)


# ------------------------------------------------------------
# TAB 5: Test Row Playground (REAL)
# ------------------------------------------------------------
with tabs[4]:
    st.markdown("## 🧾 Test Row Playground (REAL)")
    st.markdown('<div class="smallnote">Uses processed X_test row + ground-truth y_test. Independent model switch here.</div>', unsafe_allow_html=True)

    if X_test_sparse is None or y_test is None:
        st.error("Missing X_test_processed.npz / y_test.csv.")
    else:
        # Build REAL names from filenames, but do NOT trust session_state blindly
        real_display_names = []
        file_by_display = {}
        for f in model_files:
            name = f.replace("_model.pkl", "").replace(".pkl", "")
            real_display_names.append(name)
            file_by_display[name] = f

        # --- FIX: sanitize session_state so .index() never crashes ---
        default_real_model = st.session_state.get("selected_model_display", real_display_names[0])

        if ("real_model_display" not in st.session_state) or (st.session_state.real_model_display not in real_display_names):
            # also normalize older stored values like "Bagging_LR_model"
            cand = str(st.session_state.get("real_model_display", default_real_model))
            cand = cand.replace("_model", "")
            st.session_state.real_model_display = cand if cand in real_display_names else default_real_model

        real_selected_name = st.selectbox(
            "REAL Tab Model",
            real_display_names,
            index=real_display_names.index(st.session_state.real_model_display),
            key="real_model_select",
        )
        st.session_state.real_model_display = real_selected_name

        real_model_file = file_by_display[real_selected_name]
        real_model = load_model(os.path.join(MODELS_DIR, real_model_file))

        left, mid, right = st.columns([1, 1, 1])
        with left:
            if st.button("Pick Random Test Row", use_container_width=True):
                st.session_state.test_row_idx = int(np.random.default_rng(int(time.time())).integers(0, X_test_sparse.shape[0]))
        with mid:
            if st.button("Reset Counters (this model)", use_container_width=True):
                reset_stats(real_model_file)
        with right:
            if st.button("Reset ALL Counters", use_container_width=True):
                st.session_state.real_stats_by_model = {}

        idx = st.number_input(
            "Test row index",
            min_value=0,
            max_value=int(X_test_sparse.shape[0] - 1),
            value=int(st.session_state.test_row_idx),
            step=1
        )
        st.session_state.test_row_idx = int(idx)

        true_label = int(y_test.iloc[int(idx)])
        st.markdown(f"### True Label: **{true_label}**")

        stats = get_stats(real_model_file)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Total", stats["total"])
        cB.metric("Correct", stats["correct"])
        cC.metric("Wrong", stats["wrong"])
        acc = (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
        cD.metric("Accuracy (sampled)", f"{acc:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Predict & Compare (REAL)", use_container_width=True):
            try:
                X_one = X_test_sparse[int(idx):int(idx)+1].tocsr()
                probas, _, _ = predict_with_auto_dense(real_model, X_one, None)
                prob = float(np.clip(probas[0], 0.0, 1.0))
                thr = float(selected_thr)  # Use sidebar threshold
                y_pred = 1 if prob >= thr else 0

                stats["total"] += 1
                if y_pred == true_label:
                    stats["correct"] += 1
                else:
                    stats["wrong"] += 1

                render_real_result(prob, thr, real_model_file, true_label)
                st.markdown(
                    f"**True:** `{true_label}` | **Pred prob:** `{prob:.4f}` | **Pred:** `{y_pred}` | **Decision:** `{decision_text(prob, thr)}`"
                )
            except Exception as e:
                st.exception(e)


# ============================================================
# Debug Panel
# ============================================================
if show_debug:
    st.markdown("## 🔧 Debug Panel")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("PROJECT_ROOT:", PROJECT_ROOT)
    st.write("MODELS_DIR:", MODELS_DIR)
    st.write("PROCESSED_DIR:", PROCESSED_DIR)

    st.write("PREPROCESSOR_PATH:", PREPROCESSOR_PATH)
    st.write("Preprocessor loaded:", preprocessor is not None)
    st.write("Raw ready (has transform):", transformer_ok(preprocessor))

    st.write("Selected model:", selected_model_file)

    if X_test_sparse is not None:
        st.write("X_test shape:", X_test_sparse.shape, "| sparse:", True)
    if y_test is not None:
        st.write("y_test distribution:", y_test.value_counts(normalize=True).to_dict())
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown(
    """
    <div style="text-align:center; opacity:0.65; margin:36px 0 10px 0;">
      CTR Intelligence Platform · Unified UI · Raw + REAL Verification · No Crashes
    </div>
    """,
    unsafe_allow_html=True,
)
