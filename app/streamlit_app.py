import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import issparse, hstack, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# CRITICAL: define CTRPreprocessor symbol so joblib can unpickle
# ============================================================
class CTRPreprocessor(BaseEstimator, TransformerMixin):
    """
    Placeholder to satisfy joblib/pickle when the saved object expects
    __main__.CTRPreprocessor.

    NOTE:
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
# Page config
# ============================================================
st.set_page_config(
    page_title="AI Decision Console",
    page_icon="☝️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GLOBAL: cinematic FX layer (particles + cursor magnet + tilt)
# ============================================================
try:
    import streamlit.components.v1 as components
    components.html(
        """
        <style>
          /* FX canvas always on top, but lets clicks pass through */
          #fx-wrap{position:fixed; inset:0; pointer-events:none; z-index:9996;}
          #fx-canvas{width:100%; height:100%; display:block;}
          #cursor-core{
            position:fixed; top:0; left:0; width:14px; height:14px; border-radius:999px;
            background: rgba(255,255,255,.85);
            mix-blend-mode: overlay;
            z-index: 9999; pointer-events:none; transform:translate3d(-100px,-100px,0);
            box-shadow:0 0 24px rgba(255,255,255,.22), 0 0 44px rgba(0,255,225,.14);
          }
          #cursor-halo{
            position:fixed; top:0; left:0; width:240px; height:240px; border-radius:999px;
            background:
              radial-gradient(circle at 30% 35%, rgba(0,255,225,.22), transparent 55%),
              radial-gradient(circle at 70% 40%, rgba(255,0,200,.18), transparent 58%),
              radial-gradient(circle at 55% 70%, rgba(255,170,0,.14), transparent 60%);
            filter: blur(18px) saturate(1.2);
            opacity:.85;
            mix-blend-mode: screen;
            z-index:9998; pointer-events:none; transform:translate3d(-500px,-500px,0);
            transition: transform 40ms linear;
          }
          .spark{
            position:fixed; width:10px; height:10px; border-radius:999px;
            background: rgba(255,255,255,.35);
            mix-blend-mode: overlay;
            z-index:9997; pointer-events:none; opacity:.18;
            transform:translate3d(-100px,-100px,0);
          }
        </style>

        <div id="fx-wrap"><canvas id="fx-canvas"></canvas></div>
        <div id="cursor-halo"></div>
        <div id="cursor-core"></div>

        <script>
        (function(){
          if(window.__AURORA_FX__) return;
          window.__AURORA_FX__ = true;

          // ---------- Cursor ----------
          const root = document.documentElement;
          const core = document.getElementById("cursor-core");
          const halo = document.getElementById("cursor-halo");

          const sparksN = 14;
          const sparks = [];
          for(let i=0;i<sparksN;i++){
            const s = document.createElement("div");
            s.className="spark";
            document.body.appendChild(s);
            sparks.push({el:s, x:0, y:0});
          }

          let mx=innerWidth/2, my=innerHeight/2;
          let hx=mx, hy=my, cx=mx, cy=my;

          addEventListener("mousemove",(e)=>{
            mx=e.clientX; my=e.clientY;
            root.style.setProperty("--mx", mx+"px");
            root.style.setProperty("--my", my+"px");
          }, {passive:true});

          function tickCursor(){
            hx += (mx-hx)*0.08;
            hy += (my-hy)*0.08;
            cx += (mx-cx)*0.24;
            cy += (my-cy)*0.24;

            halo.style.transform = `translate3d(${hx-120}px, ${hy-120}px, 0)`;
            core.style.transform = `translate3d(${cx-7}px, ${cy-7}px, 0)`;

            // spark trail
            let tx=mx, ty=my;
            for(let i=0;i<sparks.length;i++){
              const p = sparks[i];
              p.x += (tx-p.x)*0.22;
              p.y += (ty-p.y)*0.22;
              const s = (sparks.length-i)/sparks.length;
              p.el.style.transform = `translate3d(${p.x-5}px, ${p.y-5}px, 0) scale(${0.5+s*0.9})`;
              p.el.style.opacity = (0.03+s*0.22).toFixed(3);
              tx=p.x; ty=p.y;
            }

            requestAnimationFrame(tickCursor);
          }
          requestAnimationFrame(tickCursor);

          addEventListener("mousedown", ()=>{
            halo.style.filter = "blur(14px) saturate(1.35)";
            core.style.boxShadow = "0 0 24px rgba(255,255,255,.28), 0 0 64px rgba(255,170,0,.18)";
            setTimeout(()=>{
              halo.style.filter = "blur(18px) saturate(1.2)";
              core.style.boxShadow = "0 0 24px rgba(255,255,255,.22), 0 0 44px rgba(0,255,225,.14)";
            }, 160);
          });

          // ---------- Particle field (canvas) ----------
          const c = document.getElementById("fx-canvas");
          const ctx = c.getContext("2d", {alpha:true});
          function resize(){
            c.width = Math.floor(innerWidth * devicePixelRatio);
            c.height = Math.floor(innerHeight * devicePixelRatio);
            c.style.width = innerWidth+"px";
            c.style.height = innerHeight+"px";
            ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
          }
          addEventListener("resize", resize);
          resize();

          const P = [];
          const COUNT = Math.min(120, Math.floor((innerWidth*innerHeight)/18000));
          for(let i=0;i<COUNT;i++){
            P.push({
              x: Math.random()*innerWidth,
              y: Math.random()*innerHeight,
              vx:(Math.random()-.5)*0.35,
              vy:(Math.random()-.5)*0.35,
              r: 0.7 + Math.random()*2.2,
              a: 0.06 + Math.random()*0.16
            });
          }

          function draw(){
            ctx.clearRect(0,0,innerWidth,innerHeight);

            // soft vignetting
            const g = ctx.createRadialGradient(innerWidth*0.5, innerHeight*0.45, 80, innerWidth*0.5, innerHeight*0.45, Math.max(innerWidth, innerHeight)*0.7);
            g.addColorStop(0,"rgba(255,255,255,0.00)");
            g.addColorStop(1,"rgba(0,0,0,0.16)");
            ctx.fillStyle = g;
            ctx.fillRect(0,0,innerWidth,innerHeight);

            // particles
            for(const p of P){
              p.x += p.vx; p.y += p.vy;
              if(p.x< -20) p.x = innerWidth+20;
              if(p.x> innerWidth+20) p.x = -20;
              if(p.y< -20) p.y = innerHeight+20;
              if(p.y> innerHeight+20) p.y = -20;

              // slight attraction to cursor halo center
              const dx = mx - p.x, dy = my - p.y;
              const dist = Math.sqrt(dx*dx+dy*dy)+0.001;
              const pull = Math.max(0, 1 - dist/360) * 0.018;
              p.x += dx/dist * pull;
              p.y += dy/dist * pull;

              ctx.beginPath();
              ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
              ctx.fillStyle = `rgba(255,255,255,${p.a})`;
              ctx.fill();
            }

            // links
            for(let i=0;i<P.length;i++){
              for(let j=i+1;j<P.length;j++){
                const a=P[i], b=P[j];
                const dx=a.x-b.x, dy=a.y-b.y;
                const d2=dx*dx+dy*dy;
                if(d2 < 140*140){
                  const d=Math.sqrt(d2);
                  const alpha = (1 - d/140) * 0.08;
                  ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
                  ctx.lineWidth = 1;
                  ctx.beginPath();
                  ctx.moveTo(a.x,a.y);
                  ctx.lineTo(b.x,b.y);
                  ctx.stroke();
                }
              }
            }
            requestAnimationFrame(draw);
          }
          requestAnimationFrame(draw);

          // ---------- Tilt cards (applies to elements with data-tilt="1") ----------
          function attachTilt(){
            const cards = document.querySelectorAll('[data-tilt="1"]');
            cards.forEach((card)=>{
              if(card.__tiltAttached) return;
              card.__tiltAttached = true;

              card.addEventListener("mousemove",(e)=>{
                const r = card.getBoundingClientRect();
                const px = (e.clientX - r.left) / r.width;
                const py = (e.clientY - r.top) / r.height;
                const rx = (py - 0.5) * -10;
                const ry = (px - 0.5) * 12;
                card.style.transform = `translateY(-2px) rotateX(${rx}deg) rotateY(${ry}deg)`;
              });
              card.addEventListener("mouseleave",()=>{
                card.style.transform = "translateY(0px) rotateX(0deg) rotateY(0deg)";
              });
            });
          }
          // try a few times (Streamlit renders async)
          let tries=0;
          const iv=setInterval(()=>{
            attachTilt();
            tries++;
            if(tries>20) clearInterval(iv);
          }, 350);

        })();
        </script>
        """,
        height=0,
    )
except Exception:
    pass


# ============================================================
# WHOLE APP THEME (COMPLETELY DIFFERENT LAYOUT / SHAPES)
# ============================================================
st.markdown(
    """
<style>
/* ------------------------------------------------------------
   RESET + BASE
------------------------------------------------------------ */
:root{
  --bgA:#05030b;
  --bgB:#060a14;
  --panel: rgba(255,255,255,0.055);
  --panel2: rgba(255,255,255,0.035);
  --stroke: rgba(255,255,255,0.12);
  --stroke2: rgba(255,255,255,0.16);
  --txt: rgba(245,247,255,0.96);
  --muted: rgba(245,247,255,0.70);

  --c1:#00ffe1;   /* aqua */
  --c2:#ff00d4;   /* magenta */
  --c3:#ffb300;   /* amber */
  --c4:#7c5cff;   /* electric violet */
  --c5:#20ff7a;   /* neon green */

  --rXL: 34px;
  --rL:  24px;
  --rM:  18px;
  --rS:  14px;

  --shadow: 0 26px 80px rgba(0,0,0,0.60);
  --shadow2: 0 16px 52px rgba(0,0,0,0.50);
}

.stApp{
  color: var(--txt);
  background:
    radial-gradient(1100px 760px at 15% 12%, rgba(0,255,225,0.14), transparent 60%),
    radial-gradient(900px 650px at 88% 18%, rgba(255,0,212,0.12), transparent 58%),
    radial-gradient(1000px 700px at 60% 88%, rgba(255,179,0,0.10), transparent 60%),
    radial-gradient(800px 560px at 18% 88%, rgba(124,92,255,0.10), transparent 58%),
    linear-gradient(180deg, var(--bgA) 0%, var(--bgB) 55%, var(--bgA) 100%) !important;
}

header, footer{visibility:hidden;}
[data-testid="stSidebar"]{display:none;} /* FULL CONSOLE MODE */

/* ------------------------------------------------------------
   TOP BAR (custom nav)
------------------------------------------------------------ */
.topbar{
  position: relative;
  border-radius: var(--rXL);
  padding: 18px 20px;
  background:
    radial-gradient(900px 500px at 18% 30%, rgba(0,255,225,0.12), transparent 60%),
    radial-gradient(900px 500px at 82% 30%, rgba(255,0,212,0.12), transparent 60%),
    linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border:1px solid var(--stroke);
  box-shadow: var(--shadow);
  overflow:hidden;
}
.topbar:before{
  content:"";
  position:absolute;
  inset:-3px;
  background: conic-gradient(from 200deg,
    rgba(0,255,225,0.30),
    rgba(124,92,255,0.26),
    rgba(255,0,212,0.26),
    rgba(255,179,0,0.20),
    rgba(0,255,225,0.30)
  );
  filter: blur(26px);
  opacity: .32;
  animation: spin 10s linear infinite;
}
@keyframes spin {to{transform:rotate(360deg)}}
.topbar > div{ position:relative; z-index:2; }

.brandline{
  display:flex; align-items:center; justify-content:space-between;
  gap:16px;
}
.brand{
  display:flex; align-items:center; gap:12px;
}
.logo{
  width:42px; height:42px; border-radius:14px;
  background: linear-gradient(135deg, rgba(0,255,225,0.35), rgba(255,0,212,0.25), rgba(255,179,0,0.18));
  border:1px solid rgba(255,255,255,0.16);
  box-shadow: 0 12px 34px rgba(0,0,0,0.45);
}
.titlewrap h1{
  margin:0;
  font-size: 22px;
  letter-spacing: .35px;
}
.titlewrap .sub{
  margin-top:3px;
  color: var(--muted);
  font-size: 12px;
}

.pulsechip{
  display:inline-flex; align-items:center; gap:10px;
  padding: 10px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
}
.pulseDot{
  width:10px; height:10px; border-radius:999px;
  background: var(--c1);
  box-shadow: 0 0 18px rgba(0,255,225,0.35), 0 0 34px rgba(0,255,225,0.18);
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse{
  0%,100%{ transform:scale(1); opacity:.9;}
  50%{ transform:scale(1.35); opacity:.55;}
}

/* ------------------------------------------------------------
   BENTO GRID + CARDS (tilt)
------------------------------------------------------------ */
.bento{
  display:grid;
  grid-template-columns: 1.4fr 0.9fr;
  gap: 14px;
  margin-top: 14px;
}
@media (max-width: 1200px){
  .bento{grid-template-columns: 1fr; }
}
.card{
  border-radius: var(--rL);
  padding: 16px;
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  border:1px solid var(--stroke);
  box-shadow: var(--shadow2);
  backdrop-filter: blur(12px);
  overflow:hidden;
  transform-style: preserve-3d;
  transition: transform 120ms ease, border-color 220ms ease;
}
.card:hover{ border-color: rgba(255,255,255,0.22); }
.card h3{ margin: 0 0 8px 0; }
.card .muted{ color: var(--muted); font-size: 12px; }
.cardGlow{
  position:absolute; inset:-140px; pointer-events:none;
  background:
    radial-gradient(circle at 20% 20%, rgba(0,255,225,0.14), transparent 45%),
    radial-gradient(circle at 80% 30%, rgba(255,0,212,0.12), transparent 48%),
    radial-gradient(circle at 50% 85%, rgba(255,179,0,0.10), transparent 45%);
  filter: blur(18px);
  opacity:.9;
}
.cardBody{ position:relative; z-index:2; }

.kpiGrid{
  display:grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 12px;
}
@media (max-width: 900px){
  .kpiGrid{grid-template-columns: repeat(2, minmax(0,1fr));}
}
.kpi{
  border-radius: 20px;
  padding: 14px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  position: relative;
  overflow:hidden;
}
.kpi:before{
  content:"";
  position:absolute; inset:-120px;
  background:
    radial-gradient(circle at 20% 30%, rgba(124,92,255,0.14), transparent 45%),
    radial-gradient(circle at 80% 35%, rgba(0,255,225,0.12), transparent 45%);
  filter: blur(18px);
  opacity:.8;
  animation: floaty 7.5s ease-in-out infinite;
}
@keyframes floaty{ 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
.kpi > div{ position:relative; z-index:2; }
.kpi .k{ font-size:12px; color: var(--muted); }
.kpi .v{ font-size:22px; font-weight: 900; margin-top:6px; letter-spacing:.2px; }
.kpi .s{ font-size:12px; color: rgba(245,247,255,0.66); margin-top:6px; }

/* ------------------------------------------------------------
   NAV (radio look like pills)
------------------------------------------------------------ */
div[role="radiogroup"] > label{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px;
  border-radius: 999px;
  margin-right: 8px;
}
div[role="radiogroup"] > label:hover{
  border-color: rgba(255,255,255,0.20);
}

/* ------------------------------------------------------------
   INPUTS / BUTTONS (different from your old style)
------------------------------------------------------------ */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.045) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: var(--txt) !important;
  border-radius: 18px !important;
}

.stButton > button{
  width:100%;
  border:0;
  border-radius: 20px;
  padding: 0.95rem 1rem;
  font-weight: 950;
  color: #06111a;
  background:
    linear-gradient(90deg, rgba(0,255,225,1), rgba(124,92,255,1), rgba(255,0,212,1), rgba(255,179,0,1));
  background-size: 220% 100%;
  animation: slideGlow 2.4s ease-in-out infinite;
  box-shadow: 0 18px 56px rgba(0,0,0,0.46);
  position: relative;
  overflow:hidden;
}
@keyframes slideGlow{
  0%{background-position:0% 50%;}
  50%{background-position:100% 50%;}
  100%{background-position:0% 50%;}
}
.stButton > button:before{
  content:"";
  position:absolute; top:-70%; left:-55%;
  width:42%; height:240%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.38), transparent);
  transform: skewX(-16deg);
  animation: sweep 2.1s ease-in-out infinite;
  opacity:.75;
}
@keyframes sweep{
  0%{left:-70%; opacity:0;}
  18%{opacity:.55;}
  60%{left:130%; opacity:.18;}
  100%{left:130%; opacity:0;}
}

/* ------------------------------------------------------------
   TABLE
------------------------------------------------------------ */
[data-testid="stDataFrame"]{
  border-radius: 20px;
  overflow:hidden;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 10px 44px rgba(0,0,0,0.32);
}

/* tiny helpers */
.hr{
  height:1px; background:rgba(255,255,255,0.10); border:none; margin:14px 0;
}
.tag{
  display:inline-flex; align-items:center; gap:8px;
  padding: 7px 10px; border-radius: 999px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 12px; color: rgba(245,247,255,0.84);
}
.tag i{
  width:10px; height:10px; border-radius:999px; display:inline-block;
  background: linear-gradient(90deg, rgba(0,255,225,1), rgba(255,0,212,1));
  box-shadow: 0 0 14px rgba(0,255,225,0.22);
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")

# ============================================================
# Schema
# ============================================================
RAW_FIELDS = [
    "age_level", "final_gender_code", "hour", "price", "pv", "shopping_level",
    "brand", "pvalue_level", "cate_id", "buy", "cart", "fav", "day"
]

# ============================================================
# Utilities (loading / safety)
# ============================================================
def list_model_files():
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".pkl")]
    model_like = sorted([f for f in files if f.lower().endswith("_model.pkl")])
    other = sorted([f for f in files if not f.lower().endswith("_model.pkl")])
    return model_like + other

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource
def load_preprocessor_safe():
    if not os.path.exists(PREPROCESSOR_PATH):
        return None
    try:
        return joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {str(e)}"}

def transformer_ok(prep) -> bool:
    return (prep is not None) and (not isinstance(prep, dict)) and hasattr(prep, "transform") and callable(getattr(prep, "transform"))

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
        pad = csr_matrix((X.shape[0], missing), dtype=X.dtype)
        return hstack([X, pad], format="csr")
    X = np.asarray(X)
    pad = np.zeros((X.shape[0], missing), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

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

def coerce_raw_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RAW_FIELDS:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["final_gender_code"] = out["final_gender_code"].astype(int)
    return out[RAW_FIELDS].copy()

def score_dataset(df: pd.DataFrame, model, preprocessor):
    if "score" in df.columns:
        s = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(0, 1).values.astype(float)
        return s, "Using provided 'score' column (no inference)."

    raw = coerce_raw_fields(df)
    expected = model_expected_n_features(model)

    if transformer_ok(preprocessor):
        X = preprocessor.transform(raw)
        X = ensure_feature_count(X, expected)
        if issparse(X):
            probs, _, _ = predict_with_auto_dense(model, X.tocsr(), None)
            probs = np.asarray(probs, dtype=float).reshape(-1)
        else:
            probs = np.asarray(safe_predict_proba(model, X), dtype=float).reshape(-1)
        probs = np.clip(probs, 0.0, 1.0)
        return probs, "Pipeline: preprocessor.transform → model."
    else:
        X = raw.values.astype(np.float32)
        X = ensure_feature_count(X, expected)
        probs = np.asarray(safe_predict_proba(model, X), dtype=float).reshape(-1)
        probs = np.clip(probs, 0.0, 1.0)
        return probs, "Fallback: raw numeric → model (auto feature align)."

def diversification_select(df: pd.DataFrame, rank_col: str, k: int, group_col: str | None, cap_per_group: float):
    if group_col is None or group_col not in df.columns:
        return df.head(k).copy(), "Diversification: OFF."
    max_per_group = max(1, int(np.floor(k * cap_per_group)))
    chosen, counts = [], {}
    for _, row in df.iterrows():
        g = str(row[group_col])
        if counts.get(g, 0) < max_per_group:
            chosen.append(row)
            counts[g] = counts.get(g, 0) + 1
        if len(chosen) >= k:
            break
    selected = pd.DataFrame(chosen) if chosen else df.head(k).copy()
    return selected, f"Diversification: ON · {group_col} · Max {max_per_group}/{k}."

def simple_reasons(row: pd.Series):
    reasons_pos, reasons_neg = [], []
    def gv(key, default=0):
        try:
            return float(row.get(key, default))
        except Exception:
            return float(default)

    pv = gv("pv"); cart = gv("cart"); fav = gv("fav"); buy = gv("buy")
    price = gv("price"); hour = gv("hour")

    if cart >= 1: reasons_pos.append("Added to cart (strong intent)")
    if fav  >= 1: reasons_pos.append("Saved/liked (return intent)")
    if pv   >= 6: reasons_pos.append("High browsing interest (views)")
    if buy  >= 1: reasons_pos.append("Purchase signal exists")

    if pv <= 1 and cart == 0 and fav == 0: reasons_neg.append("Very low interest signals")
    if price >= 200 and pv <= 2: reasons_neg.append("High price with weak interest")
    if hour in [0,1,2,3,4,5]: reasons_neg.append("Late-hour impression (often lower response)")

    if not reasons_pos and not reasons_neg:
        reasons_pos.append("Moderate signals (mixed)")

    return reasons_pos[:2], reasons_neg[:2]

def verdict_label(avg_selected: float):
    if avg_selected >= 0.75: return "VERY STRONG"
    if avg_selected >= 0.55: return "GOOD"
    if avg_selected >= 0.40: return "BORDERLINE"
    return "WEAK"

def style_to_budget_fraction(style: str):
    return {"Conservative": 0.10, "Balanced": 0.20, "Aggressive": 0.35}.get(style, 0.20)

def risk_text(style: str):
    return {"Conservative":"LOW", "Balanced":"MEDIUM", "Aggressive":"HIGH"}.get(style, "MEDIUM")


# ============================================================
# MODEL + PREPROCESSOR
# ============================================================
model_files = list_model_files()
if not model_files:
    st.error("No .pkl models found in /models.")
    st.stop()

preprocessor = load_preprocessor_safe()

# Topbar (brand + system chip)
st.markdown(
    """
<div class="topbar">
  <div class="brandline">
    <div class="brand">
      <div class="titlewrap">
        <h1 style="color: #ffffff; text-align: center; margin-left: 250px; font-size: 54px;">Ad Spend Decision Tool</h1>
        <div class="sub" style="text-align: center; font-size: 34px; font-weight: 600; background: linear-gradient(90deg, rgba(0,255,225,1), rgba(124,92,255,1), rgba(255,0,212,1), rgba(255,179,0,1)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-top: 5px; margin-left: 250px;   ">
          Make smarter budget decisions with AI-powered scoring
        </div>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Custom NAV (completely different from tabs)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
nav = st.radio(
    " ",
    ["Console", "Rank Lab", "Explain One", "How it Works"],
    horizontal=True,
    label_visibility="collapsed",
)

# CONTROL CARD (right-side feel but in Streamlit we keep it in a bento grid)
left, right = st.columns([1.55, 1.0], gap="large")

with left:
    st.markdown("### Control Deck")
    st.markdown("<div class='muted'>Pick your engine, style, then run the decision.</div>", unsafe_allow_html=True)

    cA, cB = st.columns([1.2, 1.0])
    with cA:
        display_names = [f.replace(".pkl", "") for f in model_files]
        selected_display = st.selectbox("Prediction Engine", display_names, index=0)
    with cB:
        style = st.selectbox("Operating Mode", ["Conservative", "Balanced", "Aggressive"], index=1)

    group_col = None
    cap_share = 0.25 # cap share is the maximum fraction per group within budget

    selected_model_file = f"{selected_display}.pkl"
    model_path = os.path.join(MODELS_DIR, selected_model_file)

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {selected_model_file}")
        st.exception(e)
        st.stop()

    # Status tags row
    prep_state = "Loaded" if transformer_ok(preprocessor) else ("Error (fallback)" if isinstance(preprocessor, dict) else "Missing (fallback)")
    st.markdown(
        f"""
<div style="margin-top:10px">
  <span class="tag"><i></i> Engine: {selected_model_file}</span>
  <span class="tag"><i></i> Mode: {style} / Risk: {risk_text(style)}</span>
  <span class="tag"><i></i> Preprocessor: {prep_state}</span>
</div>
<hr class="hr"/>
""",
        unsafe_allow_html=True
    )

    st.markdown("</div></div>", unsafe_allow_html=True)

with right:
    st.markdown("### Mission Panel")
    st.markdown("<div class='muted'>Upload → Score → Select budget → Export. This panel is the same for every mode.</div>", unsafe_allow_html=True)
    uploaded_master = st.file_uploader("Upload CSV", type=["csv"], key="master_upload")
    st.markdown("</div></div>", unsafe_allow_html=True)


def get_df_or_stop(upl):
    if upl is None:
        return None
    try:
        df = pd.read_csv(upl)
    except Exception as e:
        st.error("Could not read the CSV. Please check format/encoding.")
        st.exception(e)
        return None
    if len(df) == 0:
        st.warning("Dataset is empty.")
        return None
    return df


# ============================================================
# CONSOLE
# ============================================================
if nav == "Console":
    df = get_df_or_stop(uploaded_master)
    if df is None:
        st.info("Upload a CSV to activate the console.")
        st.stop()

    n = len(df)
    default_k = max(1, int(np.ceil(n * style_to_budget_fraction(style))))
    budget_k = st.slider("Budget Selector (Top-K rows)", 1, n, min(default_k, n))

    # WOW: KPI bento (no old metric tiles)
    st.markdown("<div class='card' data-tilt='1'><div class='cardGlow'></div><div class='cardBody'>", unsafe_allow_html=True)
    st.markdown("### Live Overview")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"<div class='kpi'><div><div class='k'>Rows</div><div class='v'>{n:,}</div><div class='s'>Uploaded opportunities</div></div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><div><div class='k'>Budget (Top-K)</div><div class='v'>{budget_k:,}</div><div class='s'>How many you will spend on</div></div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><div><div class='k'>Risk Profile</div><div class='v'>{risk_text(style)}</div><div class='s'>{style} operating mode</div></div></div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    run = st.button("RUN DECISION ", use_container_width=True)

    if run:
        with st.spinner("Scoring with aurora pipeline..."):
            scores, scoring_note = score_dataset(df, model, preprocessor)

            out = df.copy()
            out["score_click"] = np.clip(scores, 0.0, 1.0).astype(float)

            rank_col = "score_click"
            out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce").fillna(0.0)
            out = out.sort_values(rank_col, ascending=False).reset_index(drop=True)

            selected, div_note = diversification_select(out, rank_col=rank_col, k=budget_k, group_col=group_col, cap_per_group=cap_share)
            selected = selected.copy()
            selected["decision"] = "SPEND"
            skipped = out.iloc[len(selected):].copy()
            skipped["decision"] = "SKIP"

            # Reasons (only if raw fields exist)
            if all(c in selected.columns for c in RAW_FIELDS):
                why, watch = [], []
                for _, r in selected.iterrows():
                    pos, neg = simple_reasons(r)
                    why.append("; ".join(pos))
                    watch.append("; ".join(neg))
                selected["why_spend"] = why
                selected["watchouts"] = watch
                skipped["why_spend"] = skipped.get("why_spend", "")
                skipped["watchouts"] = skipped.get("watchouts", "")

            avg_sel = float(selected["score_click"].mean()) if len(selected) else 0.0
            strength = verdict_label(avg_sel)
            select_ratio = len(selected) / max(1, len(out))

            # Results header card (brand new)
            st.markdown("<div class='card' data-tilt='1'><div class='cardGlow'></div><div class='cardBody'>", unsafe_allow_html=True)
            st.markdown("### Decision Output")
            st.markdown(
                f"""
<span class="tag"><i></i> Strength: {strength}</span>
<span class="tag"><i></i> Avg Score: {avg_sel*100:.1f}%</span>
<span class="tag"><i></i> Selected: {len(selected):,} / {len(out):,} ({select_ratio*100:.1f}%)</span>
<span class="tag"><i></i> {scoring_note}</span>
<span class="tag"><i></i> {div_note}</span>
<hr class="hr"/>
""",
                unsafe_allow_html=True
            )
            st.markdown("</div></div>", unsafe_allow_html=True)

            # Column plan (safe)
            show_cols = []
            for c in ["impression_id", "ad_id", "campaign_id", "creative_id"]:
                if c in selected.columns:
                    show_cols.append(c)
            for c in ["score_click", "decision"]:
                if c in selected.columns and c not in show_cols:
                    show_cols.append(c)
            if "why_spend" in selected.columns:
                for c in ["why_spend", "watchouts"]:
                    if c not in show_cols:
                        show_cols.append(c)
            for c in ["pv", "cart", "fav", "price", "hour", "day", "channel", "cate_id", "brand"]:
                if c in selected.columns and c not in show_cols:
                    show_cols.append(c)

            # Spend list
            st.markdown("#### Spend List (Selected)")
            st.dataframe(selected.reindex(columns=show_cols), use_container_width=True, height=380)
            st.download_button(
                "EXPORT • Selected CSV",
                data=selected.reindex(columns=show_cols).to_csv(index=False).encode("utf-8"),
                file_name="aurora_selected_ads.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("#### Skip List (Not Selected)")
            st.dataframe(skipped.reindex(columns=show_cols), use_container_width=True, height=260)
            st.download_button(
                "EXPORT • Skipped CSV",
                data=skipped.reindex(columns=show_cols).to_csv(index=False).encode("utf-8"),
                file_name="aurora_skipped_ads.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# RANK LAB
# ============================================================
elif nav == "Rank Lab":
    df = get_df_or_stop(uploaded_master)
    if df is None:
        st.info("Upload a CSV to open Rank Lab.")
        st.stop()

    scores, scoring_note = score_dataset(df, model, preprocessor)
    out = df.copy()
    out["score_click"] = scores
    out["score_click"] = pd.to_numeric(out["score_click"], errors="coerce").fillna(0.0).clip(0,1)
    out = out.sort_values("score_click", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["percentile"] = (1.0 - (out.index / max(1, len(out)-1))) * 100.0

    st.markdown("<div class='card' data-tilt='1'><div class='cardGlow'></div><div class='cardBody'>", unsafe_allow_html=True)
    st.markdown("### Rank Lab")
    st.markdown(f"<div class='muted'>Sorted view for operators. {scoring_note}</div>", unsafe_allow_html=True)

    topk = st.slider("Show top K", 5, min(500, len(out)), min(60, len(out)))
    st.markdown("</div></div>", unsafe_allow_html=True)

    cols = []
    for c in ["impression_id", "ad_id", "campaign_id", "creative_id"]:
        if c in out.columns:
            cols.append(c)
    for c in ["rank", "percentile", "score_click"]:
        if c in out.columns and c not in cols:
            cols.append(c)
    for c in ["channel", "cate_id", "brand", "pv", "cart", "fav", "price", "hour"]:
        if c in out.columns and c not in cols:
            cols.append(c)

    st.dataframe(out[cols].head(topk), use_container_width=True, height=520)

    if PLOTLY_OK:
        st.markdown("<div class='card' data-tilt='1'><div class='cardGlow'></div><div class='cardBody'>", unsafe_allow_html=True)
        st.markdown("### Distribution Lens")
        fig = px.histogram(out, x="score_click", nbins=34, height=360)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(245,247,255,0.92)",
            xaxis_title="score_click",
            yaxis_title="count",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# EXPLAIN ONE
# ============================================================
elif nav == "Explain One":
    df = get_df_or_stop(uploaded_master)
    if df is None:
        st.info("Upload a CSV to use Explain One.")
        st.stop()

    scores, _ = score_dataset(df, model, preprocessor)
    out = df.copy()
    out["score_click"] = pd.to_numeric(scores, errors="coerce").astype(float)
    out["score_click"] = np.clip(out["score_click"].values, 0.0, 1.0)
    out = out.sort_values("score_click", ascending=False).reset_index(drop=True)

    idx = st.slider("Pick a row (sorted by score)", 0, len(out)-1, 0)
    row = out.iloc[int(idx)]

    k_guess = max(1, int(np.ceil(len(out) * style_to_budget_fraction(style))))
    decision = "SPEND" if idx < k_guess else "SKIP"
    pos, neg = simple_reasons(row)

    st.markdown("<div class='bento'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 0.9], gap="large")

    with c1:
        st.markdown("<div class='card' data-tilt='1'><div class='cardGlow'></div><div class='cardBody'>", unsafe_allow_html=True)
        st.markdown("### Explanation Card")
        st.markdown("<div class='muted'>No ML language. Just what an operator needs to know.</div>", unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            st.markdown(f"<div class='kpi'><div><div class='k'>Decision</div><div class='v'>{decision}</div><div class='s'>based on your mode</div></div></div>", unsafe_allow_html=True)
        with b:
            st.markdown(f"<div class='kpi'><div><div class='k'>Click Score</div><div class='v'>{float(row['score_click'])*100:.1f}%</div><div class='s'>higher = better</div></div></div>", unsafe_allow_html=True)
        with c:
            st.markdown(f"<div class='kpi'><div><div class='k'>Risk</div><div class='v'>{risk_text(style)}</div><div class='s'>{style} mode</div></div></div>", unsafe_allow_html=True)

        st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
        st.markdown("#### Why this makes sense")
        for r in pos:
            st.write(f"- {r}")
        if neg and any(n.strip() for n in neg):
            st.markdown("#### Watch-outs")
            for r in neg:
                if r.strip():
                    st.write(f"- {r}")

        st.markdown("</div></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card' data-tilt='1'><div class='cardGlow'></div><div class='cardBody'>", unsafe_allow_html=True)
        st.markdown("### Raw Signals Snapshot")
        show = {}
        for c in RAW_FIELDS:
            if c in row.index:
                show[c] = row[c]
        for c in ["channel", "cate_id", "brand", "ad_id", "campaign_id", "impression_id", "price"]:
            if c in row.index and c not in show:
                show[c] = row[c]
        st.dataframe(pd.DataFrame([show]), use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HOW IT WORKS
# ============================================================
else:
    st.markdown("### How it Works (Operator Version)")
    st.markdown(
        """
- You upload rows = **opportunities** (someone + context + ad).
- The tool produces a **score_click** (0..1). Higher = more likely to respond.
- You choose **Budget (Top-K)** = how many you can spend on.
- Optional: diversification prevents one group from dominating.
- Explanations are **signal-based** (views/cart/fav/price/time), not ML jargon.
        """
    )
    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    st.markdown("### Safety / Robustness")
    st.markdown(
        """
- If a preprocessor is available, it will be used.
- If not, the system falls back to numeric raw fields and auto-aligns feature count.
- If a model needs dense input, the system automatically converts sparse to dense.
        """
    )
    st.markdown("</div></div>", unsafe_allow_html=True)
