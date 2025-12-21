import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Streamlit App: CTR Prediction (Simple UI)
# - Dropdown to choose a saved model
# - One-click button to fill ALL inputs with random valid data
# - No paths shown in UI (clean)
# ============================================================

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="CTR Prediction", layout="centered")
st.title("CTR Prediction")
st.caption("Predict click probability for an ad impression")

# -------------------------------
# Paths (hidden from UI)
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")

if not os.path.exists(PREPROCESSOR_PATH):
    st.error("Missing preprocessing artifacts (preprocessor.joblib).")
    st.stop()

# -------------------------------
# Helpers
# -------------------------------
def list_model_files(models_dir: str):
    if not os.path.isdir(models_dir):
        return []
    allowed_ext = (".pkl", ".joblib")
    files = [f for f in os.listdir(models_dir) if f.lower().endswith(allowed_ext)]
    files = [f for f in files if os.path.isfile(os.path.join(models_dir, f))]

    def sort_key(name: str):
        n = name.lower()
        score = 100
        if "calibr" in n:
            score -= 30
        if "stack" in n:
            score -= 20
        if "weighted" in n:
            score -= 15
        if "voting" in n:
            score -= 10
        return (score, name)

    return sorted(files, key=sort_key)


def safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X).astype(float)


def transform_input(input_df: pd.DataFrame, preproc: dict) -> pd.DataFrame:
    df = input_df.copy()

    # Drop ID-like columns (same behavior as training)
    id_keywords = ["user", "userid", "nick", "adgroup_id", "campaign_id", "customer", "pid"]
    id_like_cols = [c for c in df.columns if any(k in c for k in id_keywords)]
    if id_like_cols:
        df = df.drop(columns=id_like_cols, errors="ignore")

    numeric_cols = preproc.get("numeric_cols", [])
    high_card_cols = preproc.get("high_card_cols", [])
    low_card_cols = preproc.get("low_card_cols", [])

    enc_mappings = preproc.get("high_card_mappings", {})
    label_mappings = preproc.get("low_card_mappings", {})
    global_mean = float(preproc.get("global_mean", 0.0))

    encoded = df.copy()

    # High-card target encoding
    for col in high_card_cols:
        if col not in encoded.columns:
            encoded[col] = global_mean
            continue
        mapping = enc_mappings.get(col, {}).get("mapping", {})
        encoded[col] = encoded[col].map(mapping).fillna(global_mean).astype(np.float32)

    # Low-card label encoding
    for col in low_card_cols:
        if col not in encoded.columns:
            encoded[col] = -1
            continue
        mapping = label_mappings.get(col, {})
        encoded[col] = encoded[col].map(mapping).fillna(-1).astype(np.int64)

    # Numeric: median impute + standardize
    stats = preproc.get("numeric_imputer_statistics", [])
    means = preproc.get("numeric_scaler_mean", [])
    scales = preproc.get("numeric_scaler_scale", [])

    for i, col in enumerate(numeric_cols):
        if col not in encoded.columns:
            encoded[col] = np.nan

        median = float(stats[i]) if i < len(stats) else 0.0
        mean = float(means[i]) if i < len(means) else 0.0
        scale = float(scales[i]) if i < len(scales) else 1.0
        if scale == 0:
            scale = 1.0

        encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(median).astype(np.float32)
        encoded[col] = (encoded[col] - mean) / scale

    top_features = preproc.get("top_features", [])
    if top_features:
        encoded = encoded.reindex(columns=top_features, fill_value=0)

    for c in encoded.columns:
        encoded[c] = pd.to_numeric(encoded[c], errors="coerce").fillna(0).astype(np.float32)

    return encoded


def random_valid_value_for_col(col: str):
    """
    Generates random *reasonable* values for your CTR inputs.
    These are "valid" in the sense of correct type/range, not necessarily realistic distribution.
    """
    col = col.lower()
    rng = np.random.default_rng()

    if col in ["user", "adgroup_id", "campaign_id", "cate_id", "brand"]:
        return int(rng.integers(1, 500000))
    if col == "price":
        # price can be 0..1000, with bias to smaller values
        return float(np.round(rng.gamma(shape=2.0, scale=30.0), 2))  # typical ~60
    if col == "age_level":
        return int(rng.integers(0, 7))
    if col == "final_gender_code":
        return str(rng.choice(["1", "2"]))
    if col == "shopping_level":
        return int(rng.integers(0, 4))
    if col == "pvalue_level":
        return int(rng.integers(0, 4))

    if col in ["buy", "cart", "fav", "pv"]:
        # counts: mostly small, sometimes larger
        if col == "pv":
            return int(rng.integers(0, 200))
        return int(rng.integers(0, 50))

    if col == "hour":
        return int(rng.integers(0, 24))
    if col == "day":
        return int(rng.integers(0, 8))  # small range, adjust if you want

    # fallback
    return 0


def set_random_inputs():
    # Fill all inputs in session state
    keys = [
        "user", "adgroup_id", "campaign_id", "cate_id", "brand",
        "price", "age_level", "final_gender_code", "shopping_level", "pvalue_level",
        "buy", "cart", "fav", "pv",
        "hour", "day"
    ]
    for k in keys:
        st.session_state[k] = random_valid_value_for_col(k)


# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_preprocessor(path: str):
    return joblib.load(path)

preprocessor = load_preprocessor(PREPROCESSOR_PATH)

model_files = list_model_files(MODELS_DIR)
if not model_files:
    st.error("No saved models found in /models (expected .pkl or .joblib).")
    st.stop()

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

# -------------------------------
# Sidebar: model dropdown (simple)
# -------------------------------
selected_model_file = st.selectbox("Choose model", model_files, index=0)
model = load_model(os.path.join(MODELS_DIR, selected_model_file))

# -------------------------------
# Initialize session state defaults
# -------------------------------
defaults = {
    "user": 0,
    "adgroup_id": 0,
    "campaign_id": 0,
    "cate_id": 0,
    "brand": 0,
    "price": 0.0,
    "age_level": 0,
    "final_gender_code": "1",
    "shopping_level": 0,
    "pvalue_level": 0,
    "buy": 0,
    "cart": 0,
    "fav": 0,
    "pv": 0,
    "hour": 0,
    "day": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------
# One-click random fill
# -------------------------------
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    if st.button("🎲 Fill Random Valid Data", use_container_width=True):
        set_random_inputs()
with btn_col2:
    if st.button("♻️ Reset", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v

st.divider()

# -------------------------------
# Inputs
# -------------------------------
st.subheader("User & Ad")
c1, c2, c3 = st.columns(3)

with c1:
    user = st.number_input("user", min_value=0, step=1, key="user")
    age_level = st.number_input("age_level", min_value=0, step=1, key="age_level")
    final_gender_code = st.selectbox("final_gender_code", ["1", "2"], key="final_gender_code")

with c2:
    adgroup_id = st.number_input("adgroup_id", min_value=0, step=1, key="adgroup_id")
    campaign_id = st.number_input("campaign_id", min_value=0, step=1, key="campaign_id")
    cate_id = st.number_input("cate_id", min_value=0, step=1, key="cate_id")

with c3:
    brand = st.number_input("brand", min_value=0, step=1, key="brand")
    price = st.number_input("price", min_value=0.0, step=1.0, key="price")
    shopping_level = st.number_input("shopping_level", min_value=0, step=1, key="shopping_level")
    pvalue_level = st.number_input("pvalue_level", min_value=0, step=1, key="pvalue_level")

st.subheader("Behavior (Aggregated)")
b1, b2, b3, b4 = st.columns(4)
with b1:
    buy = st.number_input("buy", min_value=0, step=1, key="buy")
with b2:
    cart = st.number_input("cart", min_value=0, step=1, key="cart")
with b3:
    fav = st.number_input("fav", min_value=0, step=1, key="fav")
with b4:
    pv = st.number_input("pv", min_value=0, step=1, key="pv")

with st.expander("Optional time features"):
    hour = st.number_input("hour", min_value=0, max_value=23, step=1, key="hour")
    day = st.number_input("day", min_value=0, step=1, key="day")

# -------------------------------
# Predict
# -------------------------------
st.divider()

if st.button("🚀 Predict CTR", use_container_width=True):
    try:
        raw_input = pd.DataFrame(
            {
                "user": [user],
                "adgroup_id": [adgroup_id],
                "campaign_id": [campaign_id],
                "cate_id": [cate_id],
                "brand": [brand],
                "price": [price],
                "age_level": [age_level],
                "final_gender_code": [final_gender_code],
                "shopping_level": [shopping_level],
                "pvalue_level": [pvalue_level],
                "buy": [buy],
                "cart": [cart],
                "fav": [fav],
                "pv": [pv],
                "hour": [hour],
                "day": [day],
            }
        )

        X_in = transform_input(raw_input, preprocessor)
        prob = float(safe_predict_proba(model, X_in)[0])

        st.success(f"Predicted Click Probability: {prob:.4f}")
        st.progress(min(max(prob, 0.0), 1.0))

    except Exception as e:
        st.error("Prediction failed. Check model/preprocessor compatibility.")
        st.exception(e)

# -------------------------------
# Debug (optional)
# -------------------------------
with st.expander("Debug"):
    st.write("Selected model:", selected_model_file)
    st.write("Model type:", type(model).__name__)
    st.write("Top features:", len(preprocessor.get("top_features", [])))

#adding top 5 features that affected this prediction
    st.subheader("Top 5 Features Affecting Prediction")
    if "top_features" in preprocessor:
        for i, feature in enumerate(preprocessor["top_features"][:5]):
            st.write(f"{i+1}. {feature}")
    else:
        st.write("No feature importance data available.")


#adding section for comparing multiple models
with st.expander("Compare Multiple Models"):
    st.subheader("Model Comparison")
    comparison_data = []
    for model_file in model_files:
        temp_model = load_model(os.path.join(MODELS_DIR, model_file))
        try:
            prob = float(safe_predict_proba(temp_model, X_in)[0])
            comparison_data.append((model_file, prob))
        except Exception as e:
            comparison_data.append((model_file, None))

    comparison_df = pd.DataFrame(comparison_data, columns=["Model", "Predicted CTR"])
    #adding color to the predicted CTR column
    st.dataframe(comparison_df)
