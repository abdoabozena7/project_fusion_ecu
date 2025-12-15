import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# Page config (small & clean UI)
# ===============================
st.set_page_config(
    page_title="CTR Prediction",
    layout="centered"
)

st.title("📊 CTR Prediction")
st.caption("Predict click probability for an ad impression")

# ===============================
# Paths
# ===============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "model_stacking.joblib")

# ===============================
# Load data & model
# ===============================
@st.cache_data
def load_feature_template():
    if not os.path.exists(X_TEST_PATH):
        st.error(f"Missing file: {X_TEST_PATH}")
        st.stop()
    return pd.read_csv(X_TEST_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

X_template = load_feature_template()
model = load_model()

# ===============================
# Sidebar info
# ===============================
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("Total features expected:", len(X_template.columns))
    st.write("Model type:", type(model).__name__)
    st.write("Samples available:", len(X_template))

# ===============================
# UI controls
# ===============================
st.subheader("Input data")

col1, col2 = st.columns(2)

with col1:
    use_random = st.button("🎲 Use Random Sample")

with col2:
    use_first = st.button("📌 Use First Sample")

# Initialize session state
if "base_row" not in st.session_state:
    st.session_state.base_row = X_template.head(1).copy()

# Handle buttons
if use_random:
    st.session_state.base_row = X_template.sample(1).copy()

if use_first:
    st.session_state.base_row = X_template.head(1).copy()

base_row = st.session_state.base_row.copy()

# ===============================
# Optional manual overrides (few only)
# ===============================
with st.expander("✏️ Optional: edit few values (advanced)"):
    editable_cols = base_row.columns[:5]  # show only first 5 to keep UI clean
    for col in editable_cols:
        if pd.api.types.is_numeric_dtype(base_row[col]):
            base_row[col] = st.number_input(
                label=col,
                value=float(base_row[col].iloc[0])
            )

# ===============================
# Prediction
# ===============================
st.divider()

if st.button("🚀 Predict CTR", use_container_width=True):
    try:
        # Ensure correct column order
        input_df = base_row[X_template.columns]

        prob = model.predict_proba(input_df)[0, 1]

        st.success(f"✅ Predicted Click Probability: **{prob:.4f}**")

        st.progress(min(prob, 1.0))

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

# ===============================
# Debug (optional)
# ===============================
with st.expander("🧪 Debug info"):
    st.write("Input shape:", base_row.shape)
    st.write("Input columns:", list(base_row.columns))
