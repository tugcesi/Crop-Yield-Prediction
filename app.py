"""
app.py – Crop Yield Predictor (Streamlit)
Run: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── Artifact paths ───────────────────────────────────────────────────────────
MODEL_PATH    = Path("model.joblib")
FEATURES_PATH = Path("feature_columns.joblib")
ENCODERS_PATH = Path("encoders.joblib")
STATS_PATH    = Path("num_stats.joblib")


# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    paths = [MODEL_PATH, FEATURES_PATH, ENCODERS_PATH, STATS_PATH]
    if not all(p.exists() for p in paths):
        return None, None, None, None
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    encoders        = joblib.load(ENCODERS_PATH)
    num_stats       = joblib.load(STATS_PATH)
    return model, feature_columns, encoders, num_stats


model, feature_columns, encoders, num_stats = load_artifacts()

if model is None:
    st.error(
        "⚠️ Model dosyaları bulunamadı. "
        "Lütfen önce `python save_model.py` komutunu çalıştırın."
    )
    st.stop()

# ── Title ────────────────────────────────────────────────────────────────────
st.title("🌾 Crop Yield Predictor")
st.markdown("Tarım koşullarını girerek **ürün verimini (Yield)** tahmin edin.")
st.markdown("---")

# ── Sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("🌱 Tarım Parametreleri")

user_input: dict = {}

for col in feature_columns:
    if col in encoders:
        # Kategorik → selectbox
        le = encoders[col]
        options = list(le.classes_)
        selected = st.sidebar.selectbox(col, options)
        user_input[col] = selected          # ham değer; encode edilecek
    elif col in num_stats:
        stats = num_stats[col]
        lo    = float(stats["min"])
        hi    = float(stats["max"])
        mean  = float(stats["mean"])
        # Büyük aralıklı sütunlar için number_input, küçükler için slider
        if hi - lo > 1000:
            val = st.sidebar.number_input(
                col, min_value=lo, max_value=hi * 2, value=mean, format="%.2f"
            )
        else:
            val = st.sidebar.slider(
                col, min_value=lo, max_value=hi, value=mean, step=(hi - lo) / 200
            )
        user_input[col] = val
    else:
        # Bilinmeyen sütun – sayısal varsay
        val = st.sidebar.number_input(col, value=0.0)
        user_input[col] = val

predict_btn = st.sidebar.button("🔍 Tahmin Et", use_container_width=True)


# ── Build feature row ────────────────────────────────────────────────────────
def build_input_df() -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        if col in encoders:
            le  = encoders[col]
            val = str(user_input[col])
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = float(le.transform([val])[0])
        else:
            row[col] = float(user_input[col])
    return pd.DataFrame([row])[feature_columns].astype(float)


# ── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    X_input    = build_input_df()
    prediction = float(model.predict(X_input)[0])

    # Sonuç gösterimi
    col1, col2 = st.columns([1, 1])

    with col1:
        st.success(f"🌾 Tahmini Ürün Verimi: **{prediction:,.2f}**")

        # Girdi özeti tablosu
        st.markdown("### 📋 Girilen Değerler")
        summary_df = pd.DataFrame({
            "Özellik": list(user_input.keys()),
            "Değer":   [str(v) for v in user_input.values()],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col2:
        # Gauge chart
        max_gauge = max(prediction * 2, 1.0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"valueformat": ",.1f"},
            title={"text": "Tahmin Edilen Yield"},
            gauge={
                "axis":  {"range": [0, max_gauge]},
                "bar":   {"color": "seagreen"},
                "steps": [
                    {"range": [0,              max_gauge * 0.33], "color": "lightyellow"},
                    {"range": [max_gauge * 0.33, max_gauge * 0.66], "color": "lightgreen"},
                    {"range": [max_gauge * 0.66, max_gauge],        "color": "mediumseagreen"},
                ],
            },
        ))
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature importance
    st.markdown("### 📊 Özellik Önemi")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Özellik": feature_columns, "Önem": importances})
            .sort_values("Önem", ascending=True)
        )
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Önem"],
            y=fi_df["Özellik"],
            orientation="h",
            marker_color="seagreen",
        ))
        fig_fi.update_layout(
            title="Özellik Önemi (LightGBM)",
            xaxis_title="Önem Skoru",
            yaxis_title="Özellik",
            height=max(350, len(feature_columns) * 30),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Bu model için özellik önemi mevcut değil.")

else:
    st.info("👈 Sol panelden tarım parametrelerini girin ve **🔍 Tahmin Et** butonuna tıklayın.")

# ── About ────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    ### 🌾 Crop Yield Predictor

    Bu uygulama, **Kaggle – Crop Yield Prediction Challenge** yarışması için
    geliştirilen makine öğrenmesi modelini kullanarak ürün verimini tahmin eder.

    **Kullanılan Yöntemler:**
    - LightGBM Regressor
    - Label Encoding (kategorik özellikler için)
    - Median / Mode ile eksik değer doldurma

    **Veri Kaynağı:**
    [Kaggle – Crop Yield Prediction Challenge](https://www.kaggle.com/competitions/crop-yield-prediction-challenge)
    """)