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

# ── Page config ───────────────────────────────────────────────���──────────────
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


# ── Artefaktları yükle ───────────────────────────────────────────────────────
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

# ── Başlık ───────────────────────────────────────────────────────────────────
st.title("🌾 Crop Yield Predictor")
st.markdown(
    "Tarım koşullarını girerek **ürün verimini (yield_tpha)** tahmin edin."
)
st.markdown("---")

# ── Sidebar girişleri ────────────────────────────────────────────────────────
st.sidebar.header("🌱 Tarım Parametreleri")

FRIENDLY_LABELS = {
    "soil_ph":              "Toprak pH",
    "soil_moisture":        "Toprak Nemi (%)",
    "avg_temperature":      "Ort. Sıcaklık (°C)",
    "total_rainfall":       "Toplam Yağış (mm)",
    "fertilizer_amount":    "Gübre Miktarı (kg)",
    "pesticide_usage":      "Pestisit Kullanımı",
    "sunlight_hours":       "Güneş Işığı (saat/yıl)",
    "nitrogen_content":     "Azot İçeriği (N)",
    "phosphorus_content":   "Fosfor İçeriği (P)",
    "potassium_content":    "Potasyum İçeriği (K)",
    "irrigation_frequency": "Sulama Sıklığı",
    "crop_type":            "Ürün Türü",
    "region":               "Bölge",
    "season":               "Mevsim",
}

user_input: dict = {}

for col in feature_columns:
    label = FRIENDLY_LABELS.get(col, col)

    if col in encoders:
        # Kategorik → selectbox
        options = list(encoders[col].classes_)
        user_input[col] = st.sidebar.selectbox(label, options)

    elif col in num_stats:
        stats = num_stats[col]
        lo    = float(stats["min"])
        hi    = float(stats["max"])
        mean  = float(stats["mean"])

        # Geniş aralıklı sütunlar için number_input, diğerleri için slider
        if hi - lo > 500:
            user_input[col] = st.sidebar.number_input(
                label,
                min_value=round(lo, 2),
                max_value=round(hi * 1.5, 2),
                value=round(mean, 2),
                step=round((hi - lo) / 200, 2),
                format="%.2f",
            )
        else:
            user_input[col] = st.sidebar.slider(
                label,
                min_value=round(lo, 2),
                max_value=round(hi, 2),
                value=round(mean, 2),
                step=round((hi - lo) / 200, 4),
            )
    else:
        user_input[col] = st.sidebar.number_input(label, value=0.0)

predict_btn = st.sidebar.button("🔍 Tahmin Et", use_container_width=True)


# ── Özellik vektörü oluştur ──────────────────────────────────────────────────
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


# ── Tahmin ───────────────────────────────────────────────────────────────────
if predict_btn:
    X_input    = build_input_df()
    prediction = float(model.predict(X_input)[0])

    left, right = st.columns(2)

    with left:
        st.success(f"### 🌾 Tahmini Verim: **{prediction:,.3f} t/ha**")

        # Girdi özeti
        st.markdown("#### 📋 Girilen Değerler")
        summary_rows = []
        for col in feature_columns:
            summary_rows.append({
                "Özellik": FRIENDLY_LABELS.get(col, col),
                "Değer":   str(user_input[col]),
            })
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        # Gauge chart
        max_gauge = max(prediction * 2, 5.0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"valueformat": ",.2f", "suffix": " t/ha"},
            title={"text": "Tahmin Edilen Verim (t/ha)"},
            gauge={
                "axis":  {"range": [0, max_gauge]},
                "bar":   {"color": "seagreen"},
                "steps": [
                    {"range": [0,                    max_gauge * 0.33], "color": "#fff9c4"},
                    {"range": [max_gauge * 0.33,     max_gauge * 0.66], "color": "#a5d6a7"},
                    {"range": [max_gauge * 0.66,     max_gauge],        "color": "#2e7d32"},
                ],
                "threshold": {
                    "line":  {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": prediction,
                },
            },
        ))
        fig_gauge.update_layout(height=340)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Özellik önemi ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Özellik Önemi")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({
                "Özellik": [FRIENDLY_LABELS.get(c, c) for c in feature_columns],
                "Önem":    importances,
            })
            .sort_values("Önem", ascending=True)
        )
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Önem"],
            y=fi_df["Özellik"],
            orientation="h",
            marker_color="seagreen",
        ))
        fig_fi.update_layout(
            title="LightGBM – Özellik Önemi",
            xaxis_title="Önem Skoru",
            yaxis_title="",
            height=max(380, len(feature_columns) * 32),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Bu model için özellik önemi bilgisi mevcut değil.")

else:
    st.info(
        "👈 Sol panelden tarım parametrelerini girin ve "
        "**🔍 Tahmin Et** butonuna tıklayın."
    )

# ── Hakkında ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    ### 🌾 Crop Yield Predictor

    Bu uygulama **Kaggle – Crop Yield Prediction Challenge** yarışması için
    geliştirilen makine öğrenmesi modelini kullanarak **ton/hektar (t/ha)** 
    cinsinden ürün verimini tahmin eder.

    | Teknik | Açıklama |
    |---|---|
    | **LightGBM Regressor** | Hızlı gradient boosting regresyonu |
    | **Label Encoding** | `crop_type`, `region`, `season` için |
    | **Median Imputation** | Eksik sayısal değerler için |

    **Veri:** [Kaggle – Crop Yield Prediction](https://www.kaggle.com/competitions/crop-yield-prediction-challenge)
    """)