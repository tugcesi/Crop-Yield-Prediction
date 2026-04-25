"""
save_model.py – Crop Yield Regression
Run: python save_model.py
Output: model.joblib, feature_columns.joblib, encoders.joblib, num_stats.joblib
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────────────────────
TARGET    = "yield_tpha"   # ✅ Düzeltildi
ID_COL    = "id"
TRAIN_CSV = "crop_yield_train.csv"

# Modele dahil edilmeyecek sütunlar (id, tarih, alan kodu)
DROP_COLS = {"id", "harvest_date", "field_id"}


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Yüklendi: {df.shape}")
    print("Sütunlar:", df.columns.tolist())
    return df


def get_col_types(df: pd.DataFrame, target: str):
    """Kategorik ve sayısal özellik sütunlarını tespit et."""
    exclude = DROP_COLS | {target}
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    num_cols = [c for c in feature_cols if df[c].dtype != object]
    return feature_cols, cat_cols, num_cols


def main() -> None:
    # ── Veri yükle ───────────────────────────────────────────────────────────
    df = load_and_prepare(TRAIN_CSV)

    feature_cols, cat_cols, num_cols = get_col_types(df, TARGET)

    print(f"\nHedef          : {TARGET}")
    print(f"Kategorik ({len(cat_cols)}): {cat_cols}")
    print(f"Sayısal   ({len(num_cols)}): {num_cols}")
    print(f"Toplam özellik : {len(feature_cols)}")

    # ── Eksik değerler ───────────────────────────────────────────────────────
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Kategorik encoding ───────────────────────────────────────────────────
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── X / y ────────────────────────────────────────────────────────────────
    X = df[feature_cols].astype(float)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Eğitim ───────────────────────────────────────────────────────────────
    print("\nLightGBM eğitiliyor...")
    model = LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"\nRMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # ── Sayısal istatistikler (slider aralıkları için) ───────────────────────
    stats = {
        col: {
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in num_cols
    }

    # ── Kaydet ───────────────────────────────────────────────────────────────
    joblib.dump(model,        "model.joblib")
    joblib.dump(feature_cols, "feature_columns.joblib")
    joblib.dump(encoders,     "encoders.joblib")
    joblib.dump(stats,        "num_stats.joblib")

    print(
        "\n✅ Kaydedildi: model.joblib | feature_columns.joblib | "
        "encoders.joblib | num_stats.joblib"
    )


if __name__ == "__main__":
    main()