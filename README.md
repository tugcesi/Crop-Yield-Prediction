# 🌾 Crop Yield Predictor

Tarım koşullarına göre **ürün verimini (Yield)** tahmin eden makine öğrenmesi uygulaması.

---

## 🚀 Kurulum ve Kullanım

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Modeli eğit (model.joblib üretir)
python save_model.py

# 3. Streamlit uygulamasını başlat
streamlit run app.py
```

---

## 🛠️ Kullanılan Teknikler

| Teknik | Açıklama |
|---|---|
| **LightGBM Regressor** | Hızlı gradient boosting regresyonu |
| **Label Encoding** | Kategorik özellikler için |
| **Median / Mode Imputation** | Eksik değerleri doldurma |

---

## 📁 Proje Yapısı

```
├── crop_yield_train.csv             # Eğitim verisi
├── crop_yield_test.csv              # Test verisi
├── crop-yield-prediction-with-ml.ipynb  # Analiz notebook'u
├── save_model.py                    # Model eğitimi
├── app.py                           # Streamlit uygulaması
├── requirements.txt                 # Bağımlılıklar
├── model.joblib                     # Eğitilmiş model
├── feature_columns.joblib           # Özellik sırası
├── encoders.joblib                  # Kategorik encoder'lar
└── num_stats.joblib                 # Sayısal sütun istatistikleri
```

---

**Veri Kaynağı:** [Kaggle – Crop Yield Prediction Challenge](https://www.kaggle.com/competitions/crop-yield-prediction-challenge)
