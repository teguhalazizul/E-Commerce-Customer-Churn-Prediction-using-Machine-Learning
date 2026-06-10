# 🛒 Prediksi Churn Pelanggan E-Commerce menggunakan Machine Learning

> Sistem prediksi berbasis Machine Learning untuk mengidentifikasi pelanggan e-commerce yang berpotensi berhenti berlangganan (churn), dilengkapi web app deployment menggunakan Flask.

---

## 📌 Deskripsi Proyek

Churn pelanggan adalah kondisi ketika pelanggan berhenti menggunakan layanan atau tidak lagi melakukan transaksi. Proyek ini membangun model klasifikasi untuk memprediksi apakah seorang pelanggan akan churn atau tidak berdasarkan data perilaku dan transaksi e-commerce.

**Tujuan bisnis:** Membantu perusahaan e-commerce merancang strategi retensi yang lebih tepat sasaran berdasarkan prediksi data-driven.

---

## 👤 Pengembang

| Nama | NIM |
|------|-----|
| Teguh Al Azizul | 2355301197 |

**Program Studi:** D4 Teknik Informatika — Politeknik Caltex Riau (2025)

---

## 📂 Dataset

- **Sumber:** [Kaggle — Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/)
- **Jumlah data:** 5.630 baris
- **Jumlah fitur:** 20 fitur
- **Target:** `Churn` (1 = churn, 0 = tidak churn)
- **Distribusi kelas:** 83.2% tidak churn, 16.8% churn

### Fitur Utama

| Fitur | Tipe | Keterangan |
|-------|------|------------|
| `Tenure` | Numerik | Lama berlangganan (bulan) |
| `Complain` | Biner | Pernah komplain bulan lalu |
| `CashbackAmount` | Numerik | Rata-rata cashback diterima |
| `DaySinceLastOrder` | Numerik | Hari sejak pesanan terakhir |
| `SatisfactionScore` | Ordinal | Skor kepuasan (1–5) |
| `PreferredLoginDevice` | Kategorikal | Perangkat login favorit |
| `PreferredPaymentMode` | Kategorikal | Metode pembayaran favorit |
| `PreferedOrderCat` | Kategorikal | Kategori produk favorit |
| `MaritalStatus` | Kategorikal | Status pernikahan |
| `CityTier` | Ordinal | Tingkat kota pelanggan |

---

## 🔄 Alur Machine Learning Pipeline

```
Dataset (5.630 baris, 20 fitur)
      │
      ▼
Exploratory Data Analysis (EDA)
  ├── Informasi dasar dataset
  ├── Analisis missing values (heatmap)
  ├── Distribusi target churn
  ├── Analisis variabel numerik (histogram + KDE)
  ├── Analisis variabel kategorikal (countplot)
  ├── Matriks korelasi antar variabel
  └── Analisis bivariate (variabel vs churn)
      │
      ▼
Preprocessing Data
  ├── Pemisahan fitur (X) dan target (y)
  ├── Identifikasi fitur numerik & kategorikal
  ├── Pipeline numerik: imputasi median + StandardScaler
  ├── Pipeline kategorikal: imputasi modus + OneHotEncoding
  └── Train/Test split (80:20, stratified)
      │
      ▼
Baseline Model (Dummy Classifier)
(Benchmark minimal: akurasi 83%)
      │
      ▼
Training Model dengan SMOTE
  ├── Logistic Regression (+ class weighting)
  └── Random Forest (200 trees, max_depth=10)
      │
      ▼
Evaluasi & Perbandingan Model
  ├── Confusion Matrix
  ├── ROC Curve & AUC
  └── Perbandingan metrics (Accuracy, Precision, Recall, F1, AUC)
      │
      ▼
Cross Validation (5-fold) pada model terbaik
      │
      ▼
Feature Importance Analysis
      │
      ▼
Deployment (Flask Web App)
```

---

## 📊 Hasil Model

### Perbandingan Performa

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7948 | 0.4419 | 0.8211 | 0.5746 | 0.8053 |
| **Random Forest** ✅ | **0.9281** | **0.7853** | **0.7895** | **0.7874** | **0.8728** |

**Model terbaik:** Random Forest (berdasarkan F1-Score tertinggi: **0.7874**)

### Cross Validation (5-fold) — Random Forest
- Mean F1-Score: **0.7534**
- Std F1-Score: **0.0242** *(rendah → model stabil & tidak overfitting)*

### Top 3 Fitur Terpenting (Feature Importance)
1. 🥇 **Tenure** (0.2537) — masa berlangganan adalah penentu loyalitas terkuat
2. 🥈 **Complain** (0.0881) — pelanggan yang komplain sangat berisiko churn
3. 🥉 **CashbackAmount** (0.0588) — cashback rendah memicu churn

### Key Insights dari EDA
- Pelanggan dengan **Tenure rendah** (baru bergabung) paling berisiko churn
- **Complain = 1** sangat kuat memicu churn
- **Cashback lebih rendah** → risiko churn lebih tinggi
- Pelanggan yang **baru saja order** cenderung bertahan
- **COD** memiliki tingkat churn tertinggi dibanding metode pembayaran lain
- **Single** memiliki tingkat churn lebih tinggi dari Married

---

## 🚀 Deployment (Flask Web App)

Model Random Forest dideploy sebagai web application menggunakan Flask, dengan tampilan form input data pelanggan dan hasil prediksi real-time.
<img width="773" height="828" alt="image" src="https://github.com/user-attachments/assets/10bb6c91-cae5-4c50-8b82-1e9e00c79c98" />


### Fitur Web App
- Form input 18 fitur pelanggan
- Hasil prediksi: **CHURN / TIDAK CHURN**
- Probabilitas churn dalam persentase
- Tingkat risiko (Rendah / Sedang / Tinggi)
- Rekomendasi strategi retensi

### Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/username/churn-prediction.git
cd churn-prediction

# Install dependencies
pip install -r requirements.txt

# Training model (jika belum ada file .pkl)
jupyter notebook FIX_PROJEK_MACHINE_LEARNING_PREDIKSI_CHURN.ipynb

# Jalankan web app
python app.py
```

Buka browser di `http://localhost:5000`

---

## 📁 Struktur Proyek

```
churn-prediction/
├── FIX_PROJEK_MACHINE_LEARNING_PREDIKSI_CHURN.ipynb  # Notebook ML
├── app.py                          # Flask web application
├── templates/
│   └── index.html                  # Halaman web prediksi
├── static/
│   └── style.css                   # Styling web
├── model_churn_rf.pkl              # Model Random Forest tersimpan
├── preprocessor.pkl                # Pipeline preprocessing tersimpan
├── E Commerce Dataset.csv          # Dataset
├── requirements.txt
└── README.md
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---------|----------|
| `pandas`, `numpy` | Manipulasi & analisis data |
| `matplotlib`, `seaborn` | Visualisasi EDA |
| `scikit-learn` | ML pipeline, model, evaluasi |
| `imbalanced-learn` | SMOTE untuk class imbalance |
| `xgboost` | (Opsional) Model alternatif |
| `Flask` | Web app deployment |
| `pickle`, `joblib` | Penyimpanan model |

---

## 🧠 Teknik yang Diterapkan

- **SMOTE** — Synthetic Minority Over-sampling untuk menangani class imbalance
- **Pipeline scikit-learn** — Workflow preprocessing yang konsisten dan reproducible
- **Stratified Train/Test Split** — Menjaga proporsi kelas di data train dan test
- **5-Fold Cross Validation** — Validasi robustness dan generalisasi model
- **Feature Importance** — Identifikasi fitur paling berpengaruh terhadap churn

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis di Politeknik Caltex Riau. Dataset bersumber dari Kaggle.
