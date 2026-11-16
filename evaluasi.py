
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# ==========================================
# 1. PERSIAPAN
# ==========================================

Path("hasil_grafik").mkdir(exist_ok=True)

# ==========================================
# 2. MEMUAT MODEL DAN SCALER
# ==========================================

print("Memuat model LSTM...")

model = load_model(
    "model/model_lstm.keras"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

data = np.load(
    "model/data_preprocessing.npz"
)

X_test = data["X_test"]
y_test = data["y_test"]

print("Model dan data berhasil dimuat!")
print("Jumlah sampel uji:", len(X_test))

# ==========================================
# 3. MELAKUKAN PREDIKSI
# ==========================================

print("\nMelakukan prediksi...")

prediksi_scaled = model.predict(
    X_test,
    verbose=0
)

# Mengembalikan data ke satuan Celsius
prediksi = scaler.inverse_transform(
    prediksi_scaled
).flatten()

aktual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

# ==========================================
# 4. MENGHITUNG METRIK EVALUASI
# ==========================================

mse = mean_squared_error(
    aktual,
    prediksi
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    aktual,
    prediksi
)

print("\n=== HASIL EVALUASI MODEL LSTM ===")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f} °C")
print(f"MAE  : {mae:.4f} °C")

# ==========================================
# 5. MENYIAPKAN TANGGAL DATA UJI
# ==========================================

df = pd.read_csv(
    "dataset/daily-min-temperatures.csv"
)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

tanggal_uji = df["Date"].iloc[
    -len(aktual):
].values

# ==========================================
# 6. MENYIMPAN HASIL PREDIKSI
# ==========================================

hasil = pd.DataFrame({
    "Tanggal": tanggal_uji,
    "Suhu_Aktual": aktual,
    "Suhu_Prediksi": prediksi
})

hasil.to_csv(
    "hasil_grafik/hasil_prediksi.csv",
    index=False
)

print("\n=== LIMA HASIL PREDIKSI PERTAMA ===")
print(hasil.head())

# ==========================================
# 7. GRAFIK AKTUAL VS PREDIKSI
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    tanggal_uji,
    aktual,
    label="Suhu Aktual",
    color="blue",
    linewidth=1
)

plt.plot(
    tanggal_uji,
    prediksi,
    label="Suhu Prediksi LSTM",
    color="orange",
    linewidth=1
)

plt.title(
    "Perbandingan Suhu Aktual dan Prediksi LSTM"
)

plt.xlabel("Tanggal")
plt.ylabel("Suhu Minimum (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "hasil_grafik/grafik_prediksi.png",
    dpi=300
)

plt.show()

# ==========================================
# 8. KESIMPULAN OUTPUT
# ==========================================

print("\nEVALUASI MODEL BERHASIL!")
print("Grafik dan CSV tersimpan.")
