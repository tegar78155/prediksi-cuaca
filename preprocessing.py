
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. MEMBACA DATASET
# ==========================================

file_dataset = Path("dataset/daily-min-temperatures.csv")

df = pd.read_csv(file_dataset)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.set_index("Date")

print("=== INFORMASI DATASET ===")
print("Jumlah data:", len(df))
print("Missing values:", df["Temp"].isna().sum())
print("Tanggal duplikat:", df.index.duplicated().sum())

# Validasi kualitas data
if df["Temp"].isna().any():
    raise ValueError("Dataset masih memiliki nilai kosong.")

if df.index.duplicated().any():
    raise ValueError("Dataset memiliki tanggal duplikat.")

# ==========================================
# 2. MEMBAGI DATA BERDASARKAN WAKTU
# ==========================================

data = df[["Temp"]].values

train_size = int(len(data) * 0.8)

train_data = data[:train_size]
test_data = data[train_size:]

print("\n=== PEMBAGIAN DATA ===")
print("Data latih:", len(train_data))
print("Data uji:", len(test_data))

# ==========================================
# 3. NORMALISASI DATA
# ==========================================

scaler = MinMaxScaler(feature_range=(0, 1))

# Scaler hanya dilatih pada data latih
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

Path("model").mkdir(exist_ok=True)

joblib.dump(scaler, "model/scaler.pkl")

print("\nScaler berhasil disimpan!")

# ==========================================
# 4. WINDOWING DATA
# ==========================================

timesteps = 30

def create_sequences(dataset, window):
    X, y = [], []

    for i in range(window, len(dataset)):
        X.append(dataset[i-window:i, 0])
        y.append(dataset[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y

X_train, y_train = create_sequences(
    train_scaled, timesteps
)

# Tambahkan 30 hari terakhir dari data latih
# sebagai konteks untuk prediksi data uji
test_context = np.concatenate(
    (train_scaled[-timesteps:], test_scaled),
    axis=0
)

X_test, y_test = create_sequences(
    test_context, timesteps
)

# ==========================================
# 5. MENYIMPAN HASIL
# ==========================================

np.savez(
    "model/data_preprocessing.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print("\n=== HASIL WINDOWING ===")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nPREPROCESSING BERHASIL!")
