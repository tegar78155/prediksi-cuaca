
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. PERSIAPAN
# ==========================================

tf.keras.utils.set_random_seed(42)

Path("model").mkdir(exist_ok=True)
Path("hasil_grafik").mkdir(exist_ok=True)

# ==========================================
# 2. MEMUAT DATA PREPROCESSING
# ==========================================

data = np.load("model/data_preprocessing.npz")

X_train = data["X_train"]
y_train = data["y_train"]

print("=== DATA PELATIHAN ===")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# ==========================================
# 3. MEMBAGI DATA TRAIN DAN VALIDASI
# ==========================================

# Menggunakan 10% data latih paling akhir
# sebagai data validasi, tanpa mengacak waktu.

split = int(len(X_train) * 0.9)

X_fit = X_train[:split]
y_fit = y_train[:split]

X_val = X_train[split:]
y_val = y_train[split:]

print("\n=== PEMBAGIAN TRAIN / VALIDASI ===")
print("Data training:", X_fit.shape)
print("Data validasi:", X_val.shape)

# ==========================================
# 4. MEMBANGUN ARSITEKTUR LSTM
# ==========================================

model = Sequential([
    tf.keras.Input(shape=(30, 1)),

    LSTM(
        units=50,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        units=50,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(units=25),

    Dense(units=1)
])

# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("\n=== ARSITEKTUR MODEL ===")
model.summary()

# ==========================================
# 6. MELATIH MODEL LSTM
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\n=== MEMULAI TRAINING ===")

history = model.fit(
    X_fit,
    y_fit,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    shuffle=False,
    verbose=1
)

# ==========================================
# 7. MENYIMPAN MODEL
# ==========================================

model.save("model/model_lstm.keras")

print("\nMODEL BERHASIL DISIMPAN!")

# ==========================================
# 8. GRAFIK LOSS TRAINING
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Grafik Loss Pelatihan LSTM")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "hasil_grafik/grafik_loss.png",
    dpi=300
)

plt.show()

print("\nTRAINING LSTM SELESAI!")
