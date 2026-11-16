
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PERSIAPAN FOLDER
# ==========================================

folder_dataset = Path("dataset")
folder_grafik = Path("hasil_grafik")

folder_dataset.mkdir(exist_ok=True)
folder_grafik.mkdir(exist_ok=True)

# ==========================================
# 2. MEMBACA DATASET
# ==========================================

url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "daily-min-temperatures.csv"
)

lokasi_file = folder_dataset / "daily-min-temperatures.csv"

if not lokasi_file.exists():
    print("Mengunduh dataset...")
    data = pd.read_csv(url)
    data.to_csv(lokasi_file, index=False)

df = pd.read_csv(lokasi_file)

print("\n=== LIMA DATA PERTAMA ===")
print(df.head())

print("\n=== UKURAN DATASET ===")
print(df.shape)

print("\n=== INFORMASI DATASET ===")
df.info()

# ==========================================
# 3. PEMERIKSAAN DATA
# ==========================================

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DATA DUPLIKAT ===")
print(df.duplicated().sum())

# ==========================================
# 4. MENGATUR KOLOM TANGGAL
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])

df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print("\n=== DATA SETELAH PREPROCESSING AWAL ===")
print(df.head())

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

# ==========================================
# 5. VISUALISASI DATA HISTORIS
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df.index,
    df["Temp"],
    color="blue",
    linewidth=0.8,
    label="Suhu Minimum Harian"
)

plt.title("Suhu Minimum Harian Melbourne 1981-1990")
plt.xlabel("Tanggal")
plt.ylabel("Suhu Minimum (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Menyimpan grafik
plt.savefig(
    folder_grafik / "grafik_suhu_harian.png",
    dpi=300
)

plt.show()

print("\nANALISIS DATASET BERHASIL DIJALANKAN!")
print("Grafik tersimpan di folder hasil_grafik.")
