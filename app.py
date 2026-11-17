
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from ai_edge_litert.interpreter import Interpreter
from threading import Lock

from pathlib import Path

app = Flask(__name__, static_folder="public", static_url_path="")

# ==========================================
# 1. MEMUAT MODEL DAN DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

interpreter = Interpreter(
    model_path=str(BASE_DIR / "model" / "model_lstm.tflite")
)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
prediction_lock = Lock()

scaler = joblib.load(
    BASE_DIR / "model" / "scaler.pkl"
)

df = pd.read_csv(
    BASE_DIR / "dataset" / "daily-min-temperatures.csv"
)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

hasil = pd.read_csv(
    BASE_DIR / "hasil_grafik" / "hasil_prediksi.csv"
)

# ==========================================
# 2. PREDIKSI SUHU BERIKUTNYA
# ==========================================

def prediksi_berikutnya():
    # Ambil 30 suhu terakhir
    data_terakhir = df["Temp"].tail(30).values

    # Normalisasi menggunakan scaler training
    data_scaled = scaler.transform(
        data_terakhir.reshape(-1, 1)
    )

    # Format input LSTM: samples, timesteps, features
    X = data_scaled.reshape(1, 30, 1).astype(np.float32)

    with prediction_lock:
        interpreter.set_tensor(input_index, X)
        interpreter.invoke()
        prediksi_scaled = interpreter.get_tensor(output_index)

    # Kembalikan ke Celsius
    prediksi = scaler.inverse_transform(
        prediksi_scaled
    )[0][0]

    tanggal_terakhir = df["Date"].max()
    tanggal_prediksi = (
        tanggal_terakhir + pd.Timedelta(days=1)
    )

    return {
        "tanggal": tanggal_prediksi.strftime("%d-%m-%Y"),
        "suhu": round(float(prediksi), 2)
    }


# ==========================================
# 3. HALAMAN UTAMA
# ==========================================

@app.route("/")
def index():
    prediksi = prediksi_berikutnya()

    return render_template(
        "index.html",
        prediksi=prediksi
    )


# ==========================================
# 4. API DATA HISTORIS
# ==========================================

@app.route("/api/historis")
def api_historis():
    data = df.tail(365)

    return jsonify({
        "tanggal": data["Date"].dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "suhu": data["Temp"].tolist()
    })


# ==========================================
# 5. API HASIL EVALUASI
# ==========================================

@app.route("/api/evaluasi")
def api_evaluasi():
    return jsonify({
        "tanggal": hasil["Tanggal"].tolist(),
        "aktual": hasil["Suhu_Aktual"].tolist(),
        "prediksi": hasil["Suhu_Prediksi"].tolist(),
        "rmse": 2.2295,
        "mae": 1.7522,
        "mse": 4.9709
    })


# ==========================================
# 6. MENJALANKAN FLASK
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
