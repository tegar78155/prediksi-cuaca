from pathlib import Path
import json

import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
source = BASE_DIR / "model" / "model_lstm.keras"
destination = BASE_DIR / "model" / "model_lstm.tflite"
scaler_path = BASE_DIR / "model" / "scaler.pkl"
scaler_destination = BASE_DIR / "model" / "scaler.json"

model = load_model(source)
inputs = tf.keras.Input(batch_shape=(1, 30, 1), dtype=tf.float32)
fixed_batch_model = tf.keras.Model(inputs, model(inputs, training=False))

converter = tf.lite.TFLiteConverter.from_keras_model(fixed_batch_model)
destination.write_bytes(converter.convert())
print(f"LiteRT model saved to {destination}")

scaler = joblib.load(scaler_path)
scaler_metadata = {
    "scale": float(scaler.scale_[0]),
    "min": float(scaler.min_[0]),
}
scaler_destination.write_text(
    json.dumps(scaler_metadata, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Scaler metadata saved to {scaler_destination}")
