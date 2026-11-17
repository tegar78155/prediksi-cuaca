from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
source = BASE_DIR / "model" / "model_lstm.keras"
destination = BASE_DIR / "model" / "model_lstm.tflite"

model = load_model(source)
inputs = tf.keras.Input(batch_shape=(1, 30, 1), dtype=tf.float32)
fixed_batch_model = tf.keras.Model(inputs, model(inputs, training=False))

converter = tf.lite.TFLiteConverter.from_keras_model(fixed_batch_model)
destination.write_bytes(converter.convert())
print(f"LiteRT model saved to {destination}")
