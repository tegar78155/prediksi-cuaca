# Prediksi Suhu LSTM

## Menjalankan secara lokal

Python 3.12 direkomendasikan agar versinya sama dengan deployment Vercel.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000`.

## Deploy ke Vercel

1. Push repository ke GitHub.
2. Di Vercel, pilih **Add New > Project** lalu import repository ini.
3. Pastikan **Root Directory** menunjuk ke folder repository. Vercel akan mendeteksi Flask dari `app.py` dan memasang dependensi dari `requirements.txt`.
4. Pilih **Deploy**. Setiap push berikutnya ke branch produksi akan memicu deployment baru.

Alternatif melalui CLI, dari folder repository:

```powershell
npx vercel
npx vercel --prod
```

Folder `public/` menyimpan aset statis yang disajikan Vercel. Aplikasi menggunakan model LiteRT di `model/model_lstm.tflite` dan metadata scaler di `model/scaler.json` agar tidak perlu memaketkan TensorFlow atau scikit-learn saat deployment. `vercel.json` mengecualikan model Keras, pickle scaler, dan berkas pelatihan yang tidak dibutuhkan saat deployment.

### Melatih ulang model

Jika ingin menjalankan ulang preprocessing, training, atau evaluasi, pasang dependensi tambahan lalu ekspor ulang model LiteRT:

```powershell
python -m pip install -r requirements-training.txt
python preprocessing.py
python train_model.py
python evaluasi.py
python convert_model.py
```
