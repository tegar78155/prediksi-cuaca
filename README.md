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

Folder `public/` menyimpan aset statis yang disajikan Vercel, sedangkan `vercel.json` mengecualikan berkas pelatihan dan aset yang tidak dibutuhkan aplikasi saat deployment.

### Catatan ukuran fungsi

Aplikasi menggunakan TensorFlow CPU untuk menjalankan model LSTM. Jika build melewati batas fungsi standar Vercel sebesar 500 MB, aktifkan **Large Functions** untuk project/account jika tersedia. Jika fitur itu tidak tersedia pada paket akun, deploy aplikasi ini pada platform yang mendukung container atau ukuran image lebih besar.
