# Model C — +Multiscale — Demo Streamlit Klasifikasi Cacat Permukaan Baja

Repo ini berisi 1 model saja (Model C — +Multiscale), siap langsung dideploy sebagai 1 app
Streamlit terpisah dari model lain.

EfficientNetB1 + fusion multiscale (tap Block3, Block5, top_activation).

## Isi folder

```
model-c-multiscale/
├── app.py                        # Aplikasi Streamlit (model ini saja)
├── model_utils.py                # ECABlock (custom layer) + fungsi load/predict
├── verify_setup.py               # Cek environment & model sebelum run app
├── requirements.txt              # Dependensi (Mac Intel / umum)
├── requirements-macos-arm.txt    # Dependensi (Mac Apple Silicon M1/M2/M3/M4)
└── models/                       # ← taruh file .keras Anda di sini
    └── README.txt
```

## 1. Instalasi

```bash
cd model-c-multiscale
python3 -m venv venv
source venv/bin/activate

# Mac Apple Silicon (M1/M2/M3/M4):
pip install -r requirements-macos-arm.txt

# Mac Intel (atau kalau requirements-macos-arm.txt bermasalah):
pip install -r requirements.txt
```

Kalau versi `tensorflow==2.15.0` tidak cocok dengan versi Kaggle Anda, edit
angka versinya di file requirements sebelum install (lihat output
`print(f"TensorFlow Version : {tf.__version__}")` di Cell 2 notebook).

## 2. Siapkan file model

Taruh 1 file `.keras` untuk **Model C — +Multiscale** di folder `models/` (lihat
`models/README.txt` untuk daftar nama file yang dikenali otomatis).

## 3. Verifikasi (opsional tapi disarankan)

```bash
python verify_setup.py
```

## 4. Jalankan aplikasi lokal

```bash
streamlit run app.py
```

## 5. Deploy ke Streamlit Community Cloud

1. Push seluruh isi folder ini (`app.py`, `model_utils.py`, requirements,
   `models/`) ke **repo GitHub baru** (khusus untuk Model C — +Multiscale).
2. Buka [share.streamlit.io](https://share.streamlit.io) → **"New app"**.
3. Pilih repo tadi, **Main file path**: `app.py`.
4. Deploy. Ulangi proses yang sama di 3 folder lain (model lain) sebagai
   repo & app terpisah — setiap app hanya memuat 1 model, jadi kebutuhan
   RAM-nya ringan dan tidak saling membebani.

> Catatan ukuran file: kalau file `.keras` Anda besar (puluhan-ratusan MB),
> cek limit ukuran repo/`git-lfs` di GitHub & Streamlit Cloud. Kalau masih
> bermasalah, model bisa di-host eksternal (mis. Hugging Face Hub) dan
> di-download otomatis saat app start — kabari saya kalau mau dibuatkan.

## 6. Catatan penting soal metodologi (dibaca sebelum dipakai untuk sidang)

Notebook training memakai **ROI crop** berdasarkan mask ground-truth
(`load_image_with_roi`, Cell 7) — bukan gambar penuh — sebagai input model.
Untuk gambar baru yang diunggah pengguna di aplikasi ini, **mask tidak
tersedia**, sehingga app mengklasifikasikan gambar **penuh** (full-frame,
resize langsung ke 224×224). Akurasi pada gambar full-frame kemungkinan
berbeda dari angka akurasi di tabel hasil evaluasi notebook (yang dievaluasi
pada crop ROI). Sampaikan perbedaan ini secara eksplisit kalau dipakai di
depan dosen penguji.

## 7. Troubleshooting singkat

| Gejala | Kemungkinan penyebab | Solusi |
|---|---|---|
| `ValueError: Unknown layer: ECABlock` saat load model | `model_utils.py` tidak ter-import sebelum `load_model`, atau definisi ECABlock beda dari notebook | Pastikan jalankan lewat `app.py`/`verify_setup.py`, dan definisi ECABlock identik dengan notebook |
| Error terkait `mixed_float16` / dtype policy | Model disimpan dengan mixed precision aktif | Update TensorFlow ke versi yang sama dengan Kaggle |
| Model gagal diload, pesan menyinggung versi Keras/format file | Versi TensorFlow lokal beda jauh dari Kaggle | Samakan versi TF |
| `pip install tensorflow-macos` gagal di M-series | Konflik Python version / arsitektur | Gunakan Python 3.10–3.12, arch arm64 |
| App di Streamlit Cloud gagal deploy / "out of memory" | File `.keras` terlalu besar untuk RAM gratis | Cek ukuran file model, upgrade paket kalau perlu |
| Aplikasi lambat saat load model pertama kali | Wajar — model besar, butuh beberapa detik deserialize | Sudah di-cache (`st.cache_resource`) — prediksi berikutnya lebih cepat |
# model-c-multiscale
