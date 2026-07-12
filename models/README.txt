Letakkan file .keras hasil training untuk Model C — +Multiscale di folder ini
(download dari /kaggle/working/models/ setelah training selesai).

Nama file yang otomatis dikenali aplikasi (salah satu saja, sesuai urutan
prioritas pencarian):

- ModelC_Multiscale_best.keras
- ModelC_best.keras
- model_c.keras

Kalau nama file Anda beda dari daftar di atas, cara termudah: rename filenya
sesuai salah satu nama di atas, ATAU edit daftar `filenames` di model_utils.py
(MODEL_CONFIGS) supaya cocok dengan nama file Anda.
