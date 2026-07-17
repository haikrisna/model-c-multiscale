"""
ood_heuristics.py
================
Heuristik ringan (murni PIL + numpy, TANPA TensorFlow/Streamlit) untuk
menyaring gambar yang kemungkinan besar BUKAN foto permukaan baja
industrial, sebelum model dipanggil.

Dipisah dari model_utils.py supaya bisa dipakai di script kalibrasi
(calibrate_ood_thresholds.py) tanpa perlu load TensorFlow/Streamlit yang
berat — kalibrasi threshold cuma butuh baca gambar & hitung statistik,
sama sekali tidak butuh model.

CATATAN JUJUR: ini heuristik sederhana berbasis statistik warna/tekstur,
BUKAN out-of-distribution detector yang proper secara riset (itu perlu
kelas negatif eksplisit + retrain, atau metode seperti Mahalanobis
distance / energy-based OOD detection). JANGAN pakai nilai default di
bawah ini tanpa kalibrasi — jalankan calibrate_ood_thresholds.py dulu
pakai sample gambar baja asli & non-baja punya kamu.
"""

from typing import Optional

import numpy as np
from PIL import Image

# Nilai default ini cuma starting point kasar (BELUM dikalibrasi pakai data
# asli) — akan di-override oleh hasil calibrate_ood_thresholds.py.
SATURATION_THRESHOLD = 0.15
MIN_TEXTURE_STD = 5.0


def compute_ood_heuristics(pil_img: Image.Image) -> dict:
    """
    Hitung statistik kasar gambar yang dipakai heuristik pre-filter.

    - mean_saturation: rata-rata channel S di HSV, dinormalisasi [0,1].
      Foto permukaan baja industrial umumnya near-grayscale (saturasi
      rendah); foto natural (kucing, orang, pemandangan) biasanya jauh
      lebih jenuh warnanya — TAPI ini bisa overlap tergantung pencahayaan
      (lihat catatan kalibrasi).
    - texture_std: std dev grayscale, proxy kasar kandungan tekstur.
      Foto close-up cacat permukaan punya variasi piksel lokal yang khas;
      gambar yang terlalu flat/blur/solid-color kemungkinan bukan itu.
    """
    img_rgb = pil_img.convert('RGB')

    img_hsv = np.asarray(img_rgb.convert('HSV')).astype('float32')
    mean_saturation = float(img_hsv[..., 1].mean()) / 255.0

    gray = np.asarray(img_rgb.convert('L')).astype('float32')
    texture_std = float(gray.std())

    return {
        'mean_saturation': mean_saturation,
        'texture_std': texture_std,
    }


def heuristic_ood_check(
    pil_img: Image.Image,
    saturation_threshold: float = SATURATION_THRESHOLD,
    min_texture_std: float = MIN_TEXTURE_STD,
) -> Optional[str]:
    """
    Return None kalau gambar lolos heuristik (lanjut ke model), atau
    string alasan penolakan kalau gambar kemungkinan besar BUKAN foto
    permukaan baja industrial.
    """
    stats = compute_ood_heuristics(pil_img)

    if stats['mean_saturation'] > saturation_threshold:
        return (
            f"Saturasi warna gambar tinggi ({stats['mean_saturation']:.2f}, "
            f"ambang {saturation_threshold:.2f}) — foto permukaan baja industrial "
            "biasanya near-grayscale. Gambar ini kemungkinan besar bukan foto "
            "permukaan baja."
        )
    if stats['texture_std'] < min_texture_std:
        return (
            f"Tekstur gambar terlalu flat (std={stats['texture_std']:.1f}, "
            f"ambang {min_texture_std:.1f}) — kemungkinan bukan foto close-up "
            "permukaan baja."
        )
    return None
