"""
model_utils.py
================
Utilitas bersama untuk aplikasi Streamlit klasifikasi cacat permukaan baja
(EfficientNetB1 + ECA + Multiscale, Model A/B/C/D).

PENTING: kelas ECABlock di bawah ini adalah SALINAN PERSIS dari definisi di
notebook training (`ubah-kerasversion.ipynb`, Cell 11). Kelas ini WAJIB
tersedia & teregistrasi persis sama sebelum `keras.models.load_model()`
dipanggil, karena Model B, C, dan D menyimpan layer custom ECABlock di
dalam arsitekturnya. Jika Anda mengubah definisi ECABlock di notebook,
salin ulang ke sini juga.
"""

import math
import os
from typing import Optional, List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# KONFIGURASI (harus sama persis dengan notebook training)
# ============================================================

INPUT_SHAPE = (224, 224)          # (height, width) — lihat Cell 3 notebook
NUM_CLASSES = 4

CLASS_NAMES = {
    0: 'Pitting (Class 1)',
    1: 'Crazing (Class 2)',
    2: 'Scratches (Class 3)',
    3: 'Inclusion/Others (Class 4)',
}
CLASS_LABELS = list(CLASS_NAMES.values())


# ============================================================
# ECA BLOCK — SALINAN PERSIS DARI NOTEBOOK (Cell 11)
# ============================================================

@tf.keras.utils.register_keras_serializable(package='CustomLayers')
class ECABlock(tf.keras.layers.Layer):
    """
    Efficient Channel Attention Block (Wang et al., 2020).
    Formula: omega = sigmoid(Conv1D(GAP(X))),  Y = omega (x) X
    """

    def __init__(self, kernel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.mul = tf.keras.layers.Multiply()

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if self.kernel_size is None:
            b, gamma = 1, 2
            t = int(abs((math.log2(channels) + b) / gamma))
            k = t if t % 2 else t + 1
            self._k = max(k, 3)
        else:
            self._k = self.kernel_size
        self._channels = channels
        self.reshape = tf.keras.layers.Reshape((channels, 1))
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1, kernel_size=self._k, padding='same',
            use_bias=False, name=self.name + '_conv1d'
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.gap(inputs)
        x = self.reshape(x)
        x = self.conv1d(x)
        x = self.sigmoid(x)
        x = tf.reshape(x, (-1, 1, 1, self._channels))
        return self.mul([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


CUSTOM_OBJECTS = {'ECABlock': ECABlock}


# ============================================================
# DAFTAR MODEL — sesuaikan `filenames` dengan nama file .keras
# hasil download dari Kaggle (lihat penjelasan checkpoint di
# notebook Cell 14: f'{model_name}_best.keras')
# ============================================================

MODEL_CONFIGS = {
    "Model A — Baseline": {
        "filenames": [
            "ModelA_Baseline_best.keras", "ModelA_best.keras", "model_a.keras",
        ],
        "desc": "EfficientNetB1 murni, tanpa ECA maupun Multiscale.",
    },
    "Model B — +ECA": {
        "filenames": [
            "ModelB_ECA_best.keras", "ModelB_best.keras", "model_b.keras",
        ],
        "desc": "EfficientNetB1 + Efficient Channel Attention (ECA) pada top_activation.",
    },
    "Model C — +Multiscale": {
        "filenames": [
            "ModelC_Multiscale_best.keras", "ModelC_best.keras", "model_c.keras",
        ],
        "desc": "EfficientNetB1 + fusion multiscale (tap Block3, Block5, top_activation).",
    },
    "Model D — +ECA+Multiscale (Proposed)": {
        "filenames": [
            "ModelD_ECA_Multiscale_PROPOSED_v6_best.keras",
            "ModelD_best.keras", "model_d.keras",
        ],
        "desc": "Model usulan: ECA (tap Block5 & top) + fusion multiscale.",
    },
}


# ============================================================
# LOADING & INFERENCE
# ============================================================

def find_model_file(folder: str, candidates: List[str]) -> Optional[str]:
    """Cari file model pertama yang cocok dari daftar nama kandidat."""
    if not folder or not os.path.isdir(folder):
        return None
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    """
    Muat file .keras. `compile=False` sengaja dipakai supaya loader TIDAK
    perlu merekonstruksi optimizer/custom loss (categorical_focal_loss) —
    yang tidak dibutuhkan untuk inferensi dan menjadi sumber error umum
    saat deploy di luar Kaggle.
    """
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
    )
    return model


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Resize ke INPUT_SHAPE dan biarkan dalam rentang [0, 255] float32 —
    TANPA pembagian manual /255 — karena EfficientNetB1 (backbone) sudah
    memiliki layer preprocessing internal (Rescaling + Normalization).
    Ini harus konsisten dengan Cell 9 notebook training.
    """
    img = pil_img.convert('RGB')
    img = img.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]), Image.BILINEAR)
    arr = np.asarray(img).astype('float32')
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


# ============================================================
# MITIGASI OUT-OF-DISTRIBUTION (OOD) — LAPISAN TAMBAHAN DI INFERENCE
# ============================================================
# LATAR BELAKANG: model ini softmax 4-kelas (Pitting/Crazing/Scratches/
# Inclusion) TANPA kelas "bukan cacat"/"bukan baja". Softmax WAJIB memilih
# salah satu dari 4 kelas untuk input apa pun — termasuk foto kucing, orang,
# dll — karena tidak ada opsi "none of the above" di ruang output model.
#
# Dua lapisan mitigasi di bawah ini murni POST-HOC / PRE-FILTER di level
# INFERENCE (app), TIDAK mengubah bobot model, TIDAK retrain, dan TIDAK
# menyentuh angka akurasi/F1/ROC-AUC yang sudah dilaporkan di notebook
# (evaluasi test set notebook masih manggil model.predict() polos tanpa
# filter ini). Filter ini murni aturan keputusan tambahan di app demo.
#
# CATATAN JUJUR (sampaikan juga di sidang kalau ditanya): dua lapisan ini
# BUKAN out-of-distribution detector yang proper secara riset (itu perlu
# kelas negatif eksplisit + retrain, atau metode seperti Mahalanobis
# distance / energy-based OOD detection). Ini heuristik + confidence
# thresholding sederhana yang cukup untuk demo, dengan keterbatasan:
#   - Heuristik saturasi/tekstur bisa false-positive (foto baja dengan
#     pencahayaan aneh ke-reject) atau false-negative (foto grayscale non-
#     baja lolos, mis. foto aspal/beton close-up).
#   - Confidence softmax dikenal cenderung overconfident pada input OOD
#     (neural network umumnya tidak well-calibrated di luar training
#     distribution), jadi threshold confidence bukan jaminan mutlak.
# Nilai threshold di bawah ini BELUM di-tuning sistematis — sebelum dipakai
# di sidang, coba beberapa contoh gambar baja asli & non-baja (kucing,
# orang, foto acak) lalu sesuaikan angkanya lewat slider di sidebar app.

from ood_heuristics import (
    SATURATION_THRESHOLD,
    MIN_TEXTURE_STD,
    compute_ood_heuristics,
    heuristic_ood_check,
)

CONFIDENCE_THRESHOLD = 0.60  # di bawah ini: model dianggap "tidak cukup yakin"
# CATATAN: SATURATION_THRESHOLD & MIN_TEXTURE_STD sekarang didefinisikan di
# ood_heuristics.py (biar bisa dipakai calibrate_ood_thresholds.py tanpa
# perlu load TensorFlow/Streamlit). Untuk kalibrasi threshold ini pakai
# data asli, jalankan: python calibrate_ood_thresholds.py --help


def predict(
    model,
    pil_img: Image.Image,
    apply_heuristic_filter: bool = True,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    saturation_threshold: float = SATURATION_THRESHOLD,
    min_texture_std: float = MIN_TEXTURE_STD,
) -> dict:
    """
    Jalankan pipeline prediksi lengkap dengan mitigasi OOD.

    Return dict:
      status : 'ok' | 'rejected_heuristic' | 'low_confidence'
      pred_idx : int atau None (None kalau status == 'rejected_heuristic',
                 karena model TIDAK dipanggil sama sekali)
      probs : np.ndarray atau None (None kalau status == 'rejected_heuristic')
      reason : str atau None, penjelasan kalau status != 'ok'
      ood_stats : dict hasil compute_ood_heuristics (selalu diisi, berguna
                  untuk ditampilkan di UI debug/sidang)
    """
    ood_stats = compute_ood_heuristics(pil_img)

    if apply_heuristic_filter:
        reject_reason = heuristic_ood_check(
            pil_img,
            saturation_threshold=saturation_threshold,
            min_texture_std=min_texture_std,
        )
        if reject_reason:
            return {
                'status': 'rejected_heuristic',
                'pred_idx': None,
                'probs': None,
                'reason': reject_reason,
                'ood_stats': ood_stats,
            }

    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    max_prob = float(probs[pred_idx])

    if max_prob < confidence_threshold:
        return {
            'status': 'low_confidence',
            'pred_idx': pred_idx,
            'probs': probs,
            'reason': (
                f"Confidence tertinggi cuma {max_prob * 100:.1f}% "
                f"(ambang {confidence_threshold * 100:.0f}%) — model tidak "
                "cukup yakin gambar ini salah satu dari 4 jenis cacat yang "
                "dikenali."
            ),
            'ood_stats': ood_stats,
        }

    return {
        'status': 'ok',
        'pred_idx': pred_idx,
        'probs': probs,
        'reason': None,
        'ood_stats': ood_stats,
    }
