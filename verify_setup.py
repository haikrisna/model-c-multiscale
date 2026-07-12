"""
verify_setup.py
================
Jalankan skrip ini SEBELUM `streamlit run app.py` untuk memastikan:
1. TensorFlow ter-install dan versinya tercatat (bandingkan dengan versi Kaggle).
2. File .keras model (Model C — +Multiscale) ditemukan.
3. Model bisa di-load tanpa error (termasuk custom layer ECABlock).

Cara pakai:
    python verify_setup.py
    python verify_setup.py --models-dir path/ke/folder/model
"""

import argparse
import sys
import time

MODEL_KEY = "Model C — +Multiscale"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    print("=" * 60)
    print("1. Cek TensorFlow")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"  TensorFlow version : {tf.__version__}")
        print(f"  Keras version       : {tf.keras.__version__ if hasattr(tf.keras, '__version__') else '(bundled dengan TF)'}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  GPU terdeteksi      : {len(gpus)} -> {gpus if gpus else '(CPU-only, tidak masalah untuk inferensi)'}")
    except ImportError:
        print("  ❌ TensorFlow belum ter-install. Jalankan: pip install -r requirements.txt")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"2. Cek file model ({MODEL_KEY})")
    print("=" * 60)
    from model_utils import MODEL_CONFIGS, find_model_file, load_keras_model

    cfg = MODEL_CONFIGS[MODEL_KEY]
    path = find_model_file(args.models_dir, cfg["filenames"])
    if path:
        print(f"  ✅ {MODEL_KEY} -> {path}")
    else:
        print(f"  ❌ {MODEL_KEY} -> tidak ditemukan (dicari: {cfg['filenames']})")
        print(f"\n  Letakkan file .keras di folder '{args.models_dir}', lalu jalankan ulang skrip ini.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("3. Coba load model")
    print("=" * 60)
    t0 = time.time()
    try:
        model = load_keras_model(path)
        dt = time.time() - t0
        n_params = model.count_params()
        print(f"  ✅ {MODEL_KEY} loaded OK ({dt:.1f}s, {n_params:,} params)")
    except Exception as e:
        print(f"  ❌ {MODEL_KEY} GAGAL DIMUAT: {e}")
        sys.exit(1)

    print()
    print("Selesai. Semua ✅ — aplikasi siap dijalankan dengan:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
