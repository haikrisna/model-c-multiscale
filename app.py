"""
app.py — Demo Streamlit: Klasifikasi Cacat Permukaan Baja
EfficientNetB1 (Model C — +Multiscale)

Repo/folder ini SUDAH BERDIRI SENDIRI: hanya berisi 1 model (Model C — +Multiscale).
Siap langsung di-push sebagai repo GitHub sendiri lalu di-deploy sebagai
1 app Streamlit Community Cloud (Main file path: app.py).

Cara menjalankan lokal:
    streamlit run app.py

Letakkan file .keras hasil training (download dari Kaggle) di folder
`models/` di repo ini (nama file yang dikenali otomatis: lihat
models/README.txt).
"""

import io

import pandas as pd
import streamlit as st
from PIL import Image

from model_utils import (
    CLASS_LABELS,
    INPUT_SHAPE,
    MODEL_CONFIGS,
    find_model_file,
    load_keras_model,
    predict,
)

MODEL_KEY = "Model C — +Multiscale"
MODEL_CFG = MODEL_CONFIGS[MODEL_KEY]

st.set_page_config(
    page_title=f"Klasifikasi Cacat Baja — {MODEL_KEY}",
    page_icon="🔩",
    layout="wide",
)

st.title(f"🔩 Klasifikasi Cacat Permukaan Baja — {MODEL_KEY}")
st.caption(MODEL_CFG["desc"])

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")

    models_folder = st.text_input(
        "Folder berisi file .keras",
        value="models",
        help="Folder lokal (relatif terhadap tempat `streamlit run` dijalankan) "
             f"yang berisi file model, mis. {MODEL_CFG['filenames'][0]}",
    )

    if st.button("🔄 Reload model (bersihkan cache)"):
        load_keras_model.clear()
        st.success("Cache model dibersihkan. Model akan dimuat ulang saat prediksi berikutnya.")

    st.divider()
    st.markdown(f"**Ukuran input model:** {INPUT_SHAPE[0]}×{INPUT_SHAPE[1]} px (RGB)")
    st.caption("Gambar yang diunggah otomatis di-resize — tidak perlu diedit manual.")

    st.divider()
    model_path_preview = find_model_file(models_folder, MODEL_CFG["filenames"])
    with st.expander("📁 Status file model", expanded=True):
        if model_path_preview:
            st.markdown(f"✅ **{MODEL_KEY}**  \n`{model_path_preview}`")
        else:
            st.markdown(
                f"❌ **{MODEL_KEY}**  \nTidak ditemukan "
                f"(dicari: {', '.join(MODEL_CFG['filenames'])})"
            )

# ------------------------------------------------------------------
# UPLOAD GAMBAR
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Unggah gambar permukaan baja", type=["jpg", "jpeg", "png"])

col_img, col_result = st.columns([1, 1.4])

predict_btn = False
pil_img = None

if uploaded_file is not None:
    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    with col_img:
        st.image(pil_img, caption="Gambar yang diunggah", use_container_width=True)
        predict_btn = st.button("🔍 Prediksi", type="primary", use_container_width=True)
else:
    with col_img:
        st.info("Unggah gambar terlebih dahulu untuk memulai prediksi.")

# ------------------------------------------------------------------
# PREDIKSI
# ------------------------------------------------------------------
if predict_btn and pil_img is not None:
    with col_result:
        model_path = find_model_file(models_folder, MODEL_CFG["filenames"])

        if model_path is None:
            st.error(
                f"File model untuk **{MODEL_KEY}** tidak ditemukan di folder "
                f"`{models_folder}`.\n\nNama yang dicari: {MODEL_CFG['filenames']}.\n\n"
                "Pastikan file `.keras` hasil training sudah diletakkan di folder tersebut, "
                "atau ubah path folder di sidebar."
            )
        else:
            try:
                with st.spinner(f"Memuat {MODEL_KEY}..."):
                    model = load_keras_model(model_path)
                with st.spinner("Menjalankan prediksi..."):
                    pred_idx, probs = predict(model, pil_img)

                st.subheader(f"Hasil: {MODEL_KEY}")
                st.metric(
                    "Prediksi kelas",
                    CLASS_LABELS[pred_idx],
                    f"{probs[pred_idx] * 100:.2f}% confidence",
                )

                df = pd.DataFrame({"Kelas": CLASS_LABELS, "Probabilitas": probs}).set_index("Kelas")
                st.bar_chart(df)
                st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

            except Exception as e:
                st.error(
                    f"Gagal memuat/menjalankan model **{MODEL_KEY}**.\n\n"
                    f"Detail error: `{e}`\n\n"
                    "Penyebab paling umum: versi TensorFlow/Keras lokal berbeda dari "
                    "versi yang dipakai saat training di Kaggle. Cek `tf.__version__` "
                    "di kedua environment lalu samakan (lihat README)."
                )
