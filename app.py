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
    CONFIDENCE_THRESHOLD,
    INPUT_SHAPE,
    MIN_TEXTURE_STD,
    MODEL_CONFIGS,
    SATURATION_THRESHOLD,
    find_model_file,
    load_keras_model,
    predict,
)

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
MODEL_KEY = "Model C — +Multiscale"
MODEL_CFG = MODEL_CONFIGS[MODEL_KEY]

RESEARCHER_INFO = {
    "nama": "Krisna Kukuh Wijaya",
    "nim": "4611422072",
    "prodi": "Teknik Informatika",
    "pembimbing": "Endang Sugiharti, S.Si., M.Kom.",
    "judul": (
        "Implementasi EfficientNet-B1 Menggunakan Efficient Channel "
        "Attention Module dan Multiscale Feature Extraction untuk "
        "Klasifikasi Cacat Permukaan Baja"
    ),
    "deskripsi": (
        "Penelitian ini mengimplementasikan arsitektur EfficientNet-B1 yang "
        "diperkuat dengan modul Efficient Channel Attention (ECA) dan teknik "
        "Multiscale Feature Extraction untuk meningkatkan akurasi klasifikasi "
        "cacat permukaan baja. Model diuji pada empat kelas cacat: Pitting, "
        "Crazing, Scratches, dan Inclusion."
    ),
}

st.set_page_config(
    page_title="Klasifikasi Cacat Baja — Krisna Kukuh Wijaya",
    page_icon="🔩",
    layout="wide",
)

# ------------------------------------------------------------------
# CSS — Shadcn-inspired: light mode, zinc palette, clean, minimal
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #09090b;
}

/* ---- Global overrides ---- */
.stApp { background-color: #ffffff; }
section[data-testid="stSidebar"] {
    background-color: #fafafa;
    border-right: 1px solid #e4e4e7;
}

/* ---- Welcome page ---- */
.welcome-wrap {
    max-width: 580px;
    margin: 6rem auto 0 auto;
    text-align: center;
}
.welcome-badge {
    display: inline-block;
    border: 1px solid #e4e4e7;
    border-radius: 6px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #71717a;
    margin-bottom: 1.5rem;
    background: #f4f4f5;
}
.welcome-title {
    font-size: 2rem;
    font-weight: 700;
    color: #09090b;
    line-height: 1.25;
    margin-bottom: 1rem;
}
.welcome-desc {
    font-size: 0.93rem;
    color: #71717a;
    line-height: 1.75;
    margin-bottom: 2.5rem;
}
.welcome-divider {
    border: none;
    border-top: 1px solid #e4e4e7;
    margin: 2rem 0;
}

/* ---- Page header ---- */
.page-header {
    padding: 1.5rem 0 1rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #e4e4e7;
}
.page-header-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 0.35rem;
}
.page-header-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #09090b;
    margin: 0;
}
.page-header-sub {
    font-size: 0.85rem;
    color: #71717a;
    margin-top: 0.3rem;
}

/* ---- Card ---- */
.card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 10px;
    padding: 1.5rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 1rem;
}

/* ---- Profile ---- */
.profile-initials {
    width: 52px;
    height: 52px;
    border-radius: 8px;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    color: #52525b;
    font-size: 1.1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
}
.profile-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #09090b;
    margin-bottom: 0.15rem;
}
.profile-nim {
    font-size: 0.8rem;
    color: #a1a1aa;
    font-family: 'Courier New', monospace;
    margin-bottom: 1.4rem;
}
.meta-row {
    display: flex;
    gap: 1rem;
    padding: 0.65rem 0;
    border-top: 1px solid #f4f4f5;
    align-items: flex-start;
}
.meta-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #a1a1aa;
    min-width: 100px;
    padding-top: 1px;
}
.meta-value {
    font-size: 0.875rem;
    color: #3f3f46;
    line-height: 1.5;
}
.judul-block {
    background: #fafafa;
    border: 1px solid #e4e4e7;
    border-left: 3px solid #d4d4d8;
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.9rem;
    color: #3f3f46;
    line-height: 1.75;
    font-style: italic;
}
.desc-block {
    font-size: 0.875rem;
    color: #71717a;
    line-height: 1.8;
}
.kelas-pill {
    display: inline-block;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.78rem;
    color: #52525b;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

/* ---- Sidebar nav ---- */
.nav-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 0.4rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #e4e4e7;
    margin: 0.8rem 0;
}

/* ---- Streamlit widget cleanup ---- */
.stButton > button {
    background: #09090b;
    color: #fafafa;
    border: none;
    border-radius: 7px;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.55rem 1.2rem;
    transition: background 0.15s;
}
.stButton > button:hover { background: #27272a; }
.stButton > button[kind="secondary"] {
    background: #ffffff;
    color: #52525b;
    border: 1px solid #e4e4e7;
}
.stButton > button[kind="secondary"]:hover {
    background: #f4f4f5;
    color: #09090b;
}
.stToggle label span { font-size: 0.875rem !important; }
div[data-testid="stPopover"] button {
    background: #ffffff !important;
    color: #52525b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 7px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.9rem !important;
}
div[data-testid="stPopover"] button:hover {
    background: #f4f4f5 !important;
    color: #09090b !important;
}
.stSlider label { font-size: 0.82rem !important; }
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    color: #09090b !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}
.stExpander {
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
    background: #fafafa !important;
}
div[data-testid="metric-container"] {
    background: #fafafa;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------------------------
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True
if "page" not in st.session_state:
    st.session_state.page = "app"

# ------------------------------------------------------------------
# WELCOME SCREEN
# ------------------------------------------------------------------
if st.session_state.show_welcome:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-badge">Aplikasi Demo Skripsi</div>
        <div class="welcome-title">Klasifikasi Cacat<br>Permukaan Baja</div>
        <div class="welcome-desc">
            Implementasi EfficientNet-B1 dengan Efficient Channel Attention Module
            dan Multiscale Feature Extraction untuk mendeteksi dan mengklasifikasi
            jenis cacat pada permukaan baja secara otomatis.
        </div>
        <hr class="welcome-divider">
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Mulai", use_container_width=True):
            st.session_state.show_welcome = False
            st.rerun()

    st.stop()

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="nav-label">Navigasi</div>', unsafe_allow_html=True)

    nav_choice = st.radio(
        "nav",
        options=["Klasifikasi", "Profil Peneliti"],
        label_visibility="collapsed",
    )

    st.session_state.page = "app" if nav_choice == "Klasifikasi" else "profile"

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if st.session_state.page == "app":
        st.markdown('<div class="nav-label">Model</div>', unsafe_allow_html=True)
        models_folder = st.text_input(
            "Folder model",
            value="models",
            label_visibility="collapsed",
            help=f"Folder berisi file .keras, mis. {MODEL_CFG['filenames'][0]}",
        )

        model_path_preview = find_model_file(models_folder, MODEL_CFG["filenames"])
        with st.expander("Status file model", expanded=True):
            if model_path_preview:
                st.markdown(f"✓ **{MODEL_KEY}**  \n`{model_path_preview}`")
            else:
                st.markdown(
                    f"✗ **{MODEL_KEY}**  \n"
                    f"Tidak ditemukan ({', '.join(MODEL_CFG['filenames'])})"
                )

        if st.button("Reload model", key="reload_btn"):
            load_keras_model.clear()
            st.success("Cache dibersihkan.")

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.caption(f"Input: {INPUT_SHAPE[0]}×{INPUT_SHAPE[1]} px — auto resize")
    else:
        models_folder = "models"

# ==================================================================
# HALAMAN: PROFIL PENELITI
# ==================================================================
if st.session_state.page == "profile":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-label">Skripsi</div>
        <div class="page-header-title">Profil Peneliti</div>
        <div class="page-header-sub">Informasi peneliti dan deskripsi proyek</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.8], gap="large")

    with col_left:
        st.markdown(f"""
        <div class="card">
            <div class="profile-initials">KK</div>
            <div class="profile-name">{RESEARCHER_INFO['nama']}</div>
            <div class="profile-nim">{RESEARCHER_INFO['nim']}</div>
            <div class="meta-row">
                <span class="meta-label">Program Studi</span>
                <span class="meta-value">{RESEARCHER_INFO['prodi']}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Pembimbing</span>
                <span class="meta-value">{RESEARCHER_INFO['pembimbing']}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Model</span>
                <span class="meta-value">EfficientNet-B1 + ECA + Multiscale</span>
            </div>
            <div class="meta-row">
                <span class="meta-label">Kelas Output</span>
                <span class="meta-value">4 kelas cacat baja</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card-title">Judul Penelitian</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="judul-block">{RESEARCHER_INFO["judul"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card-title">Deskripsi</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="desc-block">{RESEARCHER_INFO["deskripsi"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="card-title">Kelas Klasifikasi</div>', unsafe_allow_html=True)
        kelas_list = ["Pitting", "Crazing", "Scratches", "Inclusion / Others"]
        pills_html = "".join(
            f'<span class="kelas-pill">Kelas {i+1} — {k}</span>'
            for i, k in enumerate(kelas_list)
        )
        st.markdown(pills_html, unsafe_allow_html=True)

    st.stop()

# ==================================================================
# HALAMAN: KLASIFIKASI
# ==================================================================
st.markdown(f"""
<div class="page-header">
    <div class="page-header-label">{MODEL_KEY}</div>
    <div class="page-header-title">Klasifikasi Cacat Permukaan Baja</div>
    <div class="page-header-sub">{MODEL_CFG['desc']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# OOD FILTER CONTROLS — toggle + popover minimalis
# ------------------------------------------------------------------
col_toggle, col_settings, col_spacer = st.columns([1.2, 1, 4])

with col_toggle:
    apply_ood_filter = st.toggle(
        "Filter OOD",
        value=True,
        help=(
            "Menyaring gambar yang bukan foto permukaan baja sebelum masuk ke model. "
            "Nonaktifkan jika gambar baja valid tapi terus ditolak."
        ),
    )

confidence_threshold = CONFIDENCE_THRESHOLD
saturation_threshold = SATURATION_THRESHOLD
min_texture_std = MIN_TEXTURE_STD

with col_settings:
    with st.popover("Threshold"):
        st.markdown('<div class="card-title">Pengaturan Threshold</div>', unsafe_allow_html=True)
        st.caption("Nilai default belum dikalibrasi. Jalankan `calibrate_ood_thresholds.py` untuk hasil optimal.")
        confidence_threshold = st.slider(
            "Confidence minimum",
            0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05,
            key="thresh_conf",
            help="Di bawah nilai ini, prediksi ditandai tidak yakin.",
        )
        saturation_threshold = st.slider(
            "Saturasi warna maksimum",
            0.0, 1.0, SATURATION_THRESHOLD, 0.05,
            key="thresh_sat",
            help="Di atas nilai ini, gambar dianggap terlalu berwarna untuk foto baja.",
        )
        min_texture_std = st.slider(
            "Tekstur minimum (std grayscale)",
            0.0, 30.0, MIN_TEXTURE_STD, 1.0,
            key="thresh_tex",
            help="Di bawah nilai ini, gambar dianggap terlalu flat/blur.",
        )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# UPLOAD GAMBAR
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Unggah gambar permukaan baja",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

col_img, col_result = st.columns([1, 1.4])

predict_btn = False
pil_img = None

if uploaded_file is not None:
    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    with col_img:
        st.image(pil_img, caption="Gambar diunggah", use_container_width=True)
        predict_btn = st.button("Prediksi", type="primary", use_container_width=True)
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
                f"File model **{MODEL_KEY}** tidak ditemukan di folder `{models_folder}`.\n\n"
                f"Nama yang dicari: {MODEL_CFG['filenames']}.\n\n"
                "Pastikan file `.keras` sudah diletakkan di folder tersebut, "
                "atau ubah path di sidebar."
            )
        else:
            try:
                with st.spinner(f"Memuat {MODEL_KEY}..."):
                    model = load_keras_model(model_path)
                with st.spinner("Menjalankan prediksi..."):
                    result = predict(
                        model, pil_img,
                        apply_heuristic_filter=apply_ood_filter,
                        confidence_threshold=confidence_threshold,
                        saturation_threshold=saturation_threshold,
                        min_texture_std=min_texture_std,
                    )

                st.markdown(f"**Hasil — {MODEL_KEY}**")

                if result['status'] == 'rejected_heuristic':
                    st.warning(
                        f"Gambar tidak dikenali sebagai permukaan baja.\n\n"
                        f"{result['reason']}\n\n"
                        "Model tidak dijalankan. Jika ini foto baja yang valid, "
                        "nonaktifkan filter atau longgarkan threshold."
                    )
                elif result['status'] == 'low_confidence':
                    pred_idx, probs = result['pred_idx'], result['probs']
                    st.warning(
                        f"Model tidak cukup yakin.\n\n{result['reason']}"
                    )
                    with st.expander("Lihat tebakan mentah model (tidak direkomendasikan)"):
                        st.metric(
                            "Prediksi (tidak yakin)",
                            CLASS_LABELS[pred_idx],
                            f"{probs[pred_idx] * 100:.2f}%",
                        )
                        df = pd.DataFrame({"Kelas": CLASS_LABELS, "Probabilitas": probs}).set_index("Kelas")
                        st.bar_chart(df)
                        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
                else:
                    pred_idx, probs = result['pred_idx'], result['probs']
                    st.metric(
                        "Prediksi kelas",
                        CLASS_LABELS[pred_idx],
                        f"{probs[pred_idx] * 100:.2f}% confidence",
                    )
                    df = pd.DataFrame({"Kelas": CLASS_LABELS, "Probabilitas": probs}).set_index("Kelas")
                    st.bar_chart(df)
                    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

                with st.expander("Statistik gambar (debug)"):
                    st.json(result['ood_stats'])

            except Exception as e:
                st.error(
                    f"Gagal memuat atau menjalankan model **{MODEL_KEY}**.\n\n"
                    f"Error: `{e}`\n\n"
                    "Penyebab umum: versi TensorFlow/Keras berbeda antara "
                    "environment lokal dan Kaggle. Samakan versi `tf.__version__`."
                )
