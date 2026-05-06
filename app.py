import streamlit as st
import sys
import os
import tempfile
import shutil
import pandas as pd
import json
import torch
import pathlib
import platform
import numpy as np
from io import BytesIO

# -- Por si se corre en Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


# ── Asegurar que yolov5 esté en el path ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'yolov5'))

from speciesnet import SpeciesNet
from megadetector.detection import run_detector
from yolo_detector import yolo_inference
from inference_functions import species_net_to_cupybara, final_predict

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoBio – Clasificador de Fauna",
    page_icon="🦌",
    layout="centered"
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 780px; }
    .stProgress > div > div { background-color: #2e7d32; }
    .result-table { font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# ── Carga de modelos (se hace UNA sola vez con cache) ────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_resource(show_spinner="Cargando modelos… esto tarda solo la primera vez.")
def load_models():
    md_path   = os.path.join(MODELS_DIR, 'md_v5a.0.0.pt')
    cls_path  = os.path.join(MODELS_DIR, 'always_crop_99710272_22x8_v12_epoch_00148.pt')
    lbl_path  = os.path.join(MODELS_DIR, 'always_crop_99710272_22x8_v12_epoch_00148.labels.txt')
    dict_path = os.path.join(MODELS_DIR, 'species_dict.json')

    megadetector = run_detector.load_detector(md_path)
    species_net  = SpeciesNet(MODELS_DIR)
    classifier   = torch.load(cls_path, weights_only=False)
    classifier.eval()

    with open(lbl_path, encoding='utf-8') as f:
        labels = {idx: line.strip() for idx, line in enumerate(f.readlines())}

    with open(dict_path, 'r') as f:
        species_dict = json.load(f)

    return megadetector, species_net, classifier, labels, species_dict

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🦌 EcoBio – Clasificador de Fauna")
st.caption("Detección y clasificación de fauna uruguaya en videos de cámara trampa.")
st.divider()

# Parámetros opcionales
with st.expander("⚙️ Parámetros avanzados", expanded=False):
    steps = st.slider(
        "Frames a analizar por video (steps)",
        min_value=5, max_value=60, value=30,
        help="Cuántos frames se muestrean a lo largo del video. Más frames = más precisión pero más lento."
    )
    find_n_frames = st.slider(
        "Frames con objetos para detener búsqueda",
        min_value=1, max_value=15, value=10,
        help="Cuántos frames con detecciones se necesitan antes de parar de buscar."
    )
    umbral_confianza = st.slider(
        "Umbral mínimo de confianza para mostrar predicción (%)",
        min_value=0, max_value=100, value=50,
        help="Solo se muestran predicciones de SpeciesNet con confianza mayor a este valor."
    ) / 100.0

# Upload de videos
uploaded_files = st.file_uploader(
    "📁 Subí los videos a clasificar",
    type=["avi", "mp4", "AVI", "MP4"],
    accept_multiple_files=True,
    help="Formatos soportados: .AVI y .MP4"
)

if uploaded_files:
    st.info(f"**{len(uploaded_files)} video(s)** cargado(s) y listo(s) para procesar.")

    if st.button("🚀 Iniciar clasificación", type="primary", use_container_width=True):

        # Cargar modelos
        megadetector, species_net, classifier, labels, species_dict = load_models()

        # Crear directorio temporal de trabajo
        work_dir     = tempfile.mkdtemp(prefix="ecobio_")
        os.chdir(work_dir)
        videos_dir   = os.path.join(work_dir, 'videos')
        cropped_dir  = os.path.join(work_dir, 'cropped_images')
        os.makedirs(videos_dir)
        os.makedirs(cropped_dir)

        # Guardar videos subidos en el directorio temporal
        for uf in uploaded_files:
            dest = os.path.join(videos_dir, uf.name)
            with open(dest, 'wb') as f:
                f.write(uf.read())

        # ── Procesar cada video ───────────────────────────────────────────────
        resultados = []
        progress   = st.progress(0, text="Iniciando…")
        status_txt = st.empty()

        total = len(uploaded_files)

        for idx, uf in enumerate(uploaded_files):
            video_name = uf.name
            status_txt.markdown(f"⏳ Procesando **{video_name}** ({idx+1}/{total})…")
            progress.progress((idx) / total, text=f"Video {idx+1} de {total}")

            try:
                detector = yolo_inference(
                    detector_model   = megadetector,
                    classifier_model = species_net,
                    file_name        = video_name,
                    steps            = steps,
                    find_n_frames    = find_n_frames,
                    videos_path      = videos_dir,
                    save_dir_path    = cropped_dir,
                )
                yolo_metadata = detector.detect_and_predict_image()
                video_predictions = yolo_metadata.get('video_predictions', {})

                if len(video_predictions) == 0:
                    resultados.append({
                        'Archivo':    video_name,
                        'Predicción': 'sin_objeto',
                        'Confianza %': '100%'
                    })
                else:
                   for pred_class in video_predictions.keys():
                        conf = video_predictions.get(pred_class)
                        if conf > umbral_confianza: 
                            resultados.append({
                                    'Archivo':    video_name,
                                    'Predicción': pred_class,
                                    'Confianza %': f"{conf:.1%}"
                                })

            except Exception as e:
                resultados.append({
                    'Archivo':    video_name,
                    'Predicción': f'ERROR: {str(e)}',
                    'Confianza %': '–'
                })

            progress.progress((idx + 1) / total, text=f"Video {idx+1} de {total} completado")

        # Limpiar directorio temporal
        shutil.rmtree(work_dir, ignore_errors=True)

        status_txt.markdown("✅ **Procesamiento completado.**")
        progress.progress(1.0, text="¡Listo!")

        # ── Mostrar resultados ────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Resultados")

        df = pd.DataFrame(resultados)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Resumen
        col1, col2, col3 = st.columns(3)
        col1.metric("Videos procesados", df['Archivo'].nunique())
        col2.metric("Con detección",      df[df['Predicción'] != 'sin_objeto']['Archivo'].nunique())
        col3.metric("Sin objeto",          df[df['Predicción'] == 'sin_objeto']['Archivo'].nunique())

        # ── Descarga Excel ────────────────────────────────────────────────────
        st.divider()
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Clasificaciones')
        buffer.seek(0)

        st.download_button(
            label="⬇️ Descargar planilla Excel",
            data=buffer,
            file_name="clasificaciones_ecobio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )

else:
    st.markdown("""
    ### ¿Cómo usar esta aplicación?

    1. **Subí** uno o más videos (`.AVI` o `.MP4`) usando el botón de arriba.
    2. Opcionalmente ajustá los parámetros avanzados.
    3. Hacé clic en **Iniciar clasificación**.
    4. Cuando termine, **descargá la planilla Excel** con los resultados.

    """)
