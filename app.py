"""RoadGuard AI - Streamlit demo. Run with: streamlit run app.py"""
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import WEIGHTS, detect, load_model

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
load_model = st.cache_resource(load_model)


def show_detections(data):
    """Decode uploaded image bytes, run detection and show the result."""
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read that image.")
        return
    annotated, counts = detect(model, frame, conf)
    st.image(annotated, channels="BGR")
    st.write(dict(counts) if counts else "No anomalies detected.")


st.title("RoadGuard AI")
st.caption("Real-time road anomaly detection with YOLOv8")

if not WEIGHTS.exists():
    st.error("No trained model at weights/best.pt yet. Train one with train.ipynb and commit it.")
    st.stop()

model = load_model()
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.write("Classes:", ", ".join(model.names.values()))

image_tab, video_tab, camera_tab = st.tabs(["Image", "Video", "Camera"])

with image_tab:
    upload = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if upload:
        show_detections(upload.read())

with camera_tab:
    shot = st.camera_input("Take a photo of the road")
    if shot:
        show_detections(shot.getvalue())

with video_tab:
    upload = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"])
    stride = st.slider("Process every Nth frame", 1, 10, 3, help="Higher is faster on CPU")
    if upload and st.button("Run detection"):
        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.read())
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            view, totals, i = st.empty(), Counter(), 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    annotated, counts = detect(model, frame, conf)
                    totals += counts
                    view.image(annotated, channels="BGR")
                i += 1
            cap.release()
        st.write("Detections across processed frames:", dict(totals) if totals else "none")
