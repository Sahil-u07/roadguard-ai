"""RoadGuard AI - Streamlit demo. Run with: streamlit run app.py"""
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import CONF, WEIGHTS, count, load_model, pothole_sizes, predict, steady

# Streamlit Community Cloud runs apps from /mount/src. Its free server throttles apps that use too much CPU,
# so there we keep videos short and small. On a laptop the full video is processed at up to 1280 px.
ON_CLOUD = Path("/mount/src").exists()
MAX_WIDTH = 640 if ON_CLOUD else 1280  # bigger frames are scaled down first; the model sees 416 px anyway
MAX_SECONDS = 60 if ON_CLOUD else None

st.set_page_config(page_title="RoadGuard AI", page_icon=":material/add_road:", layout="wide")
load_model = st.cache_resource(load_model)


def show_counts(counts):
    """One metric per detected class, or a clear 'nothing found' message."""
    if not counts:
        st.success("No anomalies detected.")
        return
    for col, (name, n) in zip(st.columns(len(counts)), sorted(counts.items())):
        col.metric(name, n)


def show_detections(data):
    """Decode uploaded image bytes, run detection and show the original next to the result."""
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read that image.")
        return
    with st.spinner("Detecting..."):
        result = predict(model, frame, conf)
    left, right = st.columns(2)
    left.image(frame, channels="BGR", caption="Original", width="stretch")
    right.image(result.plot(), channels="BGR", caption="Detected", width="stretch")
    show_counts(count(result))
    sizes = pothole_sizes(result, ground_width)
    if sizes:
        st.write(f"Estimated pothole size, if the photo covers about {ground_width} m of road across "
                 "(match each row to a box by its confidence):")
        st.dataframe(sizes, hide_index=True)


def to_h264(src, dst):
    """Browsers cannot play OpenCV's mp4v video, so re-encode it with ffmpeg when ffmpeg is installed."""
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.run(cmd).returncode == 0


st.title(":material/add_road: RoadGuard AI")
st.caption("Finds road damage in photos and videos with a YOLOv8 model, so it can be reported and repaired early.")

if not WEIGHTS.exists():
    st.error("No trained model at weights/best.pt yet. Train one with train.ipynb and commit it.")
    st.stop()

model = load_model()
with st.sidebar:
    st.subheader(":material/tune: Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, CONF, 0.05,
                     help="Only boxes the model is at least this sure about are shown. Raise it if you see false boxes.")
    ground_width = st.number_input("Road width the photo covers (m)", 0.1, 50.0, 2.0, 0.1,
                                   help="Used to turn pothole boxes into centimetres. A phone held at chest height "
                                        "looking straight down covers about 2 m. Angled photos make the estimate less "
                                        "accurate. Depth cannot be measured from a photo.")
    st.subheader(":material/neurology: Model detects")
    for name in model.names.values():
        st.badge(name, icon=":material/warning:", color="orange")
    st.subheader(":material/help: How to use")
    st.markdown("1. Pick a tab.\n2. Upload a road photo or video, or take a photo.\n3. Boxes and counts appear below.")

image_tab, video_tab, camera_tab = st.tabs([":material/image: Image", ":material/movie: Video", ":material/photo_camera: Camera"])

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
    stride = st.slider("Run the model on every Nth frame", 1, 5, 3 if ON_CLOUD else 2,
                       help="1 is the most accurate. Higher is faster; boxes are kept on the frames in between, "
                            "so the output video stays smooth.")
    if ON_CLOUD:
        st.caption(f"On this free online server only the first {MAX_SECONDS} seconds are processed, at {MAX_WIDTH} px. "
                   "Run the app on a laptop for full-length, full-size videos.")
    if upload and st.button("Run detection", type="primary"):
        # A temp folder, not NamedTemporaryFile: on Windows OpenCV cannot open a file Python still holds open.
        with tempfile.TemporaryDirectory() as tmp:
            src, raw, out = (Path(tmp) / n for n in ("input" + Path(upload.name).suffix, "raw.mp4", "output.mp4"))
            src.write_bytes(upload.read())
            cap = cv2.VideoCapture(str(src))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            limit = round(MAX_SECONDS * fps) if MAX_SECONDS else None
            if limit and total > limit:
                total = limit
            progress, view = st.progress(0.0, "Starting..."), st.empty()
            seen, peak, result, previous, writer, i = Counter(), Counter(), None, None, None, 0
            while True:
                ok, frame = cap.read()
                if not ok or (limit and i >= limit):
                    break
                if frame.shape[1] > MAX_WIDTH:
                    frame = cv2.resize(frame, (MAX_WIDTH, round(frame.shape[0] * MAX_WIDTH / frame.shape[1])))
                if i % stride == 0:
                    found = predict(model, frame, conf)
                    result, previous = steady(found, previous), found  # drop boxes that last only one run
                    for name, n in count(result).items():
                        seen[name] += stride
                        peak[name] = max(peak[name], n)
                else:
                    result.orig_img = frame  # keep the last boxes on this frame so the output does not flicker
                annotated = result.plot()
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(annotated)
                if i % (stride * 5) == 0:  # refreshing the preview on every frame slows the page down
                    view.image(annotated, channels="BGR", width="stretch")
                    if total > 0:
                        progress.progress(min(i / total, 1.0), f"Frame {i} of {total}")
                i += 1
            cap.release()
            if writer is None:
                progress.empty()
                st.error("Could not read that video.")
                st.stop()
            writer.release()
            playable = to_h264(raw, out)
            video = (out if playable else raw).read_bytes()
        progress.progress(1.0, f"Done: {i} frames")
        view.empty()
        if playable:
            st.video(video)
        else:
            st.info("Install ffmpeg to play the result here. You can still download it below.")
        if seen:
            st.write("Seconds of video in which each class was seen, and the most seen in one frame:")
            for col, name in zip(st.columns(len(seen)), sorted(seen)):
                col.metric(f"{name} (up to {peak[name]} in one frame)", f"{seen[name] / fps:.1f} s")
        else:
            st.success("No anomalies detected.")
        st.download_button("Download annotated video", video, "roadguard_output.mp4", "video/mp4")
