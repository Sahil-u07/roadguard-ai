"""RoadGuard AI - Streamlit demo. Run with: streamlit run app.py"""
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from detector import CONF, VIDEO_CONF, WEIGHTS, Tracker, count, load_model, pothole_sizes, predict

# Streamlit Community Cloud runs apps from /mount/src. Its free server throttles apps that use too much CPU,
# so there we keep videos to the first minute and at most Full HD. On a laptop the full video keeps its own size.
ON_CLOUD = Path("/mount/src").exists()
MAX_WIDTH = 1920 if ON_CLOUD else None
MAX_SECONDS = 60 if ON_CLOUD else None
# Potholes in a wide road video are small, so videos run the model on a bigger image than photos.
# On frames from our demo road video, 960 px found about 3.5x as many boxes as 416 px.
VIDEO_IMGSZ = 640 if ON_CLOUD else 960

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


def video_writer(path, fps, width, height):
    """Returns write(frame) and close() -> True if the file plays in a browser.
    With ffmpeg, frames go straight into one high-quality H.264 encode. Without it, OpenCV writes mp4v,
    which is lower quality and which browsers cannot play, but it can still be downloaded."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 needs even sizes
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return lambda frame: proc.stdin.write(frame.tobytes()), lambda: proc.stdin.close() or proc.wait() == 0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    return writer.write, lambda: writer.release() or False


st.html(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")
st.html("""<div class="rg-hero" role="img" aria-label="RoadGuard AI: a road with a detected pothole">
  <div class="rg-road"></div><div class="rg-pothole"></div>
  <div class="rg-hero-text"><h1>RoadGuard AI</h1>
    <p>Finds road damage in photos and videos, so it can be reported and repaired early.</p>
    <span class="rg-chip">YOLOv8</span><span class="rg-chip">ByteTrack tracking</span><span class="rg-chip">Pothole + Crack</span>
  </div></div>""")

if not WEIGHTS.exists():
    st.error("No trained model at weights/best.pt yet. Train one with train.ipynb and commit it.")
    st.stop()

model = load_model()
with st.sidebar:
    st.subheader(":material/tune: Settings")
    conf = st.slider("Photo confidence threshold", 0.05, 0.95, CONF, 0.05,
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
    video_conf = st.slider("Video confidence threshold", 0.05, 0.95, VIDEO_CONF, 0.05,
                           help="Lower finds more potholes. Videos can go lower than photos, because a box is only "
                                "shown once it is found in two model runs in a row.")
    rate = st.slider("Model runs per second of video", 2, 30, 6 if ON_CLOUD else 10,
                     help="Higher is more thorough but slower. Boxes are kept on the frames in between, "
                          "so the output video stays smooth at its own frame rate.")
    if ON_CLOUD:
        st.caption(f"On this free online server only the first {MAX_SECONDS} seconds are processed, at up to Full HD. "
                   "Run the app on a laptop for full-length videos at their original size.")
    if upload and st.button("Run detection", type="primary"):
        # A temp folder, not NamedTemporaryFile: on Windows OpenCV cannot open a file Python still holds open.
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / ("input" + Path(upload.name).suffix), Path(tmp) / "output.mp4"
            src.write_bytes(upload.read())
            cap = cv2.VideoCapture(str(src))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            stride = max(1, round(fps / rate))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            limit = round(MAX_SECONDS * fps) if MAX_SECONDS else None
            if limit and total > limit:
                total = limit
            progress, view = st.progress(0.0, "Starting..."), st.empty()
            seen, peak, result, writer, i, tracker = Counter(), Counter(), None, None, 0, Tracker(video_conf, imgsz=VIDEO_IMGSZ)
            while True:
                ok, frame = cap.read()
                if not ok or (limit and i >= limit):
                    break
                if MAX_WIDTH and frame.shape[1] > MAX_WIDTH:
                    frame = cv2.resize(frame, (MAX_WIDTH, round(frame.shape[0] * MAX_WIDTH / frame.shape[1])))
                if i % stride == 0:
                    result = tracker(model, frame)
                    for name, n in count(result).items():
                        seen[name] += stride
                        peak[name] = max(peak[name], n)
                else:
                    result.orig_img = frame  # keep the last boxes on this frame so the output does not flicker
                annotated = result.plot(conf=False)  # label is the pothole's id and class; tracked scores can dip
                if writer is None:
                    write, close = writer = video_writer(out, fps, annotated.shape[1], annotated.shape[0])
                write(annotated)
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
            playable = close()
            video = out.read_bytes()
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
