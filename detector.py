"""Detection code shared by the Streamlit and Flask apps."""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "weights" / "best.pt"
# Best F1 on the validation split was at 0.44. At the old 0.25 only about half the boxes on the test split were right.
CONF = 0.45
# Potholes in road videos are small and far away, so the model is less sure of them than in close-up photos.
VIDEO_CONF = 0.25


def load_model(path=WEIGHTS):
    return YOLO(str(path))


def predict(model, frame, conf=CONF, imgsz=416):
    """Run YOLOv8 on one BGR frame and return the Ultralytics result. 416 px is the size the model was trained at."""
    return model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]


def count(result):
    """Number of detections per class name."""
    return Counter(result.names[int(c)] for c in result.boxes.cls)


def detect(model, frame, conf=CONF):
    """Run YOLOv8 on one BGR frame. Returns the annotated frame and a count of detections per class."""
    result = predict(model, frame, conf)
    return result.plot(), count(result)


def road_only(result, min_y=0.38):
    """Drop boxes centred in the top `min_y` of a forward-facing road video frame: that part is sky, buildings and
    far-off traffic, not road. On frames from our demo video every box up there was a false alarm."""
    return result[result.boxes.xywhn[:, 1] >= min_y]


def pothole_sizes(result, ground_width_m):
    """Rough width and length of each pothole box in cm, given how many metres of ground the photo covers across.
    Only meaningful for a photo taken looking down at the road; depth cannot be measured from a single photo."""
    cm_per_px = ground_width_m * 100 / result.orig_shape[1]
    return [{"Confidence": round(float(c), 2), "Width (cm)": round(float(x2 - x1) * cm_per_px),
             "Length (cm)": round(float(y2 - y1) * cm_per_px)}
            for (x1, y1, x2, y2), c, k in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
            if result.names[int(k)] == "Pothole"]
