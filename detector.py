"""Detection code shared by the Streamlit and Flask apps."""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

WEIGHTS = Path(__file__).parent / "weights" / "best.pt"
# Best F1 on the validation split was at 0.44. At the old 0.25 only about half the boxes on the test split were right.
CONF = 0.45


def load_model(path=WEIGHTS):
    return YOLO(str(path))


def predict(model, frame, conf=CONF):
    """Run YOLOv8 on one BGR frame and return the Ultralytics result."""
    return model.predict(frame, conf=conf, verbose=False)[0]


def count(result):
    """Number of detections per class name."""
    return Counter(result.names[int(c)] for c in result.boxes.cls)


def detect(model, frame, conf=CONF):
    """Run YOLOv8 on one BGR frame. Returns the annotated frame and a count of detections per class."""
    result = predict(model, frame, conf)
    return result.plot(), count(result)


def steady(result, previous):
    """Keep only boxes that overlap a same-class box from the previous model run on the video.
    A box that shows up for a single frame is almost always a false alarm."""
    if previous is None or not len(previous.boxes) or not len(result.boxes):
        return result[:0]
    same = result.boxes.cls[:, None] == previous.boxes.cls[None, :]
    return result[((box_iou(result.boxes.xyxy, previous.boxes.xyxy) > 0.2) & same).any(1)]


def pothole_sizes(result, ground_width_m):
    """Rough width and length of each pothole box in cm, given how many metres of ground the photo covers across.
    Only meaningful for a photo taken looking down at the road; depth cannot be measured from a single photo."""
    cm_per_px = ground_width_m * 100 / result.orig_shape[1]
    return [{"Confidence": round(float(c), 2), "Width (cm)": round(float(x2 - x1) * cm_per_px),
             "Length (cm)": round(float(y2 - y1) * cm_per_px)}
            for (x1, y1, x2, y2), c, k in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
            if result.names[int(k)] == "Pothole"]
