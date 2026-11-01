"""Detection code shared by the Streamlit and Flask apps."""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "weights" / "best.pt"


def load_model(path=WEIGHTS):
    return YOLO(str(path))


def predict(model, frame, conf=0.25):
    """Run YOLOv8 on one BGR frame and return the Ultralytics result."""
    return model.predict(frame, conf=conf, verbose=False)[0]


def count(result):
    """Number of detections per class name."""
    return Counter(result.names[int(c)] for c in result.boxes.cls)


def detect(model, frame, conf=0.25):
    """Run YOLOv8 on one BGR frame. Returns the annotated frame and a count of detections per class."""
    result = predict(model, frame, conf)
    return result.plot(), count(result)
