"""Detection code shared by the Streamlit and Flask apps."""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "weights" / "best.pt"


def load_model(path=WEIGHTS):
    return YOLO(str(path))


def detect(model, frame, conf=0.25):
    """Run YOLOv8 on one BGR frame. Returns the annotated frame and a count of detections per class."""
    result = model.predict(frame, conf=conf, verbose=False)[0]
    return result.plot(), Counter(result.names[int(c)] for c in result.boxes.cls)
