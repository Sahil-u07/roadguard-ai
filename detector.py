"""Detection code shared by the Streamlit and Flask apps."""
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

WEIGHTS = Path(__file__).parent / "weights" / "best.pt"
# Best F1 on the validation split was at 0.44. At the old 0.25 only about half the boxes on the test split were right.
CONF = 0.45
# Videos can start from weaker boxes, because a box must also be found again in the next model run to be shown.
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


class Tracker:
    """Follows each detection from one frame to the next (ByteTrack) so boxes in a video stay steady.
    A box is shown only once it has been found in `min_hits` model runs, which removes one-off false alarms.
    Weaker detections (down to 0.1) may keep an already confirmed box alive, so it does not flicker.
    Boxes centred in the top `min_y` of the frame are ignored: with a forward-facing camera that is sky, buildings
    and far-off traffic, not road. On frames from our demo video every box up there was a false alarm."""

    def __init__(self, conf=VIDEO_CONF, min_hits=2, imgsz=416, min_y=0.38):
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        cfg.track_high_thresh = cfg.new_track_thresh = conf  # only confident boxes can start a new track
        self.tracker, self.hits, self.min_hits, self.imgsz, self.min_y = BYTETracker(cfg), Counter(), min_hits, imgsz, min_y

    def __call__(self, model, frame):
        result = predict(model, frame, conf=0.1, imgsz=self.imgsz)
        result = result[result.boxes.xywhn[:, 1] >= self.min_y]
        tracks = self.tracker.update(result.boxes.cpu().numpy(), frame)
        if not len(tracks):
            return result[:0]
        result = result[tracks[:, -1].astype(int)]  # last column: index of the detection each track came from
        result.update(boxes=torch.as_tensor(tracks[:, :-1], dtype=torch.float32))  # smoothed box, track id, score, class
        ids = tracks[:, 4].astype(int)
        self.hits.update(ids)
        return result[torch.tensor([self.hits[i] >= self.min_hits for i in ids])]


def pothole_sizes(result, ground_width_m):
    """Rough width and length of each pothole box in cm, given how many metres of ground the photo covers across.
    Only meaningful for a photo taken looking down at the road; depth cannot be measured from a single photo."""
    cm_per_px = ground_width_m * 100 / result.orig_shape[1]
    return [{"Confidence": round(float(c), 2), "Width (cm)": round(float(x2 - x1) * cm_per_px),
             "Length (cm)": round(float(y2 - y1) * cm_per_px)}
            for (x1, y1, x2, y2), c, k in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
            if result.names[int(k)] == "Pothole"]
