import numpy as np
import torch
from ultralytics.engine.results import Results

from detector import Tracker, pothole_sizes

NAMES = {0: "Pothole", 1: "Crack"}


def result(*boxes):
    """A fake 1000x500 detection result; each box is (x1, y1, x2, y2, conf, class)."""
    return Results(np.zeros((500, 1000, 3), np.uint8), "road.jpg", NAMES, boxes=torch.tensor(boxes).reshape(-1, 6))


class FakeModel:
    """Returns the given detections, one list per call, like YOLO.predict would."""

    def __init__(self, *runs):
        self.runs = list(runs)

    def predict(self, frame, conf, verbose):
        return [result(*self.runs.pop(0))]


def test_tracker_shows_a_pothole_only_after_three_runs_and_drops_one_offs():
    pothole = (100, 100, 200, 200, 0.9, 0)
    model = FakeModel([pothole], [pothole, (800, 300, 900, 400, 0.9, 0)], [pothole], [pothole])
    track, frame = Tracker(), np.zeros((500, 1000, 3), np.uint8)
    shown = [len(track(model, frame).boxes) for _ in range(4)]
    assert shown == [0, 0, 1, 1]  # the box seen in one run only never appears


def test_tracker_keeps_a_confirmed_pothole_when_its_score_dips():
    model = FakeModel(*[[(100, 100, 200, 200, 0.9, 0)]] * 3, [(102, 102, 202, 202, 0.2, 0)])
    track, frame = Tracker(), np.zeros((500, 1000, 3), np.uint8)
    for _ in range(3):
        track(model, frame)
    assert len(track(model, frame).boxes) == 1


def test_pothole_sizes_uses_the_ground_width():
    # 1000 px across = 2 m, so 1 px = 0.2 cm. Cracks are skipped.
    sizes = pothole_sizes(result((0, 0, 250, 100, 0.66, 0), (0, 0, 10, 10, 0.9, 1)), 2.0)
    assert sizes == [{"Confidence": 0.66, "Width (cm)": 50, "Length (cm)": 20}]
