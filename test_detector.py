import numpy as np
import torch
from ultralytics.engine.results import Results

from detector import pothole_sizes, steady

NAMES = {0: "Pothole", 1: "Crack"}


def result(*boxes):
    """A fake 1000x500 detection result; each box is (x1, y1, x2, y2, conf, class)."""
    return Results(np.zeros((500, 1000, 3), np.uint8), "road.jpg", NAMES, boxes=torch.tensor(boxes).reshape(-1, 6))


def test_steady_keeps_boxes_seen_in_the_previous_run():
    previous = result((100, 100, 200, 200, 0.9, 0), (600, 100, 700, 200, 0.9, 1))
    now = result((105, 105, 205, 205, 0.8, 0),   # same pothole, moved a little: kept
                 (800, 300, 900, 400, 0.8, 0),   # new box in empty road: dropped
                 (600, 100, 700, 200, 0.8, 0))   # overlaps a crack, but it is a pothole: dropped
    assert steady(now, previous).boxes.xyxy.tolist() == [[105, 105, 205, 205]]


def test_steady_drops_everything_without_a_previous_run():
    assert len(steady(result((1, 1, 50, 50, 0.9, 0)), None).boxes) == 0


def test_pothole_sizes_uses_the_ground_width():
    # 1000 px across = 2 m, so 1 px = 0.2 cm. Cracks are skipped.
    sizes = pothole_sizes(result((0, 0, 250, 100, 0.66, 0), (0, 0, 10, 10, 0.9, 1)), 2.0)
    assert sizes == [{"Confidence": 0.66, "Width (cm)": 50, "Length (cm)": 20}]
