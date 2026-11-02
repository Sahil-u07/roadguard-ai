import numpy as np
import torch
from ultralytics.engine.results import Results

from detector import draw, pothole_sizes, road_only

NAMES = {0: "Pothole", 1: "Crack"}


def result(*boxes):
    """A fake 1000x500 detection result; each box is (x1, y1, x2, y2, conf, class)."""
    return Results(np.zeros((500, 1000, 3), np.uint8), "road.jpg", NAMES, boxes=torch.tensor(boxes).reshape(-1, 6))


def test_road_only_drops_boxes_in_the_top_of_the_frame():
    road, sky = (100, 300, 200, 400, 0.9, 0), (100, 20, 200, 120, 0.9, 0)  # centres at 70% and 14% of the height
    assert road_only(result(road, sky)).boxes.xyxy.tolist() == [[100, 300, 200, 400]]


def test_pothole_sizes_uses_the_ground_width():
    # 1000 px across = 2 m, so 1 px = 0.2 cm. Cracks are skipped.
    sizes = pothole_sizes(result((0, 0, 250, 100, 0.66, 0), (0, 0, 10, 10, 0.9, 1)), 2.0)
    assert sizes == [{"Confidence": 0.66, "Width (cm)": 50, "Length (cm)": 20}]



def test_draw_puts_a_red_box_and_white_label_on_a_copy():
    r = result((100, 300, 200, 400, 0.62, 0))
    img = draw(r)
    assert (img[350, 100] == [0, 0, 255]).all()  # left edge of the box is red (BGR)
    assert (img[297, 101] == [255, 255, 255]).all()  # label background just above the box is white
    assert not r.orig_img.any()  # the original frame is untouched
