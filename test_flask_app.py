import io

import cv2
import numpy as np

from flask_app import app

client = app.test_client()


def test_home_page_loads():
    assert b"RoadGuard AI" in client.get("/").data


def test_rejects_non_image():
    r = client.post("/", data={"image": (io.BytesIO(b"not an image"), "x.jpg")})
    assert r.status_code == 400 and b"Please upload" in r.data


def test_detects_on_real_image():
    jpg = cv2.imencode(".jpg", np.full((320, 320, 3), 127, np.uint8))[1].tobytes()
    r = client.post("/", data={"image": (io.BytesIO(jpg), "road.jpg"), "conf": "0.5"})
    assert r.status_code == 200 and b"data:image/jpeg;base64," in r.data
