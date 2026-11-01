"""RoadGuard AI - Flask interface. Run with: python flask_app.py, then open http://localhost:5000"""
import base64

import cv2
import numpy as np
from flask import Flask, render_template_string, request

from detector import CONF, WEIGHTS, detect, load_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # Flask rejects uploads over 10 MB with 413
model = load_model() if WEIGHTS.exists() else None

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoadGuard AI</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1f2124; }
    form { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; padding: 1rem; background: #f2f3f5; border-radius: 8px; }
    img { max-width: 100%; border-radius: 8px; margin-top: 1rem; }
    .error { color: #b00020; }
  </style>
</head>
<body>
  <h1>RoadGuard AI</h1>
  <p>Upload a road image to detect potholes, cracks and speed bumps.</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/png, image/jpeg" required>
    <label>Confidence <input type="number" name="conf" min="0.05" max="0.95" step="0.05" value="{{ conf }}"></label>
    <button type="submit">Detect</button>
  </form>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
  {% if image %}
    <img src="data:image/jpeg;base64,{{ image }}" alt="Road image with detected anomalies">
    <p>{% if counts %}{% for name, n in counts.items() %}{{ name }}: {{ n }}{% if not loop.last %}, {% endif %}{% endfor %}{% else %}No anomalies detected.{% endif %}</p>
  {% endif %}
</body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, conf=CONF)
    if model is None:
        return render_template_string(PAGE, conf=CONF, error="No trained model at weights/best.pt yet."), 503
    try:
        conf = min(max(float(request.form.get("conf", CONF)), 0.05), 0.95)
    except ValueError:
        conf = CONF
    upload = request.files.get("image")
    frame = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR) if upload else None
    if frame is None:
        return render_template_string(PAGE, conf=conf, error="Please upload a JPG or PNG image."), 400
    annotated, counts = detect(model, frame, conf)
    image = base64.b64encode(cv2.imencode(".jpg", annotated)[1]).decode()
    return render_template_string(PAGE, conf=conf, image=image, counts=counts)


if __name__ == "__main__":
    app.run(debug=False)
