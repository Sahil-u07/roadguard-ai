# RoadGuard AI

Real-time road anomaly detection with YOLOv8. RoadGuard AI finds potholes, cracks and speed bumps in road images and video, so that road damage can be reported and repaired before it causes accidents.

PBL-III (AIM3170) project, Department of AIML, Manipal University Jaipur, 2026.

**Live demo:** _coming soon (Streamlit Cloud)_

## Status (November 2026)

| Part | Status |
|---|---|
| Dataset merge pipeline (YOLO + Pascal-VOC sources, class remapping, splitting) | Done, tested |
| Streamlit app (image upload, video upload with annotated video download, camera photo) | Done |
| Training and evaluation notebook (Kaggle/Colab GPU) | Done |
| Pilot model on a public crack dataset (CPU) | Done, see [Pilot results](#pilot-results) |
| Pothole + crack model (CPU), used by the app now | Done, see [Current model](#current-model) |
| Full training on the merged dataset | In progress |
| Flask interface (image upload) | Done, tested |

## How it works

1. **Merge.** Every dataset names its classes differently (`pothole`, `Potholes`, `D40`...). `merge_datasets.py` maps each source's classes onto one list (`Pothole`, `Crack`, `Severe-Crack`, `Speed-Bump`), converts Pascal-VOC boxes and segmentation polygons to YOLO boxes, prefixes file names with the source name so nothing is overwritten, and splits sources that have no split 70/20/10. Any class you forget to map stops the merge with an error instead of being silently dropped.
2. **Train.** `train.ipynb` fine-tunes a COCO-pretrained YOLOv8 on the merged data (transfer learning), then evaluates `best.pt` on the held-out test split and saves the metrics and confusion matrix.
3. **Detect.** `detector.py` loads `weights/best.pt` and runs it on a frame. Both front ends use it: `app.py` (Streamlit: image, video, camera) and `flask_app.py` (Flask: image upload).

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py      # Streamlit app on http://localhost:8501
python flask_app.py       # Flask app on http://localhost:5000
```

The app needs a trained model at `weights/best.pt`.

## Build the dataset and train

1. Download the datasets listed under [Datasets](#datasets) and unzip each into `raw/` using the folder names in `sources.yaml`.
2. Fill in each source's class `map` in `sources.yaml`.
3. Merge: `python merge_datasets.py sources.yaml datasets/roadguard` (prints images and boxes per class and split).
4. Open `train.ipynb` on Kaggle or Colab with a GPU and run all cells.
5. The notebook copies the best model to `weights/best.pt` and zips the training and test results. Commit `weights/best.pt`.

Tests: `pip install pytest && pytest`

## Current model

`weights/best.pt` is a YOLOv8n that detects **Pothole** and **Crack**. We merged the Ultralytics Crack-seg dataset and a public pothole dataset ([michelpf/dataset-pothole](https://github.com/michelpf/dataset-pothole), originally from Kaggle) with `merge_datasets.py`, then trained on a CPU for 30 epochs at 416 px (about 4.9 hours, 5,045 training images). The pothole validation split is grouped by photo, so Roboflow-augmented copies of one photo never land in both train and valid. Test split (533 images, 1,118 objects):

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| All | 0.752 | 0.675 | 0.714 | 0.461 |
| Pothole | 0.741 | 0.682 | 0.729 | 0.423 |
| Crack | 0.763 | 0.669 | 0.698 | 0.499 |

F1 is 0.712. On the same crack test images the pilot model below scored mAP@0.5 0.606, so this model is better at cracks too. Inference took about 29 ms per image on a 4-core CPU. Speed bumps and severe cracks need the full run of `train.ipynb` on a GPU.

## Pilot results

To test the whole pipeline before training on the merged data, we passed the public [Ultralytics Crack-seg](https://docs.ultralytics.com/datasets/segment/crack-seg/) dataset through `merge_datasets.py` and fine-tuned YOLOv8n on a CPU: 15 epochs, 1,115 training images (30%), 416 px, about 20 minutes. Test split (112 images, 148 cracks):

| Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 0.686 | 0.620 | 0.651 | 0.606 | 0.392 |

This was a crack-only pipeline check. It has been replaced by the [current model](#current-model).

## Deploy to Streamlit Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Create app → this repo, branch `main`, main file `app.py`, Python 3.11 (under Advanced settings).
3. Deploy. `requirements.txt` installs the CPU build of PyTorch and `packages.txt` installs the system libraries OpenCV needs (Ultralytics installs OpenCV itself).

## Project structure

```
app.py                  Streamlit app (image, video, camera)
flask_app.py            Flask app (image upload)
detector.py             model loading and detection shared by both apps
test_flask_app.py       tests for the Flask app
merge_datasets.py       dataset merging and label conversion
test_merge_datasets.py  tests for the merger
sources.yaml            which datasets to merge and how their classes map
train.ipynb             GPU training and evaluation
weights/best.pt         model used by the app (Pothole + Crack)
```

## Datasets

- RAD: Road Anomaly Detection dataset, Mendeley Data. https://data.mendeley.com/datasets/8chk8vdn2z/2
- HighRPD: a high-altitude drone dataset of road pavement distress, Data in Brief (2025). https://www.sciencedirect.com/science/article/pii/S235234092500109X
- Indian Roads: YOLO V8 Indian roads dataset, Roboflow Universe. https://universe.roboflow.com/object-detection-dp5wa/yolo-v8-indian-roads-dataset
- Humps/Bumps/Potholes: Humps/Bumps & Potholes Detection, Roboflow Universe. https://universe.roboflow.com/detection-system/humps-bumps-potholes-detection
- Pothole dataset: michelpf/dataset-pothole on GitHub, from the Kaggle dataset by Anugrah Akbar Praramadhan. https://github.com/michelpf/dataset-pothole
- Crack-seg (pilot run and current model): Ultralytics. https://docs.ultralytics.com/datasets/segment/crack-seg/

Each dataset keeps its own license. Check it before redistributing images.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io), [OpenCV](https://opencv.org)



Supervisor: Ms. Akanksha Mrinali

## License

AGPL-3.0, the same license as Ultralytics YOLOv8, which this project builds on.
