# RoadGuard AI

Real-time road anomaly detection with YOLOv8. RoadGuard AI finds potholes, cracks and speed bumps in road images and video, so that road damage can be reported and repaired before it causes accidents.

PBL-III (AIM3170) project, Department of AIML, Manipal University Jaipur, 2026.

**Live demo:** _coming soon (Streamlit Cloud)_

## Status (mid-term, October 2026)

| Part | Status |
|---|---|
| Dataset merge pipeline (YOLO + Pascal-VOC sources, class remapping, splitting) | Done, tested |
| Streamlit app (image and video upload) | Done |
| Training and evaluation notebook (Kaggle/Colab GPU) | Done |
| Pilot model on a public crack dataset (CPU) | Done, see [Pilot results](#pilot-results) |
| Full training on the merged dataset | In progress |
| Flask interface, live webcam stream | Planned for end-term |

## How it works

1. **Merge.** Every dataset names its classes differently (`pothole`, `Potholes`, `D40`...). `merge_datasets.py` maps each source's classes onto one list (`Pothole`, `Crack`, `Severe-Crack`, `Speed-Bump`), converts Pascal-VOC boxes and segmentation polygons to YOLO boxes, prefixes file names with the source name so nothing is overwritten, and splits sources that have no split 70/20/10. Any class you forget to map stops the merge with an error instead of being silently dropped.
2. **Train.** `train.ipynb` fine-tunes a COCO-pretrained YOLOv8 on the merged data (transfer learning), then evaluates `best.pt` on the held-out test split and saves the metrics and confusion matrix.
3. **Detect.** `app.py` loads `weights/best.pt` and runs it on uploaded images or videos.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app needs a trained model at `weights/best.pt`.

## Build the dataset and train

1. Download the datasets listed under [Datasets](#datasets) and unzip each into `raw/` using the folder names in `sources.yaml`.
2. Fill in each source's class `map` in `sources.yaml`.
3. Merge: `python merge_datasets.py sources.yaml datasets/roadguard` (prints images and boxes per class and split).
4. Open `train.ipynb` on Kaggle or Colab with a GPU and run all cells.
5. The notebook copies the best model to `weights/best.pt` and zips the training and test results. Commit `weights/best.pt`.

Tests: `pip install pytest && pytest`

## Pilot results

_To be filled with measured numbers only._

## Deploy to Streamlit Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Create app → this repo, branch `main`, main file `app.py`, Python 3.11 (under Advanced settings).
3. Deploy. `requirements.txt` installs the CPU build of PyTorch and `packages.txt` installs the system libraries OpenCV needs.

## Project structure

```
app.py                  Streamlit demo
merge_datasets.py       dataset merging and label conversion
test_merge_datasets.py  tests for the merger
sources.yaml            which datasets to merge and how their classes map
train.ipynb             GPU training and evaluation
weights/best.pt         trained model used by the app
```

## Datasets

- RAD: Road Anomaly Detection dataset, Mendeley Data. https://data.mendeley.com/datasets/8chk8vdn2z/2
- HighRPD: a high-altitude drone dataset of road pavement distress, Data in Brief (2025). https://www.sciencedirect.com/science/article/pii/S235234092500109X
- Indian Roads: _source link to be added_
- Humps/Bumps/Potholes: _source link to be added_
- Crack-seg (used for the pilot run): Ultralytics. https://docs.ultralytics.com/datasets/segment/crack-seg/

Each dataset keeps its own license. Check it before redistributing images.

## Acknowledgements

- [collabdoor/Road-Anomaly-Detection](https://github.com/collabdoor/Road-Anomaly-Detection) (Apache-2.0), whose project and [demo video](https://www.youtube.com/watch?v=W_SFcuZuRBE) inspired this one and suggested the four source datasets. No code, model weights or results from that repository are used here.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io), [OpenCV](https://opencv.org)

## Team

- Sahil Lenka (2427010391)
- Kavish Jain (2427010427)
- Tanmay Singh (2427010440)

Supervisor: _to be added_

## License

AGPL-3.0, the same license as Ultralytics YOLOv8, which this project builds on.
