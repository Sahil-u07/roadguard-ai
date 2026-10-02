import pytest
import yaml

from merge_datasets import merge, voc_to_yolo

CLASSES = ["Pothole", "Crack", "Speed-Bump"]

VOC_XML = """<annotation>
  <size><width>200</width><height>100</height></size>
  <object><name>pothole</name><bndbox><xmin>50</xmin><ymin>25</ymin><xmax>150</xmax><ymax>75</ymax></bndbox></object>
  <object><name>manhole</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>10</ymax></bndbox></object>
</annotation>"""


def make_yolo_source(root):
    """A YOLO source with train/ and val/ splits and its own class order."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["crack", "pothole", "car"]}))
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / "a.jpg").write_bytes(b"img")
        (root / split / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n")
    # polygon row, as exported by segmentation tools
    (root / "train" / "labels" / "a.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n")


def make_voc_source(root, n):
    """A VOC source with no splits, so the merger has to split it."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n):
        # dotted names like Roboflow exports ("10.rf.abc.jpg")
        (root / "images" / f"{i}.rf.abc.jpg").write_bytes(b"img")
        (root / "labels" / f"{i}.rf.abc.xml").write_text(VOC_XML)


def test_voc_to_yolo_normalises_boxes(tmp_path):
    (tmp_path / "a.xml").write_text(VOC_XML)
    assert voc_to_yolo(tmp_path / "a.xml")[0] == ("pothole", 0.5, 0.5, 0.5, 0.5)


def test_merge_remaps_classes_and_splits(tmp_path):
    make_yolo_source(tmp_path / "rad")
    make_voc_source(tmp_path / "hrpd", 20)
    cfg = {"classes": CLASSES, "sources": [
        {"name": "rad", "path": str(tmp_path / "rad"), "map": {"crack": "Crack", "pothole": "Pothole", "car": None}},
        {"name": "hrpd", "path": str(tmp_path / "hrpd"), "format": "voc",
         "map": {"pothole": "Pothole", "manhole": None}},
    ]}
    out = tmp_path / "out"
    stats = merge(cfg, out)

    # source class 1 (pothole) becomes unified 0; car is dropped
    assert (out / "valid/labels/rad_a.txt").read_text() == "0 0.500000 0.500000 0.200000 0.200000"
    # polygon becomes its bounding box; crack becomes unified 1
    assert (out / "train/labels/rad_a.txt").read_text() == "1 0.200000 0.300000 0.200000 0.400000"
    assert sum(stats[s, "images"] for s in ("train", "valid", "test")) == 22
    assert stats["train", "images"] > stats["test", "images"]
    assert sum(stats[s, "Pothole"] for s in ("train", "valid", "test")) == 21
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "Pothole", 1: "Crack", 2: "Speed-Bump"}


def test_unmapped_class_is_an_error(tmp_path):
    make_voc_source(tmp_path / "hrpd", 1)
    cfg = {"classes": CLASSES, "sources": [
        {"name": "hrpd", "path": str(tmp_path / "hrpd"), "format": "voc", "split": [1, 0, 0], "map": {"pothole": "Pothole"}}]}
    with pytest.raises(ValueError, match="manhole"):
        merge(cfg, tmp_path / "out")
