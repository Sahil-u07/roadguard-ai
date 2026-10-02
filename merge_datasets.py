"""Merge several road datasets (YOLO or Pascal-VOC) into one YOLOv8 dataset with a unified class list.

Usage:
    python merge_datasets.py sources.yaml datasets/roadguard

Each source folder either has split folders (train/, valid/ or val/, test/), each holding
images/ and labels/, or just images/ and labels/ at its root, in which case it is split
randomly using the source's `split` ratios. YOLO sources need a data.yaml with `names`;
VOC sources (`format: voc`) keep one .xml per image in labels/.
"""
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml

SPLIT_NAMES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_to_yolo(xml_path):
    """Read a Pascal-VOC file into [(class_name, cx, cy, w, h)] with coordinates in 0..1."""
    root = ET.parse(xml_path).getroot()
    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)
    boxes = []
    for obj in root.iter("object"):
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((obj.find("name").text.strip(),
                      (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                      (x2 - x1) / width, (y2 - y1) / height))
    return boxes


def read_yolo(txt_path, names):
    """Read a YOLO label file into [(class_name, cx, cy, w, h)].

    Polygon rows (segmentation exports) are turned into their bounding box.
    """
    boxes = []
    for line in Path(txt_path).read_text().split("\n"):
        if not line.strip():
            continue
        cls, *v = line.split()
        v = [float(x) for x in v]
        if len(v) > 4:
            xs, ys = v[0::2], v[1::2]
            v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        boxes.append((names[int(cls)], *v))
    return boxes


def source_images(src, ratios, rng):
    """Yield (image_path, unified_split) for every image in a source folder."""
    splits = {SPLIT_NAMES[d.name]: d for d in src.iterdir() if d.name in SPLIT_NAMES and (d / "images").is_dir()}
    if splits:
        for split, folder in splits.items():
            for img in sorted((folder / "images").iterdir()):
                if img.suffix.lower() in IMAGE_EXTS:
                    yield img, split
        return
    train, valid, _ = ratios
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() in IMAGE_EXTS:
            r = rng.random()
            yield img, "train" if r < train else "valid" if r < train + valid else "test"


def merge(config, out):
    """Merge every source in `config` into `out`. Returns a Counter of (split, class) -> box count."""
    classes = config["classes"]
    class_id = {c: i for i, c in enumerate(classes)}
    rng = random.Random(config.get("seed", 0))
    out = Path(out)
    stats = Counter()

    for source in config["sources"]:
        src, mapping = Path(source["path"]), source["map"]
        voc = source.get("format") == "voc"
        if not voc:
            names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        for unknown in set(mapping.values()) - set(classes) - {None}:
            raise ValueError(f"{source['name']}: '{unknown}' is not in the unified classes")

        for img, split in source_images(src, source.get("split", [0.7, 0.2, 0.1]), rng):
            # Not with_suffix(): Roboflow names like "10.rf.abc.jpg" have extra dots in the stem.
            label = img.parent.parent / "labels" / (img.stem + (".xml" if voc else ".txt"))
            if not label.exists():
                boxes = []
            elif voc:
                boxes = voc_to_yolo(label)
            else:
                boxes = read_yolo(label, names)

            lines = []
            for name, *xywh in boxes:
                if name not in mapping:
                    raise ValueError(f"{source['name']}: class '{name}' is missing from its map (map it to null to drop it)")
                if mapping[name] is None:
                    continue
                lines.append(f"{class_id[mapping[name]]} " + " ".join(f"{x:.6f}" for x in xywh))
                stats[split, mapping[name]] += 1

            # Prefix with the source name so files from different datasets never overwrite each other.
            stem = f"{source['name']}_{img.stem}"
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / split / "images" / (stem + img.suffix))
            (out / split / "labels" / (stem + ".txt")).write_text("\n".join(lines))
            stats[split, "images"] += 1

    data = {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
            "test": "test/images", "names": dict(enumerate(classes))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return stats


def print_stats(stats, classes):
    print(f"{'':16}" + "".join(f"{s:>10}" for s in ("train", "valid", "test")))
    for row in ["images", *classes]:
        print(f"{row:16}" + "".join(f"{stats[s, row]:>10}" for s in ("train", "valid", "test")))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    print_stats(merge(cfg, sys.argv[2]), cfg["classes"])
