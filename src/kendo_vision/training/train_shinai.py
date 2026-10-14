"""Trains the 3-keypoint shinai model with Ultralytics."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def train(data: str = "data/shinai", base: str = "models/yolo26s-pose.pt", epochs: int = 150,
          image_size: int = 1280, batch: int = 8, output: str = "models/shinai.pt") -> None:
    """Expects data/images/{train,val} with labels alongside in data/labels/{train,val}."""
    from ultralytics import YOLO

    dataset_folder = Path(data).resolve()
    for split in ("train", "val"):
        split_folder = dataset_folder / "images" / split
        if not split_folder.is_dir():
            raise SystemExit(f"Missing {split_folder}; put whole videos in one split or the other")
    dataset = {
        "path": str(dataset_folder), "train": "images/train", "val": "images/val",
        "kpt_shape": [3, 3],  # 3 keypoints, each x, y, visibility
        "flip_idx": [0, 1, 2],  # no keypoint changes meaning when the image is mirrored
        "names": {0: "shinai"},
    }
    dataset_file = dataset_folder / "dataset.yaml"
    dataset_file.write_text(yaml.safe_dump(dataset))
    model = YOLO(base)
    runs_folder = str(Path("runs").resolve())  # absolute, or Ultralytics nests it under its own runs folder
    model.train(data=str(dataset_file), epochs=epochs, imgsz=image_size, batch=batch,
                project=runs_folder, name="shinai", exist_ok=True)
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, output)
    print(f"Copied {best_weights} to {output}")
