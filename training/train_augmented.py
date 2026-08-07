"""Train YOLO26n with cursor sprites composited entirely at runtime."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from training.runtime_compositor import UltralyticsRuntimeCursorDataset


class RuntimeCursorTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        del img_path, batch  # Data is read directly from the compressed parquet sources.
        return UltralyticsRuntimeCursorDataset(
            split="train" if mode == "train" else "val",
            length=int(os.getenv("CURSOR_TRAIN_SAMPLES" if mode == "train" else "CURSOR_VAL_SAMPLES", "10000" if mode == "train" else "1000")),
            negative_rate=float(os.getenv("CURSOR_NEGATIVE_RATE", "0.05")),
            dynamic=mode == "train",
        )


def main():
    project = Path("training/runs").resolve()
    device = os.getenv("CURSOR_DEVICE", "mps")
    batch = int(os.getenv("CURSOR_BATCH", "32"))
    workers = int(os.getenv("CURSOR_WORKERS", "2"))

    # Stage 1: learn the new detection head while preserving pretrained features.
    model = YOLO("yolo26n.pt")
    model.train(
        trainer=RuntimeCursorTrainer,
        data="training/runtime.yaml",
        epochs=int(os.getenv("CURSOR_FROZEN_EPOCHS", "10")),
        imgsz=640,
        batch=batch,
        freeze=10,
        device=device,
        workers=workers,
        project=str(project),
        name="yolo26n-cursor-runtime-frozen",
        patience=5,
        plots=False,
        exist_ok=True,
    )

    # Stage 2: adapt the complete network with a ten-times lower learning rate.
    model = YOLO(project / "yolo26n-cursor-runtime-frozen" / "weights" / "best.pt")
    model.train(
        trainer=RuntimeCursorTrainer,
        data="training/runtime.yaml",
        epochs=int(os.getenv("CURSOR_UNFROZEN_EPOCHS", "15")),
        imgsz=640,
        batch=batch,
        lr0=0.001,
        device=device,
        workers=workers,
        project=str(project),
        name="yolo26n-cursor-runtime-unfrozen",
        patience=7,
        plots=False,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
