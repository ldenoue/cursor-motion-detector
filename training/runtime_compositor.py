"""On-the-fly cursor composition without rendered-image storage."""

import io
import random

import polars as pl
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from ultralytics.data.dataset import YOLODataset

from training.augment_macos_dataset import MAC_WEIGHTS, SOURCE, composite, crop_background, weighted_mac_cursor


class RuntimeCursorDataset(Dataset):
    """Deterministically generates a fresh synthetic sample for each epoch/index.

    Returns PIL images and normalized xywh labels. A training adapter can convert
    these directly to tensors in worker processes, so only the compressed source
    screenshots and cursor sprite parquet files remain on disk.
    """

    def __init__(self, split="train", length=5000, negative_rate=0.05, seed=20260806, dynamic=None):
        split_file = {"train": "train", "val": "val", "test": "test"}[split]
        self.backgrounds = list(pl.read_parquet(SOURCE / f"backgrounds-{split_file}.parquet").iter_rows(named=True))
        cursors = list(pl.read_parquet(SOURCE / "cursors.parquet").iter_rows(named=True))
        self.mac = [row for row in cursors if row["os"] == "macos"]
        self.other = [row for row in cursors if row["os"] != "macos" and row["cursor_type"] in MAC_WEIGHTS]
        self.length = length
        self.negative_rate = negative_rate
        self.seed = seed
        self.epoch = 0
        self.dynamic = split == "train" if dynamic is None else dynamic
        self.access_counts = [0] * length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        generation = self.access_counts[index] if self.dynamic else 0
        self.access_counts[index] += 1
        rng = random.Random(self.seed + (self.epoch + generation) * self.length + index)
        row = self.backgrounds[index % len(self.backgrounds)]
        image = crop_background(Image.open(io.BytesIO(row["image"]["bytes"])), rng)
        cursor = None
        box = None
        if rng.random() >= self.negative_rate:
            cursor = weighted_mac_cursor(rng, self.mac) if rng.random() < 0.98 else rng.choice(self.other)
            box = composite(image, cursor, rng)
        if rng.random() < 0.08:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.65)))
        return {
            "image": image,
            "bbox": box,
            "class": None if box is None else 0,
            "cursor_type": "negative" if cursor is None else cursor["cursor_type"],
            "os": "none" if cursor is None else cursor["os"],
        }


class UltralyticsRuntimeCursorDataset(RuntimeCursorDataset):
    """Runtime compositor formatted for Ultralytics DetectionTrainer."""

    rect = False
    collate_fn = staticmethod(YOLODataset.collate_fn)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ultralytics uses this only for class statistics and label plots.
        cls = np.array([[0.0]], dtype=np.float32)
        bboxes = np.array([[0.5, 0.5, 0.03, 0.03]], dtype=np.float32)
        self.labels = [{"cls": cls, "bboxes": bboxes}] * len(self)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image = torch.from_numpy(np.ascontiguousarray(np.asarray(sample["image"]).transpose(2, 0, 1)))
        if sample["bbox"] is None:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            cls = torch.tensor([[0.0]], dtype=torch.float32)
            bboxes = torch.tensor([sample["bbox"]], dtype=torch.float32)
        return {
            "img": image,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros(len(cls), dtype=torch.float32),
            "im_file": f"runtime-{self.seed}-{index}.jpg",
            "ori_shape": (640, 640),
            "resized_shape": (640, 640),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
        }
