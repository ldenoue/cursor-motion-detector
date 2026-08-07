"""Runtime-generated three-frame cursor sequences for hotspot detection."""

import io
import math
import random

import numpy as np
import polars as pl
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from training.augment_macos_dataset import MAC_WEIGHTS, SOURCE, alpha_crop, crop_background, weighted_mac_cursor

IMAGE_SIZE = 640
OUTPUT_STRIDE = 4
HEATMAP_SIZE = IMAGE_SIZE // OUTPUT_STRIDE


def _cursor_sprite(cursor, rng):
    sprite = alpha_crop(cursor)
    target = rng.choices(
        [rng.randint(6, 12), rng.randint(13, 24), rng.randint(25, 40)],
        weights=[0.25, 0.55, 0.20],
        k=1,
    )[0]
    scale = target / max(sprite.size)
    size = (max(2, round(sprite.width * scale)), max(2, round(sprite.height * scale)))
    sprite = sprite.resize(size, Image.Resampling.LANCZOS)
    # Hotspots in Fraser/cursors are normalized to the alpha-cropped sprite.
    hotspot = (float(cursor["hotspot_x"]) * size[0], float(cursor["hotspot_y"]) * size[1])
    return sprite, hotspot, target


def _trajectory(rng):
    current = np.array([rng.uniform(0, IMAGE_SIZE - 1), rng.uniform(0, IMAGE_SIZE - 1)], dtype=np.float32)
    mode = rng.random()
    speed = 0.0 if mode < 0.25 else rng.uniform(1, 5) if mode < 0.65 else rng.uniform(6, 18) if mode < 0.93 else rng.uniform(19, 55)
    angle = rng.uniform(0, math.tau)
    velocity = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * speed
    acceleration = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2)], dtype=np.float32)
    previous = np.clip(current - velocity, 0, IMAGE_SIZE - 1)
    oldest = np.clip(previous - velocity + acceleration, 0, IMAGE_SIZE - 1)
    return oldest, previous, current


def _paste_at_hotspot(frame, sprite, sprite_hotspot, position):
    x = round(float(position[0]) - sprite_hotspot[0])
    y = round(float(position[1]) - sprite_hotspot[1])
    frame.paste(sprite, (x, y), sprite)


def _shift_background(image, dx, dy):
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(114, 114, 114),
    )


def _soft_motion(current, previous, threshold=10 / 255, width=30 / 255):
    return np.clip((np.abs(current - previous) - threshold) / width, 0, 1)


def _gaussian_heatmap(x, y, sigma):
    heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    radius = max(1, math.ceil(3 * sigma))
    cx, cy = int(x), int(y)
    left, right = max(0, cx - radius), min(HEATMAP_SIZE, cx + radius + 1)
    top, bottom = max(0, cy - radius), min(HEATMAP_SIZE, cy + radius + 1)
    yy, xx = np.mgrid[top:bottom, left:right]
    heatmap[top:bottom, left:right] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    heatmap[cy, cx] = 1.0
    return heatmap, cx, cy


class TemporalCursorDataset(Dataset):
    """Compose a coherent cursor trajectory and return grayscale/motion channels."""

    def __init__(self, split="train", length=10000, negative_rate=0.05, seed=20260806, dynamic=None):
        self.backgrounds = list(pl.read_parquet(SOURCE / f"backgrounds-{split}.parquet").iter_rows(named=True))
        cursors = list(pl.read_parquet(SOURCE / "cursors.parquet").iter_rows(named=True))
        self.mac = [row for row in cursors if row["os"] == "macos"]
        self.other = [row for row in cursors if row["os"] != "macos" and row["cursor_type"] in MAC_WEIGHTS]
        self.length = length
        self.negative_rate = negative_rate
        self.seed = seed
        self.epoch = 0
        self.dynamic = split == "train" if dynamic is None else dynamic

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        epoch = self.epoch if self.dynamic else 0
        rng = random.Random(self.seed + epoch * self.length + index)
        row = self.backgrounds[index % len(self.backgrounds)]
        current_background = crop_background(Image.open(io.BytesIO(row["image"]["bytes"])), rng)
        frames = [current_background.copy() for _ in range(3)]

        # Background motion prevents the temporal channels from becoming a
        # trivial cursor-only mask on scrolling and animated recordings.
        if rng.random() < 0.18:
            dx, dy = rng.randint(-12, 12), rng.randint(-18, 18)
            frames[0] = _shift_background(frames[0], -2 * dx, -2 * dy)
            frames[1] = _shift_background(frames[1], -dx, -dy)
        if rng.random() < 0.12:
            for frame in frames[:2]:
                draw = ImageDraw.Draw(frame)
                x, y = rng.randint(0, 560), rng.randint(0, 600)
                draw.rectangle((x, y, x + rng.randint(15, 80), y + rng.randint(4, 25)), fill=tuple(rng.randint(30, 230) for _ in range(3)))

        visible = rng.random() >= self.negative_rate
        position = np.zeros(2, dtype=np.float32)
        cursor_type = "negative"
        target_size = 0
        if visible:
            cursor = weighted_mac_cursor(rng, self.mac) if rng.random() < 0.98 else rng.choice(self.other)
            sprite, sprite_hotspot, target_size = _cursor_sprite(cursor, rng)
            positions = _trajectory(rng)
            for frame, point in zip(frames, positions):
                _paste_at_hotspot(frame, sprite, sprite_hotspot, point)
            position = positions[-1]
            cursor_type = cursor["cursor_type"]

        if rng.random() < 0.08:
            frames = [frame.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.65))) for frame in frames]
        gray = [np.asarray(frame.convert("L"), dtype=np.float32) / 255 for frame in frames]
        channels = np.stack([gray[2], _soft_motion(gray[2], gray[1]), _soft_motion(gray[2], gray[0])])

        heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        offset = np.zeros((2, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        mask = np.zeros((1, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        if visible:
            hx, hy = position / OUTPUT_STRIDE
            sigma = min(4.0, max(1.0, target_size / (OUTPUT_STRIDE * 3)))
            heatmap, cx, cy = _gaussian_heatmap(float(hx), float(hy), sigma)
            offset[:, cy, cx] = (hx - cx, hy - cy)
            mask[0, cy, cx] = 1.0

        return {
            "input": torch.from_numpy(channels),
            "heatmap": torch.from_numpy(heatmap[None]),
            "offset": torch.from_numpy(offset),
            "mask": torch.from_numpy(mask),
            "position": torch.from_numpy(position),
            "visible": torch.tensor(visible, dtype=torch.bool),
            "cursor_type": cursor_type,
        }
