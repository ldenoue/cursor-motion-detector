"""Build a larger, macOS-heavy cursor detection dataset.

Uses clean website screenshots and MIT-licensed cursor sprites from
https://huggingface.co/datasets/Fraser/cursors. The existing 500 original
training examples are retained, while validation and test use held-out clean
backgrounds and freshly composited cursors.
"""

import argparse
import csv
import io
import random
import shutil
from collections import Counter
from pathlib import Path

import polars as pl
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
OUTPUT = ROOT / "cursor_augmented"
MAC_WEIGHTS = {
    "arrow": 0.45,
    # Fraser's names describe writing direction: its `text_vertical` sprite is
    # the familiar tall macOS I-beam, while `text` is the uncommon horizontal
    # I-beam visible in the reported bad samples.
    "text_vertical": 0.25,
    "pointer": 0.20,
    "grab": 0.02,
    "grabbing": 0.02,
    "progress": 0.02,
    "resize_ew": 0.01,
    "resize_ns": 0.01,
    "resize_nesw": 0.01,
    "resize_nwse": 0.01,
}


def weighted_mac_cursor(rng, cursors):
    emphasized = []
    weights = []
    for cursor_type, weight in MAC_WEIGHTS.items():
        matches = [c for c in cursors if c["cursor_type"] == cursor_type]
        if matches:
            emphasized.append(matches[0])
            weights.append(weight)
    return rng.choices(emphasized, weights=weights, k=1)[0]


def alpha_crop(cursor):
    frame = cursor["frames"][0]
    image = Image.open(io.BytesIO(frame["bytes"])).convert("RGBA")
    bounds = image.getchannel("A").getbbox()
    return image.crop(bounds) if bounds else image


def crop_background(image, rng):
    """Create a non-distorted 640×640 screenshot crop."""
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = rng.randint(0, width - side)
    top = rng.randint(0, height - side)
    return image.crop((left, top, left + side, top + side)).resize((640, 640), Image.Resampling.LANCZOS)


def composite(background, cursor, rng):
    sprite = alpha_crop(cursor)
    # Include tiny cursors resulting from downscaled 1080p recordings, normal
    # cursors, and macOS accessibility-enlarged cursors.
    target = rng.choices(
        [rng.randint(6, 12), rng.randint(13, 24), rng.randint(25, 40)],
        weights=[0.25, 0.55, 0.20],
        k=1,
    )[0]
    scale = target / max(sprite.size)
    size = (max(2, round(sprite.width * scale)), max(2, round(sprite.height * scale)))
    sprite = sprite.resize(size, Image.Resampling.LANCZOS)

    width, height = background.size
    # Include a small number of edge-clipped cursors, common in recordings.
    margin_x = round(sprite.width * (0.35 if rng.random() < 0.12 else 0))
    margin_y = round(sprite.height * (0.35 if rng.random() < 0.12 else 0))
    x = rng.randint(-margin_x, max(-margin_x, width - sprite.width + margin_x))
    y = rng.randint(-margin_y, max(-margin_y, height - sprite.height + margin_y))
    background.paste(sprite, (x, y), sprite)

    visible_x1, visible_y1 = max(0, x), max(0, y)
    visible_x2, visible_y2 = min(width, x + sprite.width), min(height, y + sprite.height)
    box_width, box_height = visible_x2 - visible_x1, visible_y2 - visible_y1
    return (
        (visible_x1 + visible_x2) / 2 / width,
        (visible_y1 + visible_y2) / 2 / height,
        box_width / width,
        box_height / height,
    )


def load_rows(path):
    return list(pl.read_parquet(path).iter_rows(named=True))


def generate_split(split, count, backgrounds, mac, other, rng, metadata, negative_rate):
    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        row = backgrounds[index % len(backgrounds)]
        background = crop_background(Image.open(io.BytesIO(row["image"]["bytes"])), rng)
        negative = rng.random() < negative_rate
        cursor = None
        box = None
        if not negative:
            cursor = weighted_mac_cursor(rng, mac) if rng.random() < 0.98 else rng.choice(other)
            box = composite(background, cursor, rng)

        # Mild video-like degradation without destroying tiny pointer edges.
        if rng.random() < 0.08:
            background = background.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.65)))
        quality = rng.randint(72, 96)
        stem = f"synthetic_{index:06d}"
        background.save(image_dir / f"{stem}.jpg", quality=quality, subsampling=rng.choice([0, 2]))
        label = "" if box is None else f"0 {' '.join(f'{value:.8f}' for value in box)}\n"
        (label_dir / f"{stem}.txt").write_text(label)
        metadata.append({
            "split": split,
            "file": f"{stem}.jpg",
            "os": "none" if cursor is None else cursor["os"],
            "cursor_type": "negative" if cursor is None else cursor["cursor_type"],
            "quality": quality,
        })


def retain_original_train():
    source = ROOT / "cursor_dataset"
    for kind in ("images", "labels"):
        destination = OUTPUT / kind / "train"
        destination.mkdir(parents=True, exist_ok=True)
        for path in (source / kind / "train").glob("*"):
            shutil.copy2(path, destination / f"original_{path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--negative-rate", type=float, default=0.05)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    cursor_rows = load_rows(SOURCE / "cursors.parquet")
    mac = [row for row in cursor_rows if row["os"] == "macos"]
    other = [row for row in cursor_rows if row["os"] != "macos" and row["cursor_type"] in MAC_WEIGHTS]
    metadata = []
    retain_original_train()
    generate_split("train", args.train, load_rows(SOURCE / "backgrounds-train.parquet"), mac, other, rng, metadata, args.negative_rate)
    generate_split("val", args.val, load_rows(SOURCE / "backgrounds-val.parquet"), mac, other, rng, metadata, args.negative_rate)
    generate_split("test", args.test, load_rows(SOURCE / "backgrounds-test.parquet"), mac, other, rng, metadata, args.negative_rate)

    with (OUTPUT / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    (OUTPUT / "cursor.yaml").write_text(f"""path: {OUTPUT.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: cursor
""")
    distribution = Counter((row["split"], row["os"], row["cursor_type"]) for row in metadata)
    print(f"Generated {args.train + 500} train, {args.val} val, {args.test} test images")
    print("macOS vertical I-beam train:", distribution[("train", "macos", "text_vertical")])
    print("macOS horizontal I-beam train:", distribution[("train", "macos", "text")])
    print("macOS arrow train:", distribution[("train", "macos", "arrow")])
    print("macOS pointer train:", distribution[("train", "macos", "pointer")])


if __name__ == "__main__":
    main()
