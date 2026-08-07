import io
import json
from pathlib import Path

import polars as pl
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "cursor-dataset.parquet"
OUTPUT = ROOT / "cursor_dataset"


def main():
    frame = pl.read_parquet(SOURCE)
    counts = {}
    for index, row in enumerate(frame.iter_rows(named=True)):
        split = row["split"]
        counts[split] = counts.get(split, 0) + 1
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        stem = f"cursor_{index:04d}"
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        image.save(image_dir / f"{stem}.jpg", quality=95)

        box = json.loads(row["bbox"])
        label = f'{box["class"]} {box["x_center"]} {box["y_center"]} {box["width"]} {box["height"]}\n'
        (label_dir / f"{stem}.txt").write_text(label)

    yaml = f"""path: {OUTPUT.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: cursor
"""
    (OUTPUT / "cursor.yaml").write_text(yaml)
    print(counts)
    print(OUTPUT / "cursor.yaml")


if __name__ == "__main__":
    main()
