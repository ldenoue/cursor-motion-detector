#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
bundle_path="${1:-$repo_dir/yolo-cursor-colab.zip}"

cd "$repo_dir"
python3 -c 'import zipfile
from pathlib import Path

root = Path.cwd()
output = Path("'"$bundle_path"'").resolve()
files = [
    Path("training/__init__.py"),
    Path("training/augment_macos_dataset.py"),
    Path("training/runtime_compositor.py"),
    Path("training/runtime.yaml"),
    Path("training/train_augmented.py"),
    Path("training/temporal_dataset.py"),
    Path("training/temporal_model.py"),
    Path("training/train_temporal.py"),
    Path("training/source/backgrounds-train.parquet"),
    Path("training/source/backgrounds-val.parquet"),
    Path("training/source/backgrounds-test.parquet"),
    Path("training/source/cursors.parquet"),
]
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise SystemExit("Missing required files: " + ", ".join(missing))
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
    for path in files:
        archive.write(path, Path("yolo-cursor") / path)
print(f"Created {output} ({output.stat().st_size / 1024**2:.1f} MiB)")'
