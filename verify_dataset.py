from pathlib import Path

import numpy as np
from PIL import Image


DATASET_DIR = Path("data/dataset/town10_clear_02")
RGB_DIR = DATASET_DIR / "rgb"
MASK_DIR = DATASET_DIR / "road_masks"
EXPECTED_SIZE = (320, 180)


rgb_files = {path.name: path for path in RGB_DIR.glob("*.png")}
mask_files = {path.name: path for path in MASK_DIR.glob("*.png")}

rgb_names = set(rgb_files)
mask_names = set(mask_files)

missing_masks = sorted(rgb_names - mask_names)
missing_rgb = sorted(mask_names - rgb_names)
matching_names = sorted(rgb_names & mask_names)

errors = []

print(f"RGB images:      {len(rgb_files)}")
print(f"Road masks:      {len(mask_files)}")
print(f"Matching pairs:  {len(matching_names)}")

if missing_masks:
    errors.append(f"RGB images without masks: {missing_masks}")

if missing_rgb:
    errors.append(f"Masks without RGB images: {missing_rgb}")

for filename in matching_names:
    with Image.open(rgb_files[filename]) as rgb_image:
        rgb_size = rgb_image.size

    with Image.open(mask_files[filename]) as mask_image:
        mask = np.array(mask_image.convert("L"))
        mask_size = mask_image.size

    if rgb_size != EXPECTED_SIZE:
        errors.append(
            f"{filename}: RGB size is {rgb_size}, expected {EXPECTED_SIZE}"
        )

    if mask_size != rgb_size:
        errors.append(
            f"{filename}: RGB size {rgb_size} != mask size {mask_size}"
        )

    values = set(np.unique(mask).tolist())

    if not values.issubset({0, 255}):
        errors.append(
            f"{filename}: mask contains unexpected values {sorted(values)}"
        )

if errors:
    print("\nDataset verification FAILED:")

    for error in errors:
        print(f"- {error}")

    raise SystemExit(1)

print("\nDataset verification PASSED")
print("Every RGB image has a matching binary road mask.")