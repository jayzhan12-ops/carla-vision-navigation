from pathlib import Path

import numpy as np
from PIL import Image


# -------------------- Configuration --------------------

RUN_NAME = "town10_clear_02"
RUN_DIR = Path("data/dataset") / RUN_NAME

RGB_DIR = RUN_DIR / "rgb"
SEMANTIC_RAW_DIR = RUN_DIR / "semantic_raw"
MASK_DIR = RUN_DIR / "road_masks"

# CARLA semantic class IDs
ROAD_LINE_ID = 24
ROAD_ID = 1

ROAD_CLASS_IDS = [ROAD_LINE_ID, ROAD_ID]


# -------------------- Validate input --------------------

raw_files = sorted(SEMANTIC_RAW_DIR.glob("*.png"))

if not raw_files:
    raise FileNotFoundError(
        f"No semantic raw images found in "
        f"{SEMANTIC_RAW_DIR.resolve()}"
    )

rgb_names = {
    image_path.name
    for image_path in RGB_DIR.glob("*.png")
}

raw_names = {
    image_path.name
    for image_path in raw_files
}

if rgb_names != raw_names:
    missing_rgb = sorted(raw_names - rgb_names)
    missing_raw = sorted(rgb_names - raw_names)

    raise RuntimeError(
        "RGB and semantic filenames do not match.\n"
        f"Missing RGB: {missing_rgb}\n"
        f"Missing semantic raw: {missing_raw}"
    )

MASK_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- Create masks --------------------

total_road_pixels = 0
total_pixels = 0

for index, raw_path in enumerate(raw_files, start=1):
    # CARLA stores semantic class IDs in the red channel.
    semantic_image = np.array(
        Image.open(raw_path).convert("RGB")
    )

    class_ids = semantic_image[:, :, 0]

    # Road and RoadLine become True; everything else becomes False.
    road_pixels = np.isin(
        class_ids,
        ROAD_CLASS_IDS,
    )

    # Convert:
    # False → 0   → black
    # True  → 255 → white
    road_mask = road_pixels.astype(np.uint8) * 255

    output_path = MASK_DIR / raw_path.name
    Image.fromarray(road_mask).save(output_path)

    road_pixel_count = int(road_pixels.sum())
    image_pixel_count = road_pixels.size
    road_percentage = (
        100.0 * road_pixel_count / image_pixel_count
    )

    total_road_pixels += road_pixel_count
    total_pixels += image_pixel_count

    print(
        f"[{index:03d}/{len(raw_files)}] "
        f"{raw_path.name} — "
        f"road coverage: {road_percentage:.1f}%"
    )


# -------------------- Summary --------------------

average_road_percentage = (
    100.0 * total_road_pixels / total_pixels
)

print("\nRoad-mask generation complete.")
print(f"Masks created: {len(raw_files)}")
print(
    f"Average road coverage: "
    f"{average_road_percentage:.1f}%"
)
print(f"Output: {MASK_DIR.resolve()}")