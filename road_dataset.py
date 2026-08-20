from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


DATASET_DIR = Path("data/dataset/town10_clear_02")
BATCH_SIZE = 4


class RoadDataset(Dataset):
    def __init__(self, dataset_dir):
        self.rgb_dir = dataset_dir / "rgb"
        self.mask_dir = dataset_dir / "road_masks"
        self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))

        if not self.rgb_paths:
            raise RuntimeError("No RGB images found.")

        for rgb_path in self.rgb_paths:
            mask_path = self.mask_dir / rgb_path.name

            if not mask_path.exists():
                raise RuntimeError(f"Missing mask for {rgb_path.name}")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        mask_path = self.mask_dir / rgb_path.name

        rgb = np.array(Image.open(rgb_path).convert("RGB"), copy=True)
        mask = np.array(Image.open(mask_path).convert("L"), copy=True)

        # [height, width, channels] -> [channels, height, width]
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # Black/white pixels -> 0/1, then add the channel dimension.
        mask_tensor = torch.from_numpy(
            (mask > 127).astype(np.float32)
        ).unsqueeze(0)

        return rgb_tensor, mask_tensor


if __name__ == "__main__":
    dataset = RoadDataset(DATASET_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rgb_batch, mask_batch = next(iter(loader))

    print(f"Dataset samples: {len(dataset)}")
    print(f"Number of batches: {len(loader)}")
    print(f"RGB batch shape: {rgb_batch.shape}")
    print(f"Mask batch shape: {mask_batch.shape}")
    print(f"RGB range: {rgb_batch.min():.1f} to {rgb_batch.max():.1f}")
    print(f"Mask values: {torch.unique(mask_batch).tolist()}")