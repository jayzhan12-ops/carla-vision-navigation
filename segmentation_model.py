from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from road_dataset import RoadDataset


DATASET_DIR = Path("data/dataset/town10_clear_02")
BATCH_SIZE = 4


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        return self.layers(images)


class RoadSegmentationNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: learn features while reducing image size.
        self.encoder1 = ConvBlock(3, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.pool = nn.MaxPool2d(2)

        # Deepest part of the network.
        self.bottleneck = ConvBlock(32, 64)

        # Decoder: rebuild a full-resolution mask.
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder2 = ConvBlock(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decoder1 = ConvBlock(32, 16)

        # Produce one road/non-road value per pixel.
        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, images):
        encoder1 = self.encoder1(images)
        encoder2 = self.encoder2(self.pool(encoder1))

        bottleneck = self.bottleneck(self.pool(encoder2))

        decoder2 = self.up2(bottleneck)
        decoder2 = torch.cat((decoder2, encoder2), dim=1)
        decoder2 = self.decoder2(decoder2)

        decoder1 = self.up1(decoder2)
        decoder1 = torch.cat((decoder1, encoder1), dim=1)
        decoder1 = self.decoder1(decoder1)

        return self.output_layer(decoder1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RoadDataset(DATASET_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rgb_batch, correct_masks = next(iter(loader))

    rgb_batch = rgb_batch.to(device, non_blocking=True)
    correct_masks = correct_masks.to(device, non_blocking=True)

    model = RoadSegmentationNet().to(device)
    model.eval()

    with torch.inference_mode():
        predicted_values = model(rgb_batch)
        predicted_probabilities = torch.sigmoid(predicted_values)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Model parameters: {parameter_count:,}")
    print(f"RGB input shape: {rgb_batch.shape}")
    print(f"Correct-mask shape: {correct_masks.shape}")
    print(f"Prediction shape: {predicted_values.shape}")
    print(
        "Prediction probability range: "
        f"{predicted_probabilities.min():.3f} to "
        f"{predicted_probabilities.max():.3f}"
    )

    if predicted_values.shape != correct_masks.shape:
        raise RuntimeError("Prediction and mask shapes do not match.")

    print("Forward-pass test PASSED")