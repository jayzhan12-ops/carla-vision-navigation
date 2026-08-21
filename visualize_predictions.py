from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from road_dataset import RoadDataset
from segmentation_model import RoadSegmentationNet


DATASET_DIR = Path("data/dataset/town10_clear_02")
CHECKPOINT_PATH = Path("checkpoints/smoke_test_model.pt")
OUTPUT_PATH = Path("results/smoke_test_predictions.png")

NUMBER_OF_EXAMPLES = 4
THRESHOLD = 0.5
HEADER_HEIGHT = 30


def tensor_to_rgb_image(tensor):
    array = (
        tensor.permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def tensor_to_mask_image(tensor):
    array = (
        tensor.squeeze(0)
        .cpu()
        .numpy()
    )

    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def calculate_iou(predicted_mask, correct_mask):
    predicted_mask = predicted_mask.bool()
    correct_mask = correct_mask.bool()

    intersection = torch.logical_and(
        predicted_mask,
        correct_mask,
    ).sum().item()

    union = torch.logical_or(
        predicted_mask,
        correct_mask,
    ).sum().item()

    return intersection / max(union, 1)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = RoadDataset(DATASET_DIR)

    model = RoadSegmentationNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    number_of_examples = min(
        NUMBER_OF_EXAMPLES,
        len(dataset),
    )

    # Select examples from different parts of the dataset.
    selected_indices = torch.linspace(
        0,
        len(dataset) - 1,
        steps=number_of_examples,
    ).round().int().tolist()

    first_rgb, _ = dataset[0]
    image_height = first_rgb.shape[1]
    image_width = first_rgb.shape[2]

    canvas = Image.new(
        "RGB",
        (
            image_width * 3,
            HEADER_HEIGHT + image_height * number_of_examples,
        ),
        color=(30, 30, 30),
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 8),
        "RGB input",
        fill="white",
    )

    draw.text(
        (image_width + 10, 8),
        "Correct CARLA mask",
        fill="white",
    )

    draw.text(
        (image_width * 2 + 10, 8),
        "PyTorch prediction",
        fill="white",
    )

    iou_scores = []

    with torch.inference_mode():
        for row, dataset_index in enumerate(selected_indices):
            rgb_tensor, correct_mask = dataset[dataset_index]

            rgb_batch = (
                rgb_tensor
                .unsqueeze(0)
                .to(device)
            )

            logits = model(rgb_batch)
            probabilities = torch.sigmoid(logits)

            predicted_mask = (
                probabilities[0].cpu() >= THRESHOLD
            ).float()

            iou = calculate_iou(
                predicted_mask,
                correct_mask,
            )

            iou_scores.append(iou)

            rgb_image = tensor_to_rgb_image(
                rgb_tensor
            )

            correct_image = tensor_to_mask_image(
                correct_mask
            )

            predicted_image = tensor_to_mask_image(
                predicted_mask
            )

            y_position = (
                HEADER_HEIGHT + row * image_height
            )

            canvas.paste(
                rgb_image,
                (0, y_position),
            )

            canvas.paste(
                correct_image,
                (image_width, y_position),
            )

            canvas.paste(
                predicted_image,
                (image_width * 2, y_position),
            )

            draw.text(
                (
                    image_width * 2 + 5,
                    y_position + 5,
                ),
                f"IoU: {iou:.4f}",
                fill=(255, 0, 0),
            )

            print(
                f"Sample {dataset_index:03d} | "
                f"IoU: {iou:.4f}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(OUTPUT_PATH)

    mean_iou = sum(iou_scores) / len(iou_scores)

    print(f"\nDevice: {device}")
    print(f"Mean visualized IoU: {mean_iou:.4f}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()