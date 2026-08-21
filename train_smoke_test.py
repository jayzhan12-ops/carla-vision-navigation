from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from road_dataset import RoadDataset
from segmentation_model import RoadSegmentationNet


DATASET_DIR = Path("data/dataset/town10_clear_02")
CHECKPOINT_PATH = Path("checkpoints/smoke_test_model.pt")

BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def calculate_iou(model, loader, device):
    model.eval()

    intersection = 0
    union = 0

    with torch.inference_mode():
        for rgb_images, correct_masks in loader:
            rgb_images = rgb_images.to(
                device,
                non_blocking=True,
            )

            correct_masks = correct_masks.to(
                device,
                non_blocking=True,
            )

            predicted_values = model(rgb_images)
            probabilities = torch.sigmoid(predicted_values)

            predicted_masks = probabilities >= 0.5
            correct_masks = correct_masks >= 0.5

            intersection += torch.logical_and(
                predicted_masks,
                correct_masks,
            ).sum().item()

            union += torch.logical_or(
                predicted_masks,
                correct_masks,
            ).sum().item()

    return intersection / max(union, 1)


def main():
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = RoadDataset(DATASET_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = RoadSegmentationNet().to(device)

    # Suitable for binary road/non-road segmentation.
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    initial_iou = calculate_iou(
        model,
        loader,
        device,
    )

    print(f"Device: {device}")
    print(f"Samples: {len(dataset)}")
    print(f"Initial IoU: {initial_iou:.4f}")
    print("Starting smoke-test training...\n")

    first_loss = None
    final_loss = None
    final_iou = initial_iou

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for rgb_images, correct_masks in loader:
            rgb_images = rgb_images.to(
                device,
                non_blocking=True,
            )

            correct_masks = correct_masks.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            predicted_values = model(rgb_images)

            loss = loss_function(
                predicted_values,
                correct_masks,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * rgb_images.size(0)
            )

        final_loss = total_loss / len(dataset)

        if first_loss is None:
            first_loss = final_loss

        if epoch == 1 or epoch % 10 == 0:
            final_iou = calculate_iou(
                model,
                loader,
                device,
            )

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Loss: {final_loss:.4f} | "
                f"Training IoU: {final_iou:.4f}"
            )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": EPOCHS,
            "final_loss": final_loss,
            "training_iou": final_iou,
        },
        CHECKPOINT_PATH,
    )

    print(f"\nFirst loss: {first_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Initial IoU: {initial_iou:.4f}")
    print(f"Final training IoU: {final_iou:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH.resolve()}")

    if final_loss < first_loss and final_iou > initial_iou:
        print("Training smoke test PASSED")
    else:
        print("Training did not improve enough. Inspect the results.")


if __name__ == "__main__":
    main()