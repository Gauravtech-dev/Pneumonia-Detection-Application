from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from PIL import Image


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = Path(
    r"C:\Users\ACER\.cache\kagglehub\datasets"
    r"\paultimothymooney\chest-xray-pneumonia"
    r"\versions\2\chest_xray"
)

TEST_PATH = DATASET_PATH / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "chest_xray_resnet18.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "model"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 60)


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# DATASET
# =========================================================

if not TEST_PATH.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_PATH}"
    )

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))


# =========================================================
# MODEL
# =========================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("ResNet18 loaded successfully.")


# =========================================================
# ERROR ANALYSIS
# =========================================================

class_names = test_dataset.classes

errors = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )

        for i in range(len(labels)):

            total += 1

            actual = labels[i].item()
            predicted = predictions[i].item()

            confidence = (
                probabilities[i][predicted]
                .item()
                * 100
            )

            if actual == predicted:

                correct += 1

            else:

                image_path = test_dataset.samples[
                    total - 1
                ][0]

                errors.append({
                    "path": image_path,
                    "actual": class_names[actual],
                    "predicted": class_names[predicted],
                    "confidence": confidence,
                })


# =========================================================
# RESULTS
# =========================================================

accuracy = (
    correct / total
    if total > 0
    else 0
)

print()
print("=" * 60)
print("ERROR ANALYSIS RESULTS")
print("=" * 60)

print(
    f"Total images       : {total}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Wrong predictions  : {len(errors)}"
)

print(
    f"Accuracy            : {accuracy * 100:.2f}%"
)


# =========================================================
# SAVE ERROR REPORT
# =========================================================

report_path = (
    OUTPUT_DIR
    / "misclassified_images.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "RESNET18 ERROR ANALYSIS\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Total images: {total}\n"
    )

    file.write(
        f"Correct predictions: {correct}\n"
    )

    file.write(
        f"Wrong predictions: {len(errors)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "MISCLASSIFIED IMAGES\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    for index, error in enumerate(
        errors,
        start=1
    ):

        file.write(
            f"{index}. "
            f"Actual: {error['actual']} | "
            f"Predicted: {error['predicted']} | "
            f"Confidence: "
            f"{error['confidence']:.2f}%\n"
        )

        file.write(
            f"Path: {error['path']}\n\n"
        )


print()
print(
    "Error report saved:"
)

print(report_path)

print()
print("=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)
