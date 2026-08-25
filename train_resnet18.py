from pathlib import Path
import copy
import shutil

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    random_split,
    WeightedRandomSampler,
)
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = Path(
    r"C:\Users\ACER\.cache\kagglehub\datasets"
    r"\paultimothymooney\chest-xray-pneumonia"
    r"\versions\2\chest_xray"
)

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = (
    MODEL_DIR / "chest_xray_resnet18.pth"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "chest_xray_resnet18_best.pth"
)

PREVIOUS_MODEL_PATH = (
    MODEL_DIR / "chest_xray_resnet18_previous.pth"
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
# SETTINGS
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_RATIO = 0.20
PATIENCE = 3
SEED = 42


# =========================================================
# SEED
# =========================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# BACKUP CURRENT MODEL
# =========================================================

if FINAL_MODEL_PATH.exists():

    shutil.copy2(
        FINAL_MODEL_PATH,
        PREVIOUS_MODEL_PATH
    )

    print(
        "Previous model backed up:"
    )

    print(
        PREVIOUS_MODEL_PATH
    )


# =========================================================
# DATASET CHECK
# =========================================================

train_path = DATASET_PATH / "train"
test_path = DATASET_PATH / "test"

if not train_path.exists():
    raise FileNotFoundError(
        f"Train dataset not found:\n{train_path}"
    )

if not test_path.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{test_path}"
    )


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

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


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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
# TRAIN DATASET
# =========================================================

full_train_aug = datasets.ImageFolder(
    train_path,
    transform=train_transform
)

full_train_eval = datasets.ImageFolder(
    train_path,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=eval_transform
)

print(
    "Classes:",
    full_train_aug.classes
)

print(
    "Class mapping:",
    full_train_aug.class_to_idx
)

print(
    "Total training images:",
    len(full_train_aug)
)

print(
    "Test images:",
    len(test_dataset)
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

generator = torch.Generator().manual_seed(
    SEED
)

train_size = int(
    len(full_train_aug)
    * (1 - VAL_RATIO)
)

val_size = (
    len(full_train_aug)
    - train_size
)

train_split, val_split = random_split(
    range(len(full_train_aug)),
    [train_size, val_size],
    generator=generator
)

train_indices = list(
    train_split
)

val_indices = list(
    val_split
)


# =========================================================
# SUBSETS
# =========================================================

train_dataset = torch.utils.data.Subset(
    full_train_aug,
    train_indices
)

val_dataset = torch.utils.data.Subset(
    full_train_eval,
    val_indices
)


print()
print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# =========================================================
# BALANCED SAMPLING
# =========================================================

train_targets = [
    full_train_aug.targets[index]
    for index in train_indices
]

class_counts = torch.bincount(
    torch.tensor(train_targets),
    minlength=2
)

print()
print(
    "Training class counts:"
)

for index, class_name in enumerate(
    full_train_aug.classes
):

    print(
        f"{class_name}: "
        f"{int(class_counts[index])}"
    )


class_sample_weights = (
    1.0
    / class_counts.float()
)

sample_weights = torch.tensor(
    [
        class_sample_weights[label]
        for label in train_targets
    ],
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# =========================================================
# DATALOADERS
# =========================================================

pin_memory = (
    device.type == "cuda"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)


# =========================================================
# MODEL
# =========================================================

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================================================
# SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# =========================================================
# BEST MODEL
# =========================================================

best_macro_f1 = -1.0
best_state = None
epochs_without_improvement = 0


# =========================================================
# TRAINING
# =========================================================

for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss = (
        train_loss_total
        / train_total
    )

    train_accuracy = (
        train_correct
        / train_total
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    val_labels = []
    val_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )


    val_loss = (
        val_loss_total
        / len(val_dataset)
    )

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    val_precision_macro = precision_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )

    val_recall_macro = recall_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )

    val_f1_macro = f1_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )


    scheduler.step(
        val_f1_macro
    )


    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    print()
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Macro Precision: "
        f"{val_precision_macro * 100:.2f}%"
    )

    print(
        f"Val Macro Recall: "
        f"{val_recall_macro * 100:.2f}%"
    )

    print(
        f"Val Macro F1: "
        f"{val_f1_macro * 100:.2f}%"
    )


    # -----------------------------------------------------
    # BEST CHECKPOINT
    # -----------------------------------------------------

    if val_f1_macro > best_macro_f1:

        best_macro_f1 = val_f1_macro

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state,
            BEST_MODEL_PATH
        )

        epochs_without_improvement = 0

        print(
            "✓ Best balanced model saved"
        )

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement."
        )


    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping."
        )

        break


# =========================================================
# LOAD BEST MODEL
# =========================================================

if best_state is None:

    raise RuntimeError(
        "Best model was not created."
    )

model.load_state_dict(
    best_state
)

model.eval()


# =========================================================
# SAVE FINAL MODEL
# =========================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print()
print(
    "Final model saved:"
)

print(
    FINAL_MODEL_PATH
)


# =========================================================
# TEST
# =========================================================

test_labels = []
test_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        test_labels.extend(
            labels.numpy()
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================================================
# TEST METRICS
# =========================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision_macro = precision_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_recall_macro = recall_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)

test_f1_macro = f1_score(
    test_labels,
    test_predictions,
    average="macro",
    zero_division=0
)


print()
print("=" * 60)
print(
    "FINAL TEST RESULTS"
)
print("=" * 60)

print(
    f"Accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision: "
    f"{test_precision_macro * 100:.2f}%"
)

print(
    f"Macro Recall   : "
    f"{test_recall_macro * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{test_f1_macro * 100:.2f}%"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print()
print(
    "Classification Report:"
)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=test_dataset.classes,
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print()
print(
    "Confusion Matrix:"
)

print(cm)


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Balanced ResNet18 Confusion Matrix"
)

plt.colorbar()

ticks = range(
    len(test_dataset.classes)
)

plt.xticks(
    ticks,
    test_dataset.classes,
    rotation=45
)

plt.yticks(
    ticks,
    test_dataset.classes
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

cm_path = (
    MODEL_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150
)

plt.close()

print()
print(
    "Confusion matrix saved:"
)

print(
    cm_path
)

print()
print("=" * 60)
print(
    "TRAINING COMPLETE"
)
print("=" * 60)