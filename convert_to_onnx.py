import torch
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PTH_PATH = (
    PROJECT_ROOT
    / "model"
    / "chest_xray_resnet18.pth"
)

ONNX_PATH = (
    PROJECT_ROOT
    / "model"
    / "chest_xray_resnet18.onnx"
)


print("Loading PyTorch model...")

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2,
)

state_dict = torch.load(
    PTH_PATH,
    map_location="cpu",
)

model.load_state_dict(state_dict)

model.eval()

dummy_input = torch.randn(
    1,
    3,
    224,
    224,
)

print("Converting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    dynamo=False,
)

print()
print("ONNX model created:")
print(ONNX_PATH)

print(
    "Size:",
    round(
        ONNX_PATH.stat().st_size / (1024 * 1024),
        2,
    ),
    "MB",
)