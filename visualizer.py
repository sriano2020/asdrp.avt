import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import webbrowser
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from transformers import SegformerForSemanticSegmentation

# --------------------
# Config
# --------------------
DATASET_DIR = "CAID"

IMG_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")

MODEL_DIR = "/Users/SaiSanjayD/Documents/PythonPrograms/segformer_hf_model"

IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load test split
# --------------------
def load_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

test_txt = "CAID/ImageSets/Segmentation/test.txt"
test_files = load_split(test_txt)

# --------------------
# Transforms
# --------------------
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

mask_tf = transforms.Resize(
    (IMG_SIZE, IMG_SIZE),
    interpolation=transforms.InterpolationMode.NEAREST
)

# --------------------
# Dataset
# --------------------
class SegmentationDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        img_path = os.path.join(IMG_DIR, filename + ".png")
        mask_path = os.path.join(MASK_DIR, filename + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = img_tf(image)
        mask = mask_tf(mask)

        mask = torch.tensor(
            np.array(mask),
            dtype=torch.float32
        ).unsqueeze(0)

        # binary mask
        mask = (mask > 0).float()

        return image, mask

test_dataset = SegmentationDataset(test_files)

# --------------------
# Wrapper model
# --------------------
class SegFormerBinary(nn.Module):
    def __init__(self, model_dir):
        super().__init__()

        # Load saved HuggingFace model
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            model_dir
        )

    def forward(self, x):
        logits = self.net(pixel_values=x).logits

        # Upsample to input size
        logits = nn.functional.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        probs = torch.sigmoid(logits)
        return probs

# --------------------
# Load model
# --------------------
model = SegFormerBinary(MODEL_DIR).to(device)
model.eval()

print("Model loaded successfully.")

# --------------------
# Show Predictions in Browser
# --------------------
NUM_IMAGES = 10

indices = random.sample(
    range(len(test_dataset)),
    NUM_IMAGES
)

html = """
<html>
<head>
<title>SegFormer Predictions</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 20px;
    background: #f5f5f5;
}

.row {
    display: flex;
    gap: 20px;
    margin-bottom: 40px;
    align-items: center;
}

.card {
    background: white;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.15);
    text-align: center;
}

img {
    width: 256px;
    border-radius: 6px;
}

h2 {
    margin-bottom: 30px;
}
</style>
</head>
<body>
<h1>SegFormer Predictions</h1>
"""

with torch.no_grad():

    for idx in indices:

        image, true_mask = test_dataset[idx]

        input_img = image.unsqueeze(0).to(device)

        pred_mask = model(input_img)

        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5)

        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        true_mask_np = (
            true_mask.squeeze().numpy() * 255
        ).astype(np.uint8)

        pred_mask_np = (
            pred_mask.astype(np.uint8) * 255
        )

        # Save temp images
        image_path = f"temp_img_{idx}.png"
        gt_path = f"temp_gt_{idx}.png"
        pred_path = f"temp_pred_{idx}.png"

        Image.fromarray(image_np).save(image_path)
        Image.fromarray(true_mask_np).save(gt_path)
        Image.fromarray(pred_mask_np).save(pred_path)

        html += f"""
        <h2>{test_files[idx]}</h2>

        <div class="row">

            <div class="card">
                <h3>Actual Image</h3>
                <img src="{image_path}">
            </div>

            <div class="card">
                <h3>Actual Mask</h3>
                <img src="{gt_path}">
            </div>

            <div class="card">
                <h3>Predicted Mask</h3>
                <img src="{pred_path}">
            </div>

        </div>
        """

html += "</body></html>"

# Save HTML file
output_path = Path("segformer_predictions.html")
output_path.write_text(html)

# Open in browser
webbrowser.open(f"file://{output_path.resolve()}")

print("Opened predictions in browser.")
