import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import numpy as np

# --------------------
# Config
# --------------------
DATASET_DIR = "/Users/SaiSanjayD/Documents/PythonPrograms/CAID"

IMG_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")

IMG_SIZE = 128      # smaller = faster training
BATCH_SIZE = 4
EPOCHS = 3          # quick training
TRAIN_SPLIT = 0.8
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load official splits
# --------------------

def load_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

train_txt = "/Users/SaiSanjayD/Documents/PythonPrograms/CAID/ImageSets/Segmentation/train.txt"
val_txt   = "/Users/SaiSanjayD/Documents/PythonPrograms/CAID/ImageSets/Segmentation/val.txt"
test_txt  = "/Users/SaiSanjayD/Documents/PythonPrograms/CAID/ImageSets/Segmentation/test.txt"

train_files = load_split(train_txt)
val_files   = load_split(val_txt)
test_files  = load_split(test_txt)

# combine train + val
train_files = train_files + val_files

# --------------------
# Dataset
# --------------------
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

mask_tf = transforms.Resize(
    (IMG_SIZE, IMG_SIZE),
    interpolation=transforms.InterpolationMode.NEAREST
)

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

        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
        mask = (mask > 0).float()

        return image, mask

train_dataset = SegmentationDataset(train_files)
test_dataset = SegmentationDataset(test_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# --------------------
# Minimal CNN Model
# --------------------
class SimpleSegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = SimpleSegNet().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------
# Train
# --------------------
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    model.train()
    total_loss = 0

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

print("Training complete.")

# --------------------
# Pixel Accuracy
# --------------------
model.eval()

correct_pixels = 0
total_pixels = 0

with torch.no_grad():
    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        # Convert probabilities -> binary mask
        preds = (preds > 0.5).float()

        # Count matching pixels
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

pixel_accuracy = correct_pixels / total_pixels

print(f"Pixel Accuracy: {pixel_accuracy * 100:.2f}%")

# --------------------
# Show Predictions
# --------------------

fig, axes = plt.subplots(2, 3, figsize=(10, 7))

with torch.no_grad():
    for i in range(2):

        image, true_mask = test_dataset[i]

        input_img = image.unsqueeze(0).to(device)

        pred_mask = model(input_img)

        pred_mask = pred_mask.squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5)

        image_np = image.permute(1, 2, 0).numpy()
        true_mask_np = true_mask.squeeze().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Actual Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask_np, cmap="gray")
        axes[i, 1].set_title("Actual Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

