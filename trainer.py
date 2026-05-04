import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# --------------------
# Config
# --------------------
DATASET_DIR = "CAID"
IMG_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
EPOCHS = 5
LR = 1e-3

# --------------------
# Dataset (minimal)
# --------------------
files = sorted(os.listdir(IMG_DIR))

img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

mask_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.PILToTensor()
])

def load_sample(f):
    img = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
    mask = Image.open(os.path.join(MASK_DIR, f)).convert("L")
    return img_tf(img), mask_tf(mask).squeeze(0).long()

# --------------------
# Tiny model (very small CNN)
# --------------------
class TinySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1)  # assume binary-ish segmentation
        )

    def forward(self, x):
        return self.net(x)

model = TinySeg().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# --------------------
# Train loop
# --------------------
model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    loop = tqdm(files, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for f in loop:
        img, mask = load_sample(f)
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        img = img.unsqueeze(0)

        pred = model(img)

        loss = loss_fn(pred, mask.unsqueeze(0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {total_loss/len(files):.4f}")

# --------------------
# Evaluation (pixel accuracy)
# --------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    loop = tqdm(files, desc="Evaluating")

    for f in loop:
        img, mask = load_sample(f)
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        img = img.unsqueeze(0)
        pred = model(img)

        pred_class = pred.argmax(1).squeeze(0)

        correct += (pred_class == mask).sum().item()
        total += mask.numel()

        loop.set_postfix(acc=correct/total)

print(f"\nFinal Pixel Accuracy: {correct/total:.4f}")
