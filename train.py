import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet

from model import (
    Model1_MP_TR_BCE,
    Model2_MP_TR_DICE,
    Model3_Strided_TR_BCE,
    Model4_Strided_Bilinear_DICE,
    DiceLoss,
    BCEDiceLoss,
)

# ======================================================
# Config
# ======================================================
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Dataset
# ======================================================
class SegmentationTransform:
    def __init__(self, img_size):
        self.img_t = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.size = img_size

    def __call__(self, img, mask):
        img = self.img_t(img)

        # IMPORTANT: nearest interpolation for masks
        mask = T.functional.resize(
            mask,
            (self.size, self.size),
            interpolation=T.InterpolationMode.NEAREST,
        )

        mask = T.functional.pil_to_tensor(mask).long()

        # Oxford-IIIT Pet:
        # 1 = pet, 2 = border, 3 = background
        mask = (mask == 1).float()   # BINARY pet vs background

        return img, mask


def get_dataloaders():
    base_ds = OxfordIIITPet(
        root="./data",
        download=True,
        target_types="segmentation",
    )

    transform = SegmentationTransform(IMG_SIZE)

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, mask = self.base[idx]
            return self.transform(img, mask)

    full_ds = WrappedDataset(base_ds, transform)

    val_len = int(0.2 * len(full_ds))
    train_len = len(full_ds) - val_len

    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


# ======================================================
# Model Factory
# ======================================================
MODEL_FACTORY = {
    "model1": (Model1_MP_TR_BCE, nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([5.0]).to(DEVICE)  # class imbalance fix
    )),
    "model2": (Model2_MP_TR_DICE, DiceLoss()),
    "model3": (Model3_Strided_TR_BCE, nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([5.0]).to(DEVICE)
    )),
     "model4": (Model4_Strided_Bilinear_DICE,
               BCEDiceLoss(torch.tensor([5.0], device=DEVICE))),
}

# ======================================================
# Training Loop
# ======================================================
def train(model, criterion, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ------------------ Train ------------------
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    torch.save(model.state_dict(), "final_model.pth")
    print("Training complete. Models saved.")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["model1", "model2", "model3", "model4"],
        help="Which model variant to train",
    )
    args = parser.parse_args()

    model_cls, criterion = MODEL_FACTORY[args.model]
    model = model_cls().to(DEVICE)

    print(f"Training {args.model} on {DEVICE}")

    train_loader, val_loader = get_dataloaders()
    train(model, criterion, train_loader, val_loader)
