import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """
    Dataset for Contrastive Learning (dog nose pairs)
    Supports image_root, sample weight, and dual transforms.
    """

    def __init__(self, csv_path: str, image_root: str = ".", transform1=None, transform2=None):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform1 = transform1
        self.transform2 = transform2
        self.data = []

        with open(self.csv_path, "r", newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                self.data.append({
                    "img1": Path(row["img1"]),
                    "img2": Path(row["img2"]),
                    "label": float(row["label"]),
                    "weight": float(row.get("weight", 1.0)),
                    "id": i  # ID: row index (정수)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        path1 = self.image_root / row["img1"]
        path2 = self.image_root / row["img2"]

        try:
            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")
        except Exception as e:
            print(f"[Image Load Error] {path1} or {path2}")
            raise e

        if self.transform1:
            img1 = self.transform1(img1)
        if self.transform2:
            img2 = self.transform2(img2)

        label = torch.tensor(row["label"], dtype=torch.float32)
        weight = torch.tensor(row["weight"], dtype=torch.float32)

        meta = {
            "id": row["id"],
            "img1_path": str(row["img1"]),
            "img2_path": str(row["img2"]),
        }

        return img1, img2, label, weight, meta
