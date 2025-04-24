# generate_val_probs.py
#val 이미지 쌍에 대해 cosine 유사도 계산하고 val_probs.csv 저장
#cosine score 분포 확인용
#threshold sweep, 분석, 하드마이닝 등에 쓰임
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch.nn.functional import normalize
from dataset import PairDataset
from model.siamese_cosine import SiameseNetwork
from utils.transforms import get_val_transform
from utils.output_saver import get_output_path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    cfg = load_config()
    device = torch.device(cfg["experiment"]["device"])

    transform = get_val_transform(cfg["dataset"]["image_size"])
    val_dataset = PairDataset(cfg["dataset"]["val_csv"], image_root=cfg["dataset"]["image_root"], transform1=transform, transform2=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SiameseNetwork(
        backbone_name=cfg["model"]["name"],
        in_features=cfg["model"]["in_features"],
        feature_dim=cfg["model"]["feature_dim"],
        pretrained=False
    ).to(device)

    model.load_state_dict(torch.load(Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"], map_location=device))
    model.eval()

    rows = []
    for (img1, img2, label, _), sample in zip(tqdm(val_loader), val_dataset.data):
        img1, img2 = img1.to(device), img2.to(device)
        z1 = model.extract(img1)
        z2 = model.extract(img2)
        cos_sim = torch.nn.functional.cosine_similarity(z1, z2).cpu().numpy()

        rows.append({
            "img1": str(sample["img1"]),
            "img2": str(sample["img2"]),
            "label": int(sample["label"]),
            "cosine_score": cos_sim
        })

    df = pd.DataFrame(rows)
    df.to_csv(get_output_path(cfg, "val_probs.csv"), index=False, encoding="utf-8-sig")
    print("✅ Saved: val_probs.csv")


if __name__ == "__main__":
    main()
