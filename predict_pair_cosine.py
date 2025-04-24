# predict_pair_cosine.py

import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import cosine_similarity, normalize

from utils.transforms import get_val_transform
from model.siamese_cosine import SiameseNetwork


def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    device = torch.device(cfg["experiment"]["device"])
    transform = get_val_transform(cfg["dataset"]["image_size"])

    # 모델 로드
    model = SiameseNetwork(
        backbone_name=cfg["model"]["name"],
        in_features=cfg["model"]["in_features"],
        feature_dim=cfg["model"]["feature_dim"],
        pretrained=False  # ❗ 학습된 가중치 로드 예정
    ).to(device)

    ckpt_path = Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"]
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # CSV 불러오기
    df = pd.read_csv(cfg["dataset"]["val_csv"])
    image_root = Path(cfg["dataset"]["image_root"])

    cosine_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img1_path = image_root / row["img1"]
        img2_path = image_root / row["img2"]

        try:
            img1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
            img2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                z1 = normalize(model.extract(img1), dim=1)
                z2 = normalize(model.extract(img2), dim=1)
                sim = cosine_similarity(z1, z2).item()
        except Exception as e:
            print(f"❌ Failed: {img1_path} | {img2_path}")
            sim = -1  # fallback

        cosine_scores.append(sim)

    df["cosine"] = cosine_scores
    df.to_csv("val_probs.csv", index=False)
    print("✅ Saved: val_probs.csv")


if __name__ == "__main__":
    main()
