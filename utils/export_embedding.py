# export_embedding.py (v2 - z1+z2 통합 + label 확장)
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import yaml
from tqdm import tqdm
from dataset.pair_dataset_contrastive import PairDataset
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
    dataset = PairDataset(
        cfg["dataset"]["val_csv"],
        image_root="data",
        transform1=transform,
        transform2=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = SiameseNetwork(
        backbone_name=cfg["model"]["name"],
        in_features=cfg["model"]["in_features"],
        feature_dim=cfg["model"]["feature_dim"]
    ).to(device)
    model.load_state_dict(torch.load(Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"]))
    model.eval()

    z_all, labels_all = [], []

    for img1, img2, label, _ in tqdm(loader):
        img1, img2 = img1.to(device), img2.to(device)

        z1 = model.extract(img1).squeeze(0).cpu().numpy()
        z2 = model.extract(img2).squeeze(0).cpu().numpy()

        z_all.append(z1)
        z_all.append(z2)
        labels_all.append(label.item())
        labels_all.append(label.item())  # z1, z2 각각에 label 부여

    z_all = np.stack(z_all)
    labels_all = np.array(labels_all)

    np.save(get_output_path(cfg, "embeddings_all.npy"), z_all)
    np.save(get_output_path(cfg, "labels_all.npy"), labels_all)

    print(" Saved: embeddings_all.npy, labels_all.npy")


if __name__ == "__main__":
    main()
