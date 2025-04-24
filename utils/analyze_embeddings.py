# analyze_embeddings.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import cosine_similarity, normalize
from dataset.pair_dataset_contrastive import PairDataset
from model.siamese_cosine import SiameseNetwork
from utils.transforms import get_val_transform
from utils.output_saver import get_output_path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def extract_embeddings(cfg):
    device = torch.device(cfg["experiment"]["device"])

    transform = get_val_transform(cfg["dataset"]["image_size"])
    dataset = PairDataset(cfg["dataset"]["val_csv"], image_root="data",
                          transform1=transform, transform2=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = SiameseNetwork(
        backbone_name=cfg["model"]["name"],
        in_features=cfg["model"]["in_features"],
        feature_dim=cfg["model"]["feature_dim"]
    ).to(device)
    model.load_state_dict(torch.load(Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"]))
    model.eval()

    z1_list, z2_list, labels = [], [], []

    for img1, img2, label, _ in tqdm(loader):
        img1, img2 = img1.to(device), img2.to(device)
        
        z1 = model.extract(img1)
        z2 = model.extract(img2)
        z1_list.append(z1.squeeze(0).cpu().numpy())
        z2_list.append(z2.squeeze(0).cpu().numpy())
        labels.append(label.item())

    return np.array(z1_list), np.array(z2_list), np.array(labels)

def plot_umap(cfg, z1, z2, labels):
    z_all = np.concatenate([z1, z2], axis=0)
    y_all = np.concatenate([labels, labels], axis=0)

    reducer = umap.UMAP(n_neighbors=15, metric="cosine", random_state=42)
    z_umap = reducer.fit_transform(z_all)

    df = pd.DataFrame({"x": z_umap[:, 0], "y": z_umap[:, 1], "label": y_all})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="Set1", alpha=0.7)
    plt.title("UMAP (cosine) projection")
    plt.savefig(get_output_path(cfg, "umap_cosine.png"))
    print("✅ Saved umap_cosine.png")

def plot_cosine_histogram(cfg, z1, z2):
    cos_sim = cosine_similarity(torch.tensor(z1), torch.tensor(z2), dim=1).numpy()
    plt.figure(figsize=(7, 5))
    sns.histplot(cos_sim, bins=50, kde=True, color="dodgerblue")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("cos(z1, z2)")
    plt.savefig(get_output_path(cfg, "cosine_hist.png"))
    print("✅ Saved cosine_hist.png")

def plot_feature_std_histogram(cfg, z_all):
    stds = np.std(z_all, axis=0)
    plt.figure(figsize=(7, 5))
    sns.histplot(stds, bins=40, kde=True, color="darkorange")
    plt.title("Per-Dimension Std of Embeddings")
    plt.xlabel("Feature Dimension Std")
    plt.savefig(get_output_path(cfg, "feature_std_hist.png"))
    print("✅ Saved feature_std_hist.png")


def main():
    cfg = load_config()
    z1, z2, labels = extract_embeddings(cfg)

    plot_umap(cfg, z1, z2, labels)
    plot_cosine_histogram(cfg, z1, z2)
    plot_feature_std_histogram(cfg, np.concatenate([z1, z2], axis=0))


if __name__ == "__main__":
    main()
