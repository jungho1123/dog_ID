import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sklearn.manifold import TSNE
import umap
from utils.output_saver import get_output_path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def plot_embedding_2d(z, labels, method, cfg):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_neighbors=15, metric="cosine", random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    z_reduced = reducer.fit_transform(z)
    df = pd.DataFrame({"x": z_reduced[:, 0], "y": z_reduced[:, 1], "label": labels})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="Set1", alpha=0.7)
    plt.title(f"{method.upper()} Projection (z1 + z2)")
    plt.savefig(get_output_path(cfg, f"{method}_projection.png"))
    print(f"✅ Saved: {method}_projection.png")

def main():
    cfg = load_config()
    z = np.load(get_output_path(cfg, "embeddings_all.npy"))
    labels = np.load(get_output_path(cfg, "labels_all.npy"))

    plot_embedding_2d(z, labels, method="tsne", cfg=cfg)
    plot_embedding_2d(z, labels, method="umap", cfg=cfg)

if __name__ == "__main__":
    main()
