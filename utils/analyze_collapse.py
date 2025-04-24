# analyze_collapse.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.output_saver import get_output_path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_feature_heatmap(cfg, z_all):
    plt.figure(figsize=(12, 6))
    sns.heatmap(z_all[:50], cmap="viridis", cbar=True)
    plt.title("Feature Heatmap (First 50 Samples)")
    plt.xlabel("Feature Dimensions")
    plt.ylabel("Sample Index")
    plt.savefig(get_output_path(cfg, "feature_heatmap.png"))
    print("✅ Saved feature_heatmap.png")


def plot_std_bar(cfg, z_all):
    stds = np.std(z_all, axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(stds)), stds, color='orange')
    plt.title("Standard Deviation per Feature Dimension")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Std")
    plt.tight_layout()
    plt.savefig(get_output_path(cfg, "feature_std_bar.png"))
    print("✅ Saved feature_std_bar.png")


def print_collapse_stats(z_all):
    stds = np.std(z_all, axis=0)
    print("\n🔍 Collapse Diagnostic")
    print(f"Mean std: {stds.mean():.4f}")
    print(f"Std < 0.01: {(stds < 0.01).sum()} dims")
    print(f"Std < 0.02: {(stds < 0.02).sum()} dims")
    print(f"Std > 0.05: {(stds > 0.05).sum()} dims")


def main():
    cfg = load_config()
    z1 = np.load(get_output_path(cfg, "embeddings_z1.npy"))
    z2 = np.load(get_output_path(cfg, "embeddings_z2.npy"))
    z_all = np.concatenate([z1, z2], axis=0)

    plot_feature_heatmap(cfg, z_all)
    plot_std_bar(cfg, z_all)
    print_collapse_stats(z_all)


if __name__ == "__main__":
    main()
