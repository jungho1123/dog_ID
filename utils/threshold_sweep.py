# threshold_sweep.py (리팩터링 완료)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.output_saver import get_output_path


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sweep_threshold(cfg):
    df = pd.read_csv(get_output_path(cfg, "val_probs.csv"))

    thresholds = np.linspace(0, 1, 101)
    f1_list, precision_list, recall_list = [], [], []

    for th in thresholds:
        preds = (df["cosine_score"] > th).astype(int)
        f1 = f1_score(df["label"], preds, zero_division=0)
        precision = precision_score(df["label"], preds, zero_division=0)
        recall = recall_score(df["label"], preds, zero_division=0)

        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_list, label="F1 Score", linewidth=2)
    plt.plot(thresholds, precision_list, label="Precision", linewidth=2)
    plt.plot(thresholds, recall_list, label="Recall", linewidth=2)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs F1 / Precision / Recall")
    plt.legend()
    plt.grid(True)

    save_path = get_output_path(cfg, "threshold_sweep.png")
    plt.savefig(save_path)
    print(f"✅ Saved threshold_sweep.png → {save_path}")


if __name__ == "__main__":
    cfg = load_config()
    sweep_threshold(cfg)
