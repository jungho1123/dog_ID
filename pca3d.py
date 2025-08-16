
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score,roc_auc_score
from umap import UMAP

from dataset.pair_dataset_contrastive import PairDataset
from model.siamese_cosine import SiameseNetwork
from utils.transforms import get_val_transform
from utils.output_saver import get_output_path

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_embeddings(cfg, normalize=True):
    device = torch.device(cfg["experiment"]["device"])
    transform = get_val_transform(cfg["dataset"]["image_size"])
    dataset = PairDataset(cfg["dataset"]["val_csv"], image_root="data", transform1=transform, transform2=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = SiameseNetwork(cfg["model"]["name"], cfg["model"]["in_features"], cfg["model"]["feature_dim"]).to(device)
    model.load_state_dict(torch.load(Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"]))
    model.eval()

    z1_all, z2_all, labels, ids, img_paths1, img_paths2 = [], [], [], [], [], []

    with torch.no_grad():
        for img1, img2, label,weight, meta in tqdm(loader):
            img1, img2 = img1.to(device), img2.to(device)
            z1 = model.extract(img1, normalize=normalize).squeeze(0).cpu().numpy()
            z2 = model.extract(img2, normalize=normalize).squeeze(0).cpu().numpy()

            z1_all.append(z1)
            z2_all.append(z2)
            labels.append(label.item())
            ids.append(meta["id"])
            img_paths1.append(meta["img1_path"])
            img_paths2.append(meta["img2_path"])

    # 저장용 데이터프레임
    z1_all_np = np.stack(z1_all)
    z2_all_np = np.stack(z2_all)
    df = pd.DataFrame({
        "pair_id": ids,
        "img1_path": img_paths1,
        "img2_path": img_paths2,
        "label": labels
    })
    for i in range(z1_all_np.shape[1]):
        df[f"z1_{i}"] = z1_all_np[:, i]
        df[f"z2_{i}"] = z2_all_np[:, i]

    tag = "normalized" if normalize else "raw"
    df.to_csv(get_output_path(cfg, f"embeddings_{tag}.csv"), index=False)
    print(f"[INFO] Saved embeddings ({tag}) with metadata to CSV.")

    return z1_all_np, z2_all_np, np.array(labels), ids, img_paths1, img_paths2

# PCA 3D 시각화
def plot_pca_3d(z_all, labels_all, types, cfg):
    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z_all)
    df = pd.DataFrame(z_pca, columns=["PC1", "PC2", "PC3"])
    df["Label"] = labels_all
    df["Type"] = types

    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                        color=df["Label"].map({0: "Negative", 1: "Positive"}),
                        symbol="Type", opacity=0.7,
                        color_discrete_map={"Negative": "red", "Positive": "blue"},
                        title="3D PCA Embedding: z1 vs z2")
    fig.update_traces(marker=dict(size=1))
    fig.write_html(get_output_path(cfg, "pca3d_interactive.html"))

# UMAP 2D (static + interactive)
def plot_umap_2d(z_all, labels_all, types, cfg):
    reducer = UMAP(n_components=2, metric="cosine", init="random", random_state=42)
    z_umap = reducer.fit_transform(z_all)
    df = pd.DataFrame(z_umap, columns=["x", "y"])
    df["Label"] = labels_all
    df["Type"] = types

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="Label", style="Type", palette={0: "red", 1: "blue"}, s=20)
    plt.title("UMAP 2D Projection (cosine)")
    plt.savefig(get_output_path(cfg, "umap2d.png"))

    fig = px.scatter(df, x="x", y="y",
                     color=df["Label"].map({0: "Negative", 1: "Positive"}),
                     symbol="Type",
                     title="UMAP 2D Projection (interactive)",
                     opacity=0.7,
                     color_discrete_map={"Negative": "red", "Positive": "blue"})
    fig.update_traces(marker=dict(size=1))
    fig.write_html(get_output_path(cfg, "umap2d_interactive.html"))

# UMAP 3D

def plot_umap_3d(z_all, labels_all, types, cfg):
    reducer = UMAP(n_components=3, metric="cosine", init="random", random_state=42)
    z_umap = reducer.fit_transform(z_all)
    df = pd.DataFrame(z_umap, columns=["UMAP1", "UMAP2", "UMAP3"])
    df["Label"] = labels_all
    df["Type"] = types

    fig = px.scatter_3d(df, x="UMAP1", y="UMAP2", z="UMAP3",
                        color=df["Label"].map({0: "Negative", 1: "Positive"}),
                        symbol="Type",
                        title="3D UMAP Projection (cosine)", opacity=0.7,
                        color_discrete_map={"Negative": "red", "Positive": "blue"})
    fig.update_traces(marker=dict(size=1))
    fig.write_html(get_output_path(cfg, "umap3d_interactive.html"))

# z1 - z2 PCA

def plot_zdiff_pca(z1, z2, labels, cfg):
    zdiff = z1 - z2
    zdiff_pca = PCA(n_components=3).fit_transform(zdiff)
    df = pd.DataFrame(zdiff_pca, columns=["PC1", "PC2", "PC3"])
    df["Label"] = labels

    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                        color=df["Label"].map({0: "Negative", 1: "Positive"}),
                        opacity=0.7,
                        title="3D PCA of z1 - z2",
                        color_discrete_map={"Negative": "red", "Positive": "blue"})
    fig.update_traces(marker=dict(size=1))
    fig.write_html(get_output_path(cfg, "zdiff_pca3d.html"))

# cosine 유사도 히스토그램

def plot_cosine_histogram(z1, z2, labels, cfg):
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1))
    df = pd.DataFrame({"cosine": cos_sim, "label": labels})
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="cosine", hue="label", bins=50, kde=True,
                 palette={0: "red", 1: "blue"}, element="step", stat="density")
    plt.title("Cosine Similarity Distribution")
    plt.savefig(get_output_path(cfg, "cosine_hist.png"))

# feature-wise 표준편차 바그래프

def plot_feature_std_bar(z_all, cfg):
    stds = np.std(z_all, axis=0)
    plt.figure(figsize=(16, 5))
    bars = plt.bar(np.arange(len(stds)), stds, color="gray", edgecolor="black")
    plt.axhline(y=np.mean(stds), color='red', linestyle='--', label='Mean Std')
    plt.xlabel("Feature Dimension")
    plt.ylabel("Standard Deviation")
    plt.title("Feature-wise Standard Deviation (with Mean Line)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_output_path(cfg, "feature_std_bar.png"))

# confusion matrix 시각화

def plot_confusion_matrix(z1, z2, labels, cfg):
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1))
    preds = (cos_sim > 0.7).astype(int)
    cm = confusion_matrix(labels, preds)
    f1 = f1_score(labels, preds)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (F1 = {f1:.4f}) @ threshold=0.70")
    plt.tight_layout()
    plt.savefig(get_output_path(cfg, "confusion_matrix.png"))
    print("[INFO] Saved: confusion_matrix.png")

# cosine margin 기준 결정경계 시각화

def plot_decision_boundary_cosine(z1, z2, labels, cfg):
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1))
    preds = (cos_sim > 0.7).astype(int)

    zdiff = z1 - z2
    zdiff_pca = PCA(n_components=3).fit_transform(zdiff)

    df = pd.DataFrame(zdiff_pca, columns=["x", "y", "z"])
    df["Label"] = labels
    df["Pred"] = preds
    df["Match"] = (df["Label"] == df["Pred"]).astype(str)

    fig = px.scatter_3d(df, x="x", y="y", z="z",
                        color="Match", symbol="Label",
                        title="Decision Boundary (z1 - z2, cosine margin=0.7)")
    fig.update_traces(marker=dict(size=1))
    fig.write_html(get_output_path(cfg, "cosine_margin_decision_boundary.html"))

def plot_instance_scatter(z1, z2, labels, cfg):
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1))
    df = pd.DataFrame({"cosine": cos_sim, "label": labels})
    plt.figure(figsize=(6, 4))
    sns.stripplot(data=df, x="label", y="cosine", hue="label", palette={0: "red", 1: "blue"}, dodge=False, jitter=0.3,size=2)
    plt.title("Instance-wise Cosine Similarity Distribution")
    plt.xlabel("Label")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(get_output_path(cfg, "cosine_instance_scatter.png"))


def plot_angle_histogram(z1, z2, labels, cfg):
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1))
    angles = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
    df = pd.DataFrame({"angle": angles, "label": labels})
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="angle", hue="label", bins=50, element="step", palette={0: "red", 1: "blue"}, stat="density")
    plt.title("Angle Distribution between z1 and z2 Vectors")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(get_output_path(cfg, "angle_distribution.png"))


def extract_hard_samples(z1, z2, labels, ids, img1_paths, img2_paths, cfg):
    cos_sim = np.sum(z1 * z2, axis=1) / (np.linalg.norm(z1, axis=1) * np.linalg.norm(z2, axis=1))
    errors = np.abs(labels - cos_sim)
    
    df = pd.DataFrame({
        "pair_id": ids,
        "img1_path": img1_paths,
        "img2_path": img2_paths,
        "label": labels,
        "cosine": cos_sim,
        "error": errors
    })

    df_sorted = df.sort_values("error", ascending=False).head(20)
    df_sorted.to_csv(get_output_path(cfg, "top20_hard_samples.csv"), index=False)
    print("[INFO] Saved top 20 hardest samples to CSV (with image paths)")

# 메인 실행 함수
def main():
    cfg = load_config()
    prefix = cfg.get("visualization", {}).get("prefix", "")

    if prefix == "raw_":
        print("[INFO] Running RAW visualizations only")
        z1_raw, z2_raw, labels ,_ ,_ ,_= extract_embeddings(cfg, normalize=False)
        
        
        
        z_all_raw = np.concatenate([z1_raw, z2_raw], axis=0)
        labels_all = np.concatenate([labels, labels], axis=0)
        types = ["z1"] * len(labels) + ["z2"] * len(labels)

        plot_pca_3d(z_all_raw, labels_all, types, cfg)
        plot_umap_2d(z_all_raw, labels_all, types, cfg)
        plot_umap_3d(z_all_raw, labels_all, types, cfg)
        plot_zdiff_pca(z1_raw, z2_raw, labels, cfg)
        plot_feature_std_bar(z_all_raw, cfg)

    elif prefix == "norm_":
        print("[INFO] Running NORMALIZED cosine visualizations only")
        z1, z2, labels, ids, img1_paths, img2_paths = extract_embeddings(cfg, normalize=True)

        plot_cosine_histogram(z1, z2, labels, cfg)
        plot_confusion_matrix(z1, z2, labels, cfg)
        plot_decision_boundary_cosine(z1, z2, labels, cfg)
        plot_instance_scatter(z1, z2, labels, cfg)
        plot_angle_histogram(z1, z2, labels, cfg)
        extract_hard_samples(z1, z2, labels, ids, img1_paths, img2_paths, cfg)
        cos_sim = np.sum(z1 * z2, axis=1)
        best_f1, best_thresh, best_auc = 0, 0.5, 0
        for t in np.linspace(0, 1, 100):
            preds = (cos_sim > t).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_auc = roc_auc_score(labels, cos_sim)

        eval_txt_path = Path(cfg["save"]["save_dir"]) / f"{Path(cfg['save']['save_name']).stem}_{prefix}eval.txt"

        with open(eval_txt_path, "w") as f:
            f.write(f"F1 Score (best): {best_f1:.4f}\n")
            f.write(f"AUC: {best_auc:.4f}\n")
            f.write(f"Best Threshold: {best_thresh:.2f}\n")
            f.write(f"From model: {Path(cfg['save']['save_dir']) / cfg['save']['save_name']}\n")
        print(f"[INFO] Evaluation summary saved to: {eval_txt_path}")

    else:
        print("[WARNING] visualization.prefix 값이 'raw_' 또는 'norm_' 이어야 합니다.")


if __name__ == "__main__":
    main()
