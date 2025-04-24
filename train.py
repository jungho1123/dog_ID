# train.py 
import os
import yaml
import time
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.pair_dataset_contrastive import PairDataset
from model.siamese_cosine import SiameseNetwork
from loss.contrastive_loss import ContrastiveLoss
from utils.transforms import get_train_transform, get_val_transform
from sklearn.metrics import f1_score, roc_auc_score

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_best_f1_threshold(preds, labels):
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        preds_bin = (preds > t).astype(int)
        f1 = f1_score(labels, preds_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    auc = roc_auc_score(labels, preds)
    return best_f1, best_thresh, auc

def main():
    cfg = load_config()
    torch.manual_seed(cfg["experiment"]["seed"])
    device = torch.device(cfg["experiment"]["device"])

    transform_train = get_train_transform(cfg["dataset"]["image_size"])
    transform_val = get_val_transform(cfg["dataset"]["image_size"])

    image_root = cfg["dataset"]["image_root"]
    train_dataset = PairDataset(cfg["dataset"]["pair_csv"], image_root=image_root, transform1=transform_train, transform2=transform_train)
    val_dataset = PairDataset(cfg["dataset"]["val_csv"], image_root=image_root, transform1=transform_val, transform2=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    model = SiameseNetwork(
        backbone_name=cfg["model"]["name"],
        in_features=cfg["model"]["in_features"],
        feature_dim=cfg["model"]["feature_dim"],
        pretrained=cfg["model"].get("pretrained", False)
    ).to(device)

    criterion = ContrastiveLoss(margin=cfg["train"]["margin"], reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])

    best_f1 = 0.0
    save_dir = Path(cfg["save"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg["save"]["save_name"]

    log_lines = []
    start_time = time.time()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        print(f"\n Epoch [{epoch+1}/{cfg['train']['epochs']}]")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for img1, img2, label, weight ,_ in progress_bar:
            img1, img2 = img1.to(device), img2.to(device)
            label, weight = label.float().to(device), weight.to(device)

            optimizer.zero_grad()
            z1, z2 = model(img1, img2)
            loss = criterion(z1, z2, label)
            weighted_loss = (loss * weight).mean()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
            progress_bar.set_postfix(loss=weighted_loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f" Avg Train Loss: {avg_train_loss:.4f}")

        model.eval()
        cosine_scores, true_labels = [], []
        with torch.no_grad():
            for img1, img2, label, _ ,_ in val_loader:
                img1, img2 = img1.to(device), img2.to(device)
                z1 = model.extract(img1)
                z2 = model.extract(img2)
                cos_sim = torch.nn.functional.cosine_similarity(z1, z2).cpu().numpy()
                cosine_scores.extend(cos_sim)
                true_labels.extend(label.numpy())

        cosine_scores = np.array(cosine_scores)
        true_labels = np.array(true_labels)
        f1, best_thresh, auc = compute_best_f1_threshold(cosine_scores, true_labels)
        print(f" Val F1: {f1:.4f} | AUC: {auc:.4f} | Best Thresh: {best_thresh:.2f}")

        epoch_time = time.time() - epoch_start
        log_line = f"[Epoch {epoch+1}] Loss: {avg_train_loss:.4f} | Val F1: {f1:.4f} | AUC: {auc:.4f} | Thresh: {best_thresh:.2f} | Time: {epoch_time:.1f}s"
        log_lines.append(log_line)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f" Best model saved to: {save_path} (F1: {f1:.4f})")

    total_time = time.time() - start_time
    log_lines.append(f"\nTotal training time: {total_time:.1f} seconds")
    with open("train_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(" Saved log to train_log.txt")

if __name__ == "__main__":
    main()
