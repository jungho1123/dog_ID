# # analyze_pca3d.py
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# import yaml
# from mpl_toolkits.mplot3d import Axes3D

# from dataset import PairDataset
# from model.siamese_cosine import SiameseNetwork
# from utils.transforms import get_val_transform
# from utils.output_saver import get_output_path


# def load_config(path="config.yaml"):
#     with open(path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)


# @torch.no_grad()
# def main():
#     cfg = load_config()
#     device = torch.device(cfg["experiment"]["device"])

#     transform = get_val_transform(cfg["dataset"]["image_size"])
#     dataset = PairDataset(cfg["dataset"]["val_csv"], image_root="data", transform1=transform, transform2=transform)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     model = SiameseNetwork(
#         backbone_name=cfg["model"]["name"],
#         in_features=cfg["model"]["in_features"],
#         feature_dim=cfg["model"]["feature_dim"],
#         pretrained=cfg["model"].get("pretrained", False)
#     ).to(device)
#     model.load_state_dict(torch.load(Path(cfg["save"]["save_dir"]) / cfg["save"]["save_name"]))
#     model.eval()

#     z1_list, labels = [], []

#     for img1, img2, label, _ in loader:
#         img1 = img1.to(device)
#         z1 = model.extract(img1)
#         z1 = torch.nn.functional.normalize(z1, dim=1)
#         z1_list.append(z1.squeeze(0).cpu().numpy())
#         labels.append(label.item())

#     z1_array = np.stack(z1_list)
#     labels = np.array(labels)

#     # 3D PCA
#     pca = PCA(n_components=3)
#     z1_3d = pca.fit_transform(z1_array)

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter(z1_3d[:, 0], z1_3d[:, 1], z1_3d[:, 2], c=labels, cmap='coolwarm', alpha=0.7)

#     ax.set_title("3D PCA of Embeddings")
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     ax.set_zlabel("PC3")
#     plt.colorbar(scatter)

#     save_path = get_output_path(cfg, "pca3d_embedding.png")
#     plt.savefig(save_path)
#     print(f"✅ Saved: {save_path}")


# if __name__ == "__main__":
#     main()
