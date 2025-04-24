import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils.transforms import get_val_transform
import pandas as pd
import numpy as np
from collections import defaultdict
from model.siamese_cosine import SiameseNetwork

# 설정
root_dir = Path("data/mini/train")
output_csv = "checkpoints/seresnext50_ibn_custom_nose_vectors_mean.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리
transform = get_val_transform(img_size=224)

# 데이터셋 로딩
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 모델 로딩
model = SiameseNetwork(
    backbone_name="seresnext50_ibn_custom",
    in_features=2048,
    feature_dim=512,
    pretrained=False
)
model.load_state_dict(torch.load("checkpoints/seresnext50_ibn_custom.pth", map_location=device))
model.to(device)
model.eval()

# 클래스별 벡터 저장
class_vectors = defaultdict(list)

for img, label in loader:
    img = img.to(device)
    with torch.no_grad():
        z = model.extract(img, normalize=False)
        z = z.squeeze(0).cpu()
        class_id = dataset.classes[label.item()]
        class_vectors[class_id].append(z)

# 클래스별 전체 평균 → 정규화 → CSV 저장
data = []
for class_id, vectors in class_vectors.items():
    if len(vectors) == 0:
        continue
    stacked = torch.stack(vectors, dim=0)
    mean_vec = stacked.mean(dim=0)
    norm_vec = F.normalize(mean_vec, dim=0).numpy()

    admin_id = class_id.split("/")[0] if "/" in class_id else class_id.split("_")[0]
    row = {
        "admin_id": "admin_" + str(admin_id),
        "class_id": class_id,
        **{f"f{i}": norm_vec[i] for i in range(512)}
    }
    data.append(row)

# 저장
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding="utf-8")
