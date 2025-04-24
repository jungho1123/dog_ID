import os
import random
import pandas as pd
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

def generate_pairs_csv(data_dir, output_csv_path, prefix="mini/train", pos_neg_ratio=2):
    data_dir = Path(data_dir)
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    all_data = []

    # 클래스별 이미지 경로 수집
    class_to_images = {}
    for class_dir in class_dirs:
        images = sorted(class_dir.glob("*.jpg"))
        if len(images) >= 2:
            class_to_images[class_dir.name] = images

    # 양성쌍 생성
    pos_pairs = []
    for class_id, images in tqdm(class_to_images.items(), desc=f"Generating positive pairs for {data_dir.name}"):
        for img1, img2 in combinations(images, 2):
            pos_pairs.append({
                'img1': f"{prefix}/{class_id}/{img1.name}",
                'img2': f"{prefix}/{class_id}/{img2.name}",
                'label': 1
            })

    # 음성쌍 생성 (positive:negative = 1:2)
    neg_pairs = []
    class_ids = list(class_to_images.keys())
    for _ in tqdm(range(len(pos_pairs) * pos_neg_ratio), desc=f"Generating negative pairs for {data_dir.name}"):
        while True:
            cls1, cls2 = random.sample(class_ids, 2)
            img1_list, img2_list = class_to_images[cls1], class_to_images[cls2]
            if img1_list and img2_list:
                img1 = random.choice(img1_list)
                img2 = random.choice(img2_list)
                neg_pairs.append({
                    'img1': f"{prefix}/{cls1}/{img1.name}",
                    'img2': f"{prefix}/{cls2}/{img2.name}",
                    'label': 0
                })
                break

    # 합치고 저장
    all_data = pos_pairs + neg_pairs
    random.shuffle(all_data)
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False)
    return df

# 경로 설정

base_dir = Path("E:/download/dog_nose/dog_nose_cos/data/mini")
train_dir = base_dir / "train"
val_dir = base_dir / "val"

# 출력 경로
train_csv_path = base_dir / "train.csv"
val_csv_path = base_dir / "val.csv"

# 생성 실행 (prefix만 바뀌면 경로가 자동 조정됨)
df_train = generate_pairs_csv(train_dir, train_csv_path, prefix="mini/train")
df_val = generate_pairs_csv(val_dir, val_csv_path, prefix="mini/val")
