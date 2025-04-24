#utils/transforms.py
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

#CLAHE= 대비 향상 기법
#Unsharp Mask=원본과 블러 이미지 차이 강조 ->테두리,디테일 선명하게
#RandomHorizontalFlip 좌우반전
class CLAHEandSharpen:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), sigma=1.0):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.sigma = sigma

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

        blur = cv2.GaussianBlur(img_eq, (0, 0), sigmaX=self.sigma)
        sharpened = cv2.addWeighted(img_eq, 1.5, blur, -0.5, 0)

        return Image.fromarray(sharpened)


def get_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CLAHEandSharpen(clip_limit=2.0, tile_grid_size=(8, 8), sigma=1.0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
