 # DOG_NOSE_COS: 강아지 비문 인식 프로젝트

강아지의 코(비문)를 활용한 생체인식 시스템입니다.  
Siamese Network 구조를 기반으로, ResNeXt50-IBN 및 SE-ResNet 백본을 사용하여  
특징 벡터를 추출하고 cosine similarity로 본인 여부를 판별합니다.

---

##  주요 기능

- 비문 이미지 전처리 (CLAHE + Sobel 등)
- Siamese 기반 벡터 추출 모델
- Cosine Similarity 기반 검증 로직
- PyTorch 기반 모델 학습 및 추론
- Firebase 연동 계획 포함

---

##  사용 기술 스택

- Python 3.10
- PyTorch
- timm (Backbone)
- IBN-Net (Custom CNN)
- wandb, OpenCV, numpy, matplotlib 등

---

##  디렉토리 구조

- 추가예정


---

##  라이선스

- 이 프로젝트의 코드는 MIT License를 따릅니다.
- `backbone/timm`, `backbone/ibn_net` 디렉토리의 외부 코드들은 각자의 라이센스를 따릅니다.  
  자세한 내용은 `THIRD_PARTY.md`를 참고해주세요.

---

##  기여 / 문의

문의: [snsdl1123@naver.com]

