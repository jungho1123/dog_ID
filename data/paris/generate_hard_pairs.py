#하드마이닝용 csv뽑기
# generate_hard_pairs.py
import pandas as pd

def main():
    df = pd.read_csv("val_probs.csv")

    # 🎯 Hard Negative 조건: label=0 이면서 cosine_score이 0.3~0.6 사이
    hard_df = df[(df["label"] == 0) & (df["cosine_score"] >= 0.3) & (df["cosine_score"] <= 0.6)].copy()

    # 필요한 컬럼만 추출
    hard_df = hard_df[["img1", "img2", "label"]]

    # 저장
    hard_df.to_csv("hard_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved hard_pairs.csv ({len(hard_df)} samples)")

if __name__ == "__main__":
    main()
