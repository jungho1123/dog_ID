# merge_without_duplicates.py
import pandas as pd

def canonical_pair(row):
    """ img1, img2 쌍을 정렬된 튜플로 변환 → (작은 쪽, 큰 쪽) """
    return tuple(sorted([row["img1"], row["img2"]]))

def main():
    train_df = pd.read_csv("data/mini/train.csv")
    hard_df = pd.read_csv("hard_pairs.csv")

    # 💡 기준 쌍 정렬해서 비교하기 (img1, img2)와 (img2, img1)도 동일하게 취급
    train_df["pair_key"] = train_df.apply(canonical_pair, axis=1)
    hard_df["pair_key"] = hard_df.apply(canonical_pair, axis=1)

    # 📌 중복 제거: hard와 겹치는 쌍을 train에서 제거
    duplicated_keys = set(hard_df["pair_key"])
    train_df = train_df[~train_df["pair_key"].isin(duplicated_keys)]

    # 🔧 weight 컬럼 추가
    train_df["weight"] = 1.0
    hard_df["weight"] = 2.0

    # 병합
    merged_df = pd.concat([
        train_df.drop(columns=["pair_key"]),
        hard_df.drop(columns=["pair_key"])
    ], ignore_index=True)

    # 셔플 후 저장
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    merged_df.to_csv("train_merged.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved train_merged.csv (총 {len(merged_df)}개 쌍, 중복 제거 완료)")

if __name__ == "__main__":
    main()
