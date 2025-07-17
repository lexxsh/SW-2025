import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm


def confidence_top2_prob_diff_voting(model_probs, threshold=0.1):
    """
    model_probs: (N, M) 배열 - N개의 샘플, M개의 모델 확률 예측값
    threshold: 확률값 차이 기준 (예: 0.1)
    """
    N, M = model_probs.shape

    # 1. confidence 계산
    confidence = np.abs(model_probs - 0.5)  # (N, M)

    # 2. confidence 기준 내림차순 정렬 인덱스
    top2_indices = np.argsort(-confidence, axis=1)[:, :3]  # (N, 2)

    # 3. top2 확률 추출
    row_indices = np.arange(N)[:, None]
    top2_probs = model_probs[row_indices, top2_indices]  # (N, 2)
    prob_diff = np.abs(top2_probs[:, 0] - top2_probs[:, 1])  # (N,)

    # 4. 조건 분기
    top2_mean = top2_probs.mean(axis=1)
    full_mean = model_probs.mean(axis=1)

    final_probs = np.where(prob_diff <= threshold, top2_mean, full_mean)
    return final_probs


def main(args):
    if len(args.csv_files) < 2:
        print("❌ Error: 2개 이상의 CSV 파일을 입력해주세요.")
        return

    print("📂 1. 첫 번째 CSV 파일을 기준으로 데이터프레임을 생성합니다.")
    # 첫 번째 파일을 기준으로 삼음
    ensemble_df = pd.read_csv(args.csv_files[0])
    ensemble_df.rename(columns={"generated": "pred_0"}, inplace=True)

    print("📥 2. 나머지 CSV 파일들을 읽어와 예측값을 추가합니다.")
    for i, file_path in enumerate(tqdm(args.csv_files[1:], desc="Reading CSVs")):
        df = pd.read_csv(file_path)
        ensemble_df[f"pred_{i+1}"] = df["generated"]

    print("\n⚙️  3. confidence top-2 기반 앙상블 전략을 적용합니다.")
    # pred_ 컬럼만 선택
    pred_cols = [col for col in ensemble_df.columns if col.startswith("pred_")]
    model_probs = ensemble_df[pred_cols].values  # (N, M)

    # 앙상블 계산
    final_probs = confidence_top2_prob_diff_voting(
        model_probs, threshold=args.threshold
    )
    ensemble_df["generated"] = final_probs

    # ID와 최종 결과만 저장
    submission_df = ensemble_df[["ID", "generated"]]
    submission_df.to_csv(
        f"/shared/home/kdd/HZ/sw/Z_ensembled/{args.output_csv}", index=False
    )

    print(f"✅ 4. 앙상블 결과를 저장했습니다 → {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_files",
        required=True,
        nargs="+",
        help="앙상블할 CSV 파일 경로 목록 (2개 이상)",
    )
    parser.add_argument(
        "--output_csv",
        default="submission_ensemble.csv",
        help="결과를 저장할 CSV 파일명",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="top2 확률값 차이 허용 임계값 (기본: 0.1)",
    )
    args = parser.parse_args()
    main(args)