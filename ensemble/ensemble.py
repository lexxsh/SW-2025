import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm


def main(args):
    if len(args.csv_files) < 2:
        print("Error: 2개 이상의 CSV 파일을 입력해주세요.")
        return

    print("1. 첫 번째 CSV 파일을 기준으로 데이터프레임을 생성합니다.")
    # 첫 번째 파일을 기준으로 삼음
    ensemble_df = pd.read_csv(args.csv_files[0])
    # 예측 컬럼명을 'pred_0'으로 변경
    ensemble_df.rename(columns={"generated": "pred_0"}, inplace=True)

    print("2. 나머지 CSV 파일들을 읽어와 예측값을 추가합니다.")
    # 두 번째 파일부터 루프를 돌며 예측값 컬럼을 추가
    for i, file_path in enumerate(tqdm(args.csv_files[1:], desc="Reading CSVs")):
        df = pd.read_csv(file_path)
        ensemble_df[f"pred_{i+1}"] = df["generated"]

    print("\n3. 모든 예측값의 평균을 계산합니다.")
    # pred_ 로 시작하는 모든 컬럼을 선택
    pred_cols = [col for col in ensemble_df.columns if col.startswith("pred_")]
    # 각 행(row)에 대해 평균 계산
    ensemble_df["generated"] = ensemble_df[pred_cols].mean(axis=1)

    # ID와 최종 예측값만 남김
    submission_df = ensemble_df[["ID", "generated"]]

    print(f"4. 최종 앙상블 결과를 '{args.output_csv}' 파일로 저장합니다.")
    submission_df.to_csv(
        f"./submission/{args.output_csv}", index=False
    )
    print("✅ 앙상블 완료!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_files",
        required=True,
        nargs="+",
        help="앙상블할 CSV 파일 경로 목록 (2개 이상)",
    )
    ap.add_argument(
        "--output_csv",
        default="submission_ensemble.csv",
        help="결과를 저장할 CSV 파일명",
    )
    args = ap.parse_args()
    main(args)