"""
inference.py — 파인튜닝된 DeBERTa 모델로 테스트셋 추론 & 제출 파일 생성
"""

import os, torch, numpy as np, pandas as pd, logging, warnings
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)
from datasets import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="모델 체크포인트(.pt) 경로")
parser.add_argument("--output_csv", default="./submission/submission_main.csv", help="결과 저장 경로")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda")
MODEL_DIR = args.model_path  # train.py에서 저장된 경로
MAX_LEN, BATCH_SIZE = 512, 4

test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
test_sents = (test["title"] + " " + test["paragraph_text"]).tolist()

# ────────────────────── 2. 전처리 ──────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


test_ds = Dataset.from_dict({"text": test_sents})
test_ds = test_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

# ────────────────────── 3. 모델 로드 ──────────────────────
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ────────────────────── 4. 예측 ──────────────────────
test_logits = trainer.predict(test_ds).predictions
test_probs = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()

# ────────────────────── 5. 제출 파일 저장 ──────────────────────
sub = pd.read_csv(
    "./data/sample_submission.csv", encoding="utf-8-sig"
)
sub["generated"] = test_probs
sub.to_csv(
    f"./submission/{args.output_csv}",
    index=False,
    encoding="utf-8-sig",
)
logger.info(f"✅ Saved → ./submission/{args.output_csv}")