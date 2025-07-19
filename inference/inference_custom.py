#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py — 슬라이딩 윈도우 기반 학습 모델(SimpleClassifier)로 테스트셋 추론
"""

import os, torch, numpy as np, pandas as pd, logging, warnings
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import argparse
from tqdm import tqdm

# safetensors 라이브러리가 없다면 설치해야 합니다: pip install safetensors
from safetensors.torch import load_file

# ────────────────────── 0. Argument 파싱 ──────────────────────
parser = argparse.ArgumentParser()
# --model_path 인자에 .pt 파일이 아닌 체크포인트 폴더 경로를 지정합니다.
parser.add_argument(
    "--model_path",
    default="/home/elicer/ckpt_sota_with_hz_think/checkpoint-24818",
    help="모델 체크포인트 폴더 경로",
)
parser.add_argument(
    "--output_csv", default="./submission/submission_custom.csv", help="결과 저장 경로"
)
args = parser.parse_args()

# ────────────────────── 1. 설정 ──────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "team-lucid/deberta-v3-base-korean"
MAX_LEN, BATCH_SIZE = 512, 4


# ────────────────────── 2. 모델 정의 ──────────────────────
class SimpleClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        h = self.norm(cls_token)
        h = self.drop(h)
        return self.fc(h).squeeze(-1)


# ────────────────────── 3. 데이터 로딩 ──────────────────────
test_df = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
test_sents = (test_df["title"] + " " + test_df["paragraph_text"]).tolist()

# ────────────────────── 4. 전처리 ──────────────────────
# 토크나이저는 체크포인트 폴더가 아닌 원래 모델의 것을 사용해도 무방합니다.
# 만약 체크포인트에 커스텀 토크나이저가 있다면 args.model_path를 사용합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


test_dataset = TestDataset(test_sents, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ────────────────────── 5. 모델 로딩 (수정된 부분) ──────────────────────
# 먼저 모델의 전체 구조를 만듭니다.
model = SimpleClassifier(MODEL_NAME).to(DEVICE)

# 체크포인트 폴더 내의 실제 가중치 파일 경로를 지정합니다.
# 일반적으로 'model.safetensors' 또는 'pytorch_model.bin' 입니다.
weights_path = os.path.join(args.model_path, "model.safetensors")
if not os.path.exists(weights_path):
    weights_path = os.path.join(args.model_path, "pytorch_model.bin")

# 가중치 파일 확장자에 따라 다른 방식으로 state_dict를 로드합니다.
if weights_path.endswith(".bin"):
    state_dict = torch.load(weights_path, map_location=DEVICE)
else:
    # .safetensors 파일 로드
    state_dict = load_file(weights_path, device=str(DEVICE))

# 모델에 state_dict를 적용합니다.
model.load_state_dict(state_dict)
model.eval()
print(f"✅ 모델 로딩 완료: {weights_path}")


# ────────────────────── 6. 추론 ──────────────────────
all_probs = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="🔍 Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)

# ────────────────────── 7. 제출 파일 저장 ──────────────────────
sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
sub["generated"] = all_probs
output_path = f"{args.output_csv}"
sub.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 결과 저장 완료 → {output_path}")
