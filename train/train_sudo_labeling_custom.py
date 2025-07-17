#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_with_pt_model.py — SimpleClassifier 기반 저장된 .pt 모델로 재학습
"""

import os, gc, torch, numpy as np, pandas as pd, logging, warnings
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import random
import argparse
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

# ────── 0. Argument 파싱 ──────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_csv",
    type=str,
    default="./data/train_sudo_label.csv",
    help="Training CSV path",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./ckpt/train_sudo_custom",
    help="Checkpoint output dir",
)
parser.add_argument(
    "--model_ckpt",
    type=str,
    default="./ckpt/full_text_sliding_window/epoch_1.pt",
    help="Pretrained .pt 파일 경로",
)
parser.add_argument(
    "--sampling",
    nargs=2,  # 두 개의 숫자 (pos, neg)
    type=int,  # 정수형으로 변환
    default=None,  # 기본값: None (sampling X)
    metavar=("POS", "NEG"),  # 도움말에 표시할 이름
    help="Sample sizes: POS NEG",
)
args = parser.parse_args()

# ────── 1. 설정 ──────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "team-lucid/deberta-v3-base-korean"
MAX_LEN, BATCH_SIZE = 512, 4


# 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# ────── 2. 데이터 로드 및 전처리 ──────
train = pd.read_csv(args.train_csv, encoding="utf-8-sig")

if args.sampling:
    pos_count, neg_count = args.sampling
    pos_df = train[train["generated"] == 1]
    neg_df = train[train["generated"] == 0]
    pos_sample = pos_df.sample(n=pos_count, random_state=42)
    neg_sample = neg_df.sample(n=neg_count, random_state=42)

    pos_sents = (pos_sample["title"] + " " + pos_sample["paragraph_text"]).tolist()
    neg_sents = (neg_sample["title"] + " " + neg_sample["paragraph_text"]).tolist()
    train_sents = pos_sents + neg_sents
    y = pd.Series([1] * len(pos_sents) + [0] * len(neg_sents)).reset_index(drop=True)
    logger.info(f"Sampling 적용 → 총 {len(y)}개")
else:
    train_sents = (train["title"] + " " + train["paragraph_text"]).tolist()
    y = train["generated"]
    logger.info(f"Sampling 미적용 → 총 {len(y)}개")

X_train, X_val, y_train, y_val = train_test_split(
    train_sents, y, test_size=0.2, stratify=y, random_state=42
)

# ────── 3. HuggingFace Dataset 생성 ──────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
val_ds = Dataset.from_dict({"text": X_val, "label": y_val})
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)


# ────── 4. SimpleClassifier 정의 및 로딩 ──────
class SimpleClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        h = self.norm(cls_token)
        h = self.drop(h)
        logits = self.fc(h).squeeze(-1)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


# HuggingFace Trainer와 호환되게 감싸기
class WrappedClassifier(SimpleClassifier):
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = super().forward(input_ids, attention_mask, labels)
        return SequenceClassifierOutput(
            loss=output.get("loss", None),
            logits=output["logits"],
        )


# 모델 로드
model = WrappedClassifier(MODEL_ID).to(DEVICE)
ckpt = torch.load(args.model_ckpt, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])


# ────── 5. Metrics ──────
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(labels, probs)
    return {"AUC": auc}


# ────── 6. Trainer 설정 ──────
training_args = TrainingArguments(
    output_dir=args.save_dir,
    logging_dir="./logs",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    metric_for_best_model="AUC",
    save_strategy="epoch",
    save_total_limit=3,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ────── 7. 학습 ──────
trainer.train()

# ────── 8. 평가 및 저장 ──────
eval_result = trainer.evaluate()
logger.info(f"📊 Evaluation Results: {eval_result}")

trainer.save_model(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
logger.info(f"✅ 모델 저장 완료: {args.save_dir}")
