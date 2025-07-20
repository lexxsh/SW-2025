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
    AutoModelForSequenceClassification,
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
    help="Pretrained .pt 파일 경로",
)
parser.add_argument(
    "--sampling",
    nargs="+",  # 1개 이상 인자 허용
    type=int,
    default=None,
    help="Sampling strategy: "
    "1 arg → ratio (neg = ratio × pos) | "
    "2 args → exact counts POS NEG",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    help="Learning late",
)
parser.add_argument(
    "--scheduler_type",
    type=str,
    default="cosine",
    help="Scheduler type",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="Weight decay",
)
parser.add_argument(
    "--drop_out",
    type=float,
    default=0.2,
    help="Drop out",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Epochs",
)
parser.add_argument(
    "--test_size",
    type=float,
    default=0.2,
    help="Test size",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed",
)
args = parser.parse_args()

# ────── 1. 설정 ──────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "team-lucid/deberta-v3-base-korean"
MAX_LEN = 512


# 시드 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

# ────── 2. 데이터 로드 및 전처리 ──────
train = pd.read_csv(args.train_csv, encoding="utf-8-sig")

if "paragraphs" in train.columns:
    train = train.rename(columns={"paragraphs": "paragraph_text"})

if args.sampling:
    pos_df = train[train["generated"] == 1]
    neg_df = train[train["generated"] == 0]

    if len(args.sampling) == 2:
        pos_count, neg_count = args.sampling

        pos_sample = pos_df.sample(n=pos_count, random_state=args.seed)
        neg_sample = neg_df.sample(n=neg_count, random_state=args.seed)

        pos_sents = (pos_sample["title"] + " " + pos_sample["paragraph_text"]).tolist()
        neg_sents = (neg_sample["title"] + " " + neg_sample["paragraph_text"]).tolist()

    elif len(args.sampling) == 1:
        # 2) 언더샘플링: neg → args.sampling × pos 개수로 제한
        target_neg = min(len(neg_df), args.sampling[0] * len(pos_df))
        neg_sample = neg_df.sample(n=target_neg, random_state=42)

        # 3) 문장·라벨 합치기
        pos_sents = (pos_df["title"] + " " + pos_df["paragraph_text"]).tolist()
        neg_sents = (neg_sample["title"] + " " + neg_sample["paragraph_text"]).tolist()

    train_sents = pos_sents + neg_sents
    y = pd.Series([1] * len(pos_sents) + [0] * len(neg_sents)).reset_index(drop=True)
else:
    train_sents = (train["title"] + " " + train["paragraph_text"]).tolist()
    y = train["generated"]
    logger.info(f"Sampling 미적용 → 총 {len(y)}개")

X_train, X_val, y_train, y_val = train_test_split(
    train_sents, y, test_size=args.test_size, stratify=y, random_state=args.seed
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
        self.drop = nn.Dropout(args.drop_out)
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
if args.model_ckpt:
    model = WrappedClassifier(MODEL_ID).to(DEVICE)
    ckpt = torch.load(args.model_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2
    ).to(DEVICE)


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
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    metric_for_best_model="AUC",
    save_strategy="epoch",
    save_total_limit=3,
    lr_scheduler_type=args.scheduler_type,
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

os.makedirs(args.save_dir, exist_ok=True)
trainer.save_model(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
logger.info(f"✅ 모델 저장 완료: {args.save_dir}")
