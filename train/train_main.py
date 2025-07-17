"""
train.py — DeBERTa-v3-base-korean 하이브리드 샘플링 분류기 파인튜닝
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

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True, help="Training CSV path")
parser.add_argument("--save_dir", type=str, required=True, help="Checkpoint output dir")
parser.add_argument("--sampling", type=bool, default=False, help="sampling")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda")
MODEL_ID = "team-lucid/deberta-v3-base-korean"
MAX_LEN, BATCH_SIZE = 512, 4  # 32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

train = pd.read_csv(args.train_csv, encoding="utf-8-sig")

if args.sampling:
    pos_df = train[train["generated"] == 1]
    neg_df = train[train["generated"] == 0]
    pos_sample = pos_df.sample(n=40000, random_state=42)
    neg_sample = neg_df.sample(n=10000, random_state=42)

    pos_sents = (pos_sample["title"] + " " + pos_sample["paragraphs"]).tolist()
    neg_sents = (neg_sample["title"] + " " + neg_sample["paragraphs"]).tolist()
    train_sents = pos_sents + neg_sents
    y = pd.Series([1] * len(pos_sents) + [0] * len(neg_sents)).reset_index(drop=True)
    logger.info(f"sampling O -> len: {len(pos_sample) + len(neg_sample)}")

else:
    train_sents = (train["title"] + " " + train["paragraphs"]).tolist()
    y = train["generated"]
    logger.info(f"sampling X -> len: {len(train)}")

X_train, X_val, y_train, y_val = train_test_split(
    train_sents, y, test_size=0.2, stratify=y, random_state=42
)

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

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(
    DEVICE
)


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    return {"AUC": auc}


training_args = TrainingArguments(
    output_dir=args.save_dir,
    logging_dir="./logs",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
    metric_for_best_model="AUC",
    # save_steps=20000,
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
trainer.train()

eval_result = trainer.evaluate()
logger.info(f"📊 Evaluation Results: {eval_result}")

trainer.save_model(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
logger.info(f"✅ Fine-tuned model saved to {args.save_dir}")