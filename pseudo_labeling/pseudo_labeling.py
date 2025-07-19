import pandas as pd
import re
import os, torch, numpy as np, pandas as pd, logging, warnings
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="./ckpt/full_text_sliding_window/epoch_1.pt",
    help="모델 체크포인트(.pt) 경로",
)
parser.add_argument(
    "--train_csv", default="./data/train.csv", help="원본 학습 데이터셋"
)
parser.add_argument(
    "--output_csv",
    default="./data/train_pseudo_label.csv",
    help="결과 저장 경로",
)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "team-lucid/deberta-v3-base-korean"
MAX_LEN, BATCH_SIZE = 512, 4

# ───── 1. CSV 불러오기 ─────
df = pd.read_csv("./data/train.csv")


# ───── 2. 문단 분리 함수 수정 ─────
def split_into_paragraphs(text):
    # 한 줄 개행 기준으로 자르되, 완전히 빈 줄은 무시
    return [p.strip() for p in text.split("\n") if p.strip()]


# ───── 3. 행 단위로 문단 분리 및 새 DataFrame 생성 ─────
rows = []
for _, row in df.iterrows():
    paragraphs = split_into_paragraphs(row["full_text"])
    for idx, para in enumerate(paragraphs):
        rows.append(
            {
                "title": row["title"],
                "paragraph_index": idx,
                "paragraph_text": para,
                "generated": row["generated"],  # ⬅️ generated 값 유지
            }
        )

# ───── 4. 새 DataFrame 저장 ─────
paragraph_df = pd.DataFrame(rows)
paragraph_df_0 = paragraph_df[paragraph_df["generated"] == 0]
paragraph_df_1 = paragraph_df[paragraph_df["generated"] == 1]


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
test_df = paragraph_df_1
test_sents = (test_df["title"] + " " + test_df["paragraph_text"]).tolist()

# ────────────────────── 4. 전처리 ──────────────────────
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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ────────────────────── 5. 모델 로딩 ──────────────────────
model = SimpleClassifier(MODEL_NAME).to(DEVICE)
ckpt = torch.load(args.model_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

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
test_df["generated"] = all_probs
paragraph_df_1["generated"] = (test_df["generated"] > 0.5).astype(int)

# ───── 3. 병합 ─────
merged_df = pd.concat([paragraph_df_0, paragraph_df_1], ignore_index=True)
output_path = args.output_csv
merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 결과 저장 완료 → {output_path}")
