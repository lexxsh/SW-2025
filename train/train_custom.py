import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_csv",
    type=str,
    default="./data/train.csv",
    help="Training CSV path",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="./ckpt/full_text",
    help="Checkpoint output dir",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=2e-5,
    help="Learning late",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Epochs",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed",
)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

model_name = "team-lucid/deberta-v3-base-korean"
train_csv_path = args.train_csv
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
max_length = 512
window_size = 400
overlap = 100

device_num = 0
device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


def sliding_window_split(text, tokenizer, window_size=400, overlap=100):
    """텍스트를 슬라이딩 윈도우로 분할"""
    tokens = tokenizer.tokenize(text)

    if len(tokens) <= window_size:
        return [text]

    windows = []
    start = 0

    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        window_tokens = tokens[start:end]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        windows.append(window_text)

        if end >= len(tokens):
            break
        start += window_size - overlap

    return windows


class SlidingWindowDataset(Dataset):
    """슬라이딩 윈도우 학습용 데이터셋"""

    def __init__(self, df: pd.DataFrame, tokenizer, window_size=400, overlap=100):
        self.windows = []
        self.labels = []

        print("🔄 Creating sliding windows...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing documents"):
            text = str(row["full_text"])
            label = int(row["generated"])

            windows = sliding_window_split(text, tokenizer, window_size, overlap)
            self.windows.extend(windows)
            self.labels.extend([label] * len(windows))

        self.tokenizer = tokenizer
        print(f"✅ Total windows created: {len(self.windows)}")

        # 클래스 분포 확인
        unique, counts = np.unique(self.labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"   Label {u}: {c:,} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        text = self.windows[idx]
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, attention_mask, label


class SimpleClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        d = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        h = self.norm(cls_token)
        h = self.drop(h)
        return self.fc(h).squeeze(-1)


def run_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for input_ids, attention_mask, label in tqdm(dataloader, desc="Training"):
        input_ids, attention_mask, label = [
            t.to(device) for t in (input_ids, attention_mask, label)
        ]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 정확도 계산
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct_predictions += (predictions == label).sum().item()
        total_predictions += label.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy


def train_sliding_window():
    print("🚀 RoBERTa-Base 슬라이딩 윈도우 학습 시작!")

    # 데이터 로드
    df = pd.read_csv(train_csv_path).dropna(subset=["full_text"])
    print(f"✅ Loaded {len(df)} documents")
    print("Document class distribution:")
    print(df["generated"].value_counts())

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 데이터셋 생성
    train_ds = SlidingWindowDataset(df, tokenizer, window_size, overlap)

    # DataLoader
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
    )

    # 모델, 옵티마이저, 손실함수
    model = SimpleClassifier(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 저장 디렉토리
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔥 Training started with {len(train_ds)} windows")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Epochs: {epochs}")

    # 학습
    best_accuracy = 0.0
    for ep in range(1, epochs + 1):
        print(f"\n📍 Epoch {ep}/{epochs}")

        train_loss, train_acc = run_epoch(model, train_dl, optimizer, criterion)

        print(f"[Epoch {ep}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 체크포인트 저장
        ckpt_path = os.path.join(save_dir, f"epoch_{ep}.pt")
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "config": {
                    "model_name": model_name,
                    "window_size": window_size,
                    "overlap": overlap,
                    "lr": lr,
                    "batch_size": batch_size,
                },
            },
            ckpt_path,
        )
        print(f"✅ Checkpoint saved → {ckpt_path}")

        if train_acc > best_accuracy:
            best_accuracy = train_acc
            # 최고 성능 모델 별도 저장
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                },
                best_path,
            )
            print(f"🎯 New best accuracy: {best_accuracy:.4f} → {best_path}")

    print(f"\n🎉 슬라이딩 윈도우 학습 완료!")
    print(f"📁 모델 저장 위치: {save_dir}")
    print(f"🏆 Best accuracy: {best_accuracy:.4f}")

    return save_dir


# ================= 실행 =================
if __name__ == "__main__":
    model_dir = train_sliding_window()
    print(f"\n✨ 슬라이딩 윈도우 학습 완료! 모델 위치: {model_dir}")
