#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, random, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # ← tqdm 추가


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# ─────────────────────── 1. 설정 ──────────────────────────────────────────────
MODEL_PATH = "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"
BATCH_SIZE = 32
INPUT_CSV = "./data/train_human_paragraphs.csv"
NUM_SAMPLES = 18036
OUT_FULL = f"./data/train_llama.csv"
OUT_CHANGED = f"./data/train_generated_only_llama_3_1_8B_{NUM_SAMPLES//1000}k.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────── 2. 프롬프트 템플릿 ────────────────────────────────────
PROMPTS = {
    "narrative_essay": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 감수성이 풍부한 작가입니다. 주어진 글을 바탕으로, 개인적인 경험이나 감정을 녹여내어 독자의 마음에 울림을 주는 부드럽고 서정적인 '수필' 형식으로 문체를 바꿔주세요. 원문의 핵심 정보는 유지해야 합니다. 분량도 유지해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

**[중요] 오직 변환된 본문만 답변하고, 괄호를 사용한 설명이나 "(최종 버전을 제공합니다.)" 와 같은 부가적인 코멘트는 절대 출력하지 마세요. 또한 반문하지 마세요.**

{paragraph}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    "logical_essay": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 논리적인 칼럼니스트입니다. 주어진 글의 내용을 바탕으로, 명확한 주장이나 의견을 설득력 있게 전달하는 지적인 '에세이' 형식으로 문체를 바꿔주세요. 분량도 유지해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

**[중요] 오직 변환된 본문만 답변하고, 괄호를 사용한 설명이나 "(최종 버전을 제공합니다.)" 와 같은 부가적인 코멘트는 절대 출력하지 마세요. 또한 반문하지 마세요.**

{paragraph}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    "civic_essay": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 뛰어난 설명가입니다. 주어진 글의 내용을 독자들이 알아듯기 쉽게 설명해주세요. 설명체 + 확인형 종결어미 조합으로 설명해주세요. 분량도 유지해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

**[중요] 오직 변환된 본문만 답변하고, 괄호를 사용한 설명이나 "(최종 버전을 제공합니다.)" 와 같은 부가적인 코멘트는 절대 출력하지 마세요. 또한 반문하지 마세요.**

{paragraph}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    "news": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 냉철한 기자입니다. 주어진 글에서 감정적인 표현은 모두 배제하고, 객관적인 사실과 정보만을 사용하여 육하원칙에 따라 간결하고 명료한 '뉴스 기사' 형식으로 문체를 바꿔주세요.  분량도 유지해주세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

**[중요] 오직 변환된 본문만 답변하고, 괄호를 사용한 설명이나 "(최종 버전을 제공합니다.)" 와 같은 부가적인 코멘트는 절대 출력하지 마세요. 또한 반문하지 마세요.**

{paragraph}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    "summary": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 뛰어난 요약 전문가입니다. 주어진 글의 핵심 내용을 간결하게 '요약'해주세요. 단, 원래 글의 문체와 톤은 최대한 유지하면서 요약해야 합니다. 스타일을 바꾸지 마세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

**[중요] 오직 변환된 본문만 답변하고, 괄호를 사용한 설명이나 "(최종 버전을 제공합니다.)" 와 같은 부가적인 코멘트는 절대 출력하지 마세요. 또한 반문하지 마세요.**

{paragraph}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
}
STYLE_LIST = list(PROMPTS.keys())

# ─────────────────────── 3. 모델 로딩 ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("style-aug")
warnings.filterwarnings("ignore")

log.info(f"🤔 모델 로딩: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=False, padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = (
    AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    .to(DEVICE)
    .eval()
)


# ─────────────────────── 4. 헬퍼 ──────────────────────────────────────────────
def generate_batch(prompts, max_tokens=256, temperature=0.7, top_p=0.95):
    """HF 모델 배치 생성 → list[str]"""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        DEVICE
    )

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    txt = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return [re.split(r"<\|eot_id\|>|<\|end_of_text\|>", t)[0].strip() for t in txt]


# ─────────────────────── 5. 데이터 읽기 & 샘플 ─────────────────────────────────
log.info(f"📄 CSV 로드: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
if "generated" not in df.columns:
    df["generated"] = 0

human_df = df[df["generated"] == 0]
sample_df = (
    human_df.sample(n=NUM_SAMPLES, random_state=42)
    if len(human_df) >= NUM_SAMPLES
    else human_df
).reset_index()
log.info(f"🧑 대상 행: {len(sample_df)} / 전체 {len(df)}")

split_idx = np.array_split(sample_df.index, len(STYLE_LIST))

# ─────────────────────── 6. 스타일별 변환 ─────────────────────────────────────
changed_rows = []

for s_idx, style in enumerate(STYLE_LIST):
    log.info(f"\n🔥 [{style.upper()}] ({s_idx+1}/5)")
    rows_slice = sample_df.loc[split_idx[s_idx]]
    if rows_slice.empty:
        log.info("대상 없음, 스킵")
        continue

    prompts = [
        PROMPTS[style].format(paragraph=r["paragraph_text"])
        for _, r in rows_slice.iterrows()
    ]
    log.info(f"🚀 프롬프트 {len(prompts)}개 → LLM")

    generated = []

    for st in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"{style} gen"):
        generated += generate_batch(prompts[st : st + BATCH_SIZE])

    for (_, orig), gen_text in tqdm(
        zip(rows_slice.iterrows(), generated),
        total=len(rows_slice),
        desc=f"{style} apply",
        leave=False,
    ):
        row_idx = orig["index"]
        df.at[row_idx, "paragraph_text"] = gen_text
        df.at[row_idx, "generated"] = 1
        changed_rows.append(df.loc[[row_idx]])

    log.info("✅ 완료")

# ─────────────────────── 7. 저장 ──────────────────────────────────────────────
log.info("\n💾 CSV 저장")
df.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")
log.info(f"전체 파일 → {OUT_FULL}")

if changed_rows:
    pd.concat(changed_rows).to_csv(OUT_CHANGED, index=False, encoding="utf-8-sig")
    log.info(f"변환 행 {len(changed_rows)}개 → {OUT_CHANGED}")
else:
    log.info("변환된 행 없음")
