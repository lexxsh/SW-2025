import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams
import random
from tqdm import tqdm

# --- 1. 설정값 ---
MODEL_NAME = "rtzr/ko-gemma-2-9b-it"
TENSOR_PARALLEL_SIZE = 1
INPUT_CSV = "./data/train_human_paragraphs.csv"
# 최종 결과 파일들
OUTPUT_CSV_FULL_FINAL = "./data/train_gemma.csv"
OUTPUT_CSV_GENERATED_ONLY = "./data/gemma_human_generated_only.csv"
OUTPUT_CSV_COMPARISON = "./data/gemma_human_comparison_original.csv"
CHUNK_SIZE = 2048


# --- 2. 프롬프트 정의 ---
# (이전과 동일)
PROMPTS = {
    "narrative_essay": """<start_of_turn>user
[지시]
당신은 감수성이 풍부한 작가입니다. 주어진 글을 바탕으로,'수필' 형식으로 문체를 바꿔주세요. 원문의 핵심 정보는 유지해야 합니다. 
오직 변환된 수필 본문만 생성하고, 그 어떤 부연 설명이나 코멘트도 절대 추가하지 마세요.
줄바꿈없이 생성해주세요. 

[원본 글]
{paragraph}

[결과]
<end_of_turn>
<start_of_turn>model
""",
    "logical_essay": """<start_of_turn>user
[지시]
당신은 논리적인 칼럼니스트입니다. 주어진 글의 내용을 바탕으로, '에세이' 형식으로 문체를 바꿔주세요. 
오직 변환된 에세이 본문만 생성하고, 그 어떤 부연 설명이나 코멘트도 절대 추가하지 마세요.
줄바꿈없이 생성해주세요. 

[원본 글]
{paragraph}

[결과]
<end_of_turn>
<start_of_turn>model
""",
    "civic_essay": """<start_of_turn>user
[지시]
당신은 뛰어난 설명가입니다. 주어진 글의 내용을 독자들이 알아듣기 쉽게 설명해주세요. 
오직 변환된 설명글 본문만 생성하고, 그 어떤 부연 설명이나 코멘트도 절대 추가하지 마세요.
줄바꿈없이 생성해주세요. 

[원본 글]
{paragraph}

[결과]
<end_of_turn>
<start_of_turn>model
""",
    "news": """<start_of_turn>user
[지시]
당신은 냉철한 기자입니다. 주어진 글에서 감정적인 표현은 모두 배제하고, 객관적인 사실과 정보만을 사용하여 육하원칙에 따라 간결하고 명료한 '뉴스 기사' 형식으로 문체를 바꿔주세요. 
오직 변환된 뉴스 기사 본문만 생성하고, 그 어떤 부연 설명이나 코멘트도 절대 추가하지 마세요.
줄바꿈없이 생성해주세요. 

[원본 글]
{paragraph}

[결과]
<end_of_turn>
<start_of_turn>model
""",
    "summary": """<start_of_turn>user
[지시]
당신은 뛰어난 요약 전문가입니다. 주어진 글의 핵심 내용을 간결하게 '요약'해주세요. 단, 원래 글의 문체와 톤은 최대한 유지하면서 요약해야 합니다. 스타일을 바꾸지 마세요. 오직 요약된 본문만 생성하고, 그 어떤 부연 설명이나 코멘트도 절대 추가하지 마세요.
줄바꿈없이 생성해주세요. 

[원본 글]
{paragraph}

[결과]
<end_of_turn>
<start_of_turn>model
""",
}

# --- 3. LLM 모델 로딩 ---
print(f"🤔 LLM 모델 ({MODEL_NAME})을 로딩하는 중...")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
)
print("✅ LLM 모델 로딩 완료!")


# --- 4. 데이터 로딩 및 생성 대상 샘플링 ---
print(f"📄 '{INPUT_CSV}' 파일에서 데이터를 읽는 중...")
df = pd.read_csv(INPUT_CSV)
print(f"✅ 총 {len(df)}개의 문단 데이터 로딩 완료.")

### ▼▼▼ 네가 원했던 '2~3번에 한번씩' 샘플링 로직 부활! ▼▼▼ ###
indices_to_process = []
process_in_n_steps = random.randint(2, 3)
# df.index 전체를 대상으로 샘플링 진행 (100개 제한 삭제)
for idx in df.index:
    process_in_n_steps -= 1
    if process_in_n_steps == 0:
        indices_to_process.append(idx)
        process_in_n_steps = random.randint(2, 3)

# 샘플링된 데이터만 따로 모음
df_to_process = df.loc[indices_to_process]
print(
    f"🎯 총 {len(df)}개 중 {len(df_to_process)}개의 문단을 생성 대상으로 선택했습니다."
)
### ▲▲▲ 여기까지 ▲▲▲ ###


# --- 5. 선택된 데이터만 생성 및 실시간 저장 ---
prompt_styles = list(PROMPTS.keys())
sampling_params = SamplingParams(
    temperature=0.7, top_p=0.95, max_tokens=1024, stop=["<end_of_turn>"]
)

# df_to_process (골라낸 애들)을 대상으로 루프 실행
for i in tqdm(
    range(0, len(df_to_process), CHUNK_SIZE), desc="🔥 선택된 문단 생성 진행률"
):
    chunk_df = df_to_process.iloc[i : i + CHUNK_SIZE]

    chunk_prompts = []
    for paragraph in chunk_df["paragraphs"]:
        random_style = random.choice(prompt_styles)
        prompt_template = PROMPTS[random_style]
        chunk_prompts.append(prompt_template.format(paragraph=paragraph))

    outputs = llm.generate(chunk_prompts, sampling_params)
    generated_texts = [output.outputs[0].text.strip() for output in outputs]

    # 생성된 텍스트로 현재 청크 데이터 만들기
    current_chunk_processed = chunk_df.copy()
    current_chunk_processed["paragraphs"] = generated_texts
    current_chunk_processed["generated"] = 1

    # 비교 데이터프레임 만들기
    comparison_df = pd.DataFrame(
        {
            "title": chunk_df["title"],
            "paragraph_index": chunk_df["paragraph_index"],
            "original_paragraph": chunk_df["paragraphs"],
            "generated_paragraph": generated_texts,
        }
    )

    # 파일 저장 모드 결정 (첫 청크는 'w'rite, 나머지는 'a'ppend)
    write_mode = "w" if i == 0 else "a"
    include_header = i == 0

    # 생성된 데이터와 비교 데이터만 실시간으로 저장
    current_chunk_processed.to_csv(
        OUTPUT_CSV_GENERATED_ONLY,
        mode=write_mode,
        header=include_header,
        index=False,
        encoding="utf-8-sig",
    )
    comparison_df.to_csv(
        OUTPUT_CSV_COMPARISON,
        mode=write_mode,
        header=include_header,
        index=False,
        encoding="utf-8-sig",
    )

print("\n✅ 중간 저장 파일 생성 완료!")
print(f"   - 생성된 행만 모음: '{OUTPUT_CSV_GENERATED_ONLY}'")
print(f"   - 원본/생성본 비교: '{OUTPUT_CSV_COMPARISON}'")


# --- 6. 최종 결과물 합치기 ---
print("\n... 이제 원본 데이터와 생성된 데이터를 합쳐 최종 파일을 만듭니다 ...")
try:
    # 중간 저장된 생성본 파일 다시 읽기
    generated_df = pd.read_csv(OUTPUT_CSV_GENERATED_ONLY)
    # 인덱스를 기준으로 업데이트해야 하므로, 원본의 인덱스를 가져와 설정
    generated_df.index = indices_to_process[: len(generated_df)]

    # 원본 데이터프레임에 generated 컬럼 추가하고 0으로 초기화
    df["generated"] = 0
    # update 기능을 사용해 생성된 내용만 덮어쓰기
    df.update(generated_df)

    # 최종 파일 저장
    df.to_csv(OUTPUT_CSV_FULL_FINAL, index=False, encoding="utf-8-sig")
    print(f"\n🎉 모든 작업 완료! 최종 파일 저장 성공! -> '{OUTPUT_CSV_FULL_FINAL}'")

except FileNotFoundError:
    print(
        f"\n⚠️ 오류: 중간 저장 파일 '{OUTPUT_CSV_GENERATED_ONLY}'을 찾을 수 없습니다. 생성이 하나도 안됐을 수 있습니다."
    )
except Exception as e:
    print(f"\n⚠️ 최종 파일 생성 중 오류 발생: {e}")
