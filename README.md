# 2025 DACON SW중심 경진대회
## Pretrained Checkpoint & dataset
## OpenSource Model
데이터 증강 & 학습시 사용한 모델은 다음과 같습니다.
```bash
- Data Augmentation
https://huggingface.co/SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx
https://huggingface.co/rtzr/ko-gemma-2-9b-it

- Train
https://huggingface.co/team-lucid/deberta-v3-base-korean
```
## Folder Structure
ckpt, data 파일은 위의 Pretrained Checkpoint & dataset 섹션에서 다운로드 받은후 드라이브 구조와 동일하게 위치하면 됩니다.
data_augmentation 코드를 활용하여 데이터 증강을 진행했으며, 완료된 결과 또한 드라이브에 포함되어 있습니다.
```bash
SW-2025/
├── ckpt/               # weight files
│   ├── 1.pt
│   ├── 2.pt
│   └── 3.pt
├── data/               # 기존 데이터 & 증강 데이터 
│   ├── train.csv
│   ├── train_llama.csv
│   ├── train_gemma.csv
│   ├── test.csv
│   └── sample_submission.csv
├── data_augmentation/  # 데이터 증강 코드
│   └── augmentation.py
├── emsemble/           # 앙상블 코드
│   ├── ensemble_2.py
│   └── ensemble.py
├── inference/          # 추론 코드
│   ├── inference_full_text.py
│   └── inference_main.py
├── train/              # 학습 코드
│   ├── train_full_text.py
│   ├── train_main.py
│   └── train_sudo_labeling.py
└── README.md
```
## Conda Environmet

라이브러리 버전은 environment.yml에 저장되어 있습니다.
아래 코드를 순서대로 실행시키면 됩니다.

```bash
conda env create -f environment.yml
conda activate sw
```
## Inference

### Inference_full_text
```bash
python ./inference/inference_full_text.py \
    --model_path ./ckpt/full_text_sliding_window/epoch_1.pt
```
### Inference_Augmetnation
```bash
python ./inference/inference_main.py \
    --model_path ./ckpt/llama \
    --output_csv submission_llama.csv
```
```bash
python ./inference/inference_main.py \
    --model_path ./ckpt/gemma \
    --output_csv submission_gemma.csv
```
### Inference_sudo_labeling
```bash
python ./inference/inference_main.py \
    --model_path ./ckpt/train_sudo \
    --output_csv submission_train_sudo.csv
```
```bash
준비중,,,
```
### Emsemble
```bash
python ./emsemble/ensemble.py --csv_files 1.csv 2.csv --output_csv ensembled.csv
```

## Train
### Data_Augmentation
```bash
python ./data_augmentation/augmentation.py
```
### train_full_text
```bash
python ./train/train_full_text.py
```
### train_Augmentation
```bash
python ./train/train_main.py \
    --train_csv ./data/train_llama.csv \
    --save_dir ./ckpt/llama \
    --sampling 18000 39000
```
```bash
python ./train/train_main.py \
    --train_csv ./data/train_gemma.csv \
    --save_dir ./ckpt/gemma
```
### train_sudo_labeling
```bash
python ./train/train_sudo_labeling.py \
    --train_csv ./data/train_sudo_label.csv \
    --sampling True
```
```bash
준비중,,,
```