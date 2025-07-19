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
- ckpt, data 파일은 위의 Pretrained Checkpoint & dataset 섹션에서 다운로드 받은후 드라이브 구조와 동일하게 위치하면 됩니다.
- data_augmentation 코드를 활용하여 데이터 증강을 진행했으며, 완료된 결과 또한 드라이브에 포함되어 있습니다.
```bash
SW-2025/
├── ckpt/               # weight files
│   ├── full_text/epoch_1.pt
│   ├── llama
│   ├── train_pseudo
│   ├── train_pseudo_custom
│   └── gemma
├── data/               # 기존 데이터 & 증강 데이터 
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv 
│   ├── train_paragraphs.csv
│   ├── train_llama.csv
│   ├── train_gemma.csv
│   └── train_paragraphs_with_pseudo_label.csv 
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

- 라이브러리 버전은 requirements에 저장되어 있습니다.
- 아래 명령어를 순서대로 실행시키면 됩니다.

```bash
conda create -n python=3.10
conda activate sw
python -m pip install requirements.txt
```
## Inference & Ensemble
1. Inference_full_test
2. Inference_Augmetnation(llama)
3. Inference_Augmetnation(gemma)
4. Inference_sudo_labeling
5. Inference_full_test+sudo_labeling 
- 위 네가지에 대한 추론진행후 앙상블을 진행하는 파일을 쉘 스크립트를 통해 작성하였습니다.
- 아래 명령어를 순서대로 실행시키면 됩니다.

```bash
chmod +x ./scripts/inference.sh
./scripts/inference.sh
```

이후 최종 결과물은 submission/final_ensemble.csv 에 저장됩니다.


## Train
- 학습과정은 위에서 추론을 5번 한거와 같이 총 5개의 학습을 진행했습니다.
- 가장 먼저 학습을 진행하기전, 데이터증강을 위한 코드입니다.
- 이 과정을 통하여 나온결과는 드라이브에 동일하게 포함되어 있습니다.
### Data_Augmentation
1. 문단별로 train 분리하기
```bash 
python ./data_augmentation/train_paragraphs.py
```
2. llama증강
```bash 
python ./data_augmentation/llama_augmentation.py
```
3. gemma증강
```bash
python ./data_augmentation/gemma_augmentation.py
```
### train_full_text
train.csv를 sliding window를 활용하여 학습하는 코드입니다.
```bash
python ./train/train_full_text.py
```
### train_Augmentation
증강한 데이터셋(llama, gemma)를 학습하는 코드입니다.
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
sliding window를 진행한 모델로 수도라벨링 데이터셋을 학습하는 코드입니다.
```bash
python ./train/train_sudo_labeling.py \
    --train_csv ./data/train_sudo_label.csv \
    --sampling True \
    --save_dir ./ckpt/train_sudo 
```
sliding window를 학습한 모델에 이어서 수도라벨링 데이터셋을 학습하는 코드입니다.
```bash
python ./train/train_sudo_labeling_custom.py \
    --train_csv ./data/train_sudo_label.csv \
    --save_dir ./ckpt/train_sudo_custom
```