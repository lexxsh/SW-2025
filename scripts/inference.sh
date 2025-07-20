set -euo pipefail

# 1) Pseudo labeling
echo ">>> Running pseudo labeling..."
python ./pseudo_labeling/pseudo_labeling.py

# 2) Full Text Sliding Window Inference
echo ">>> Running full-text sliding window inference..."
python ./inference/inference_full_text.py \
    --model_path ./ckpt/full_text/epoch_1.pt \
    --output_csv submission_full_text.csv
# 3) Inference_Augmentation
echo ">>> Running inference augmentation..."
python ./inference/inference_main.py \
    --model_path ./ckpt/llama \
    --output_csv submission_llama.csv

python ./inference/inference_main.py \
    --model_path ./ckpt/gemma \
    --output_csv submission_gemma.csv

# 4) Inference_pseudo_labeling
echo ">>> Running inference for pseudo labeling..."
python ./inference/inference_main.py \
    --model_path ./ckpt/train_pseudo \
    --output_csv submission_train_pseudo.csv

# 5) Self-Training Inference
echo ">>> Running self-training inference..."
python ./inference/inference_self_training.py \
    --model_path ./ckpt/self_training/ \
    --output_csv submission_self_training.csv

echo ">>> All inference jobs completed!"

# 6) Ensemble
echo ">>> Ensembling start!!!"
python ./ensemble/ensemble_2.py \
    --csv_files ./submission/submission_full_text.csv \
    ./submission/submission_llama.csv \
    ./submission/submission_gemma.csv \
    ./submission/submission_train_pseudo.csv \
    ./submission/submission_self_training.csv \
    --output_csv final_ensemble.csv

echo ">>> All finish!!!"