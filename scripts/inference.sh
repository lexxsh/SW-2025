set -euo pipefail

# 1) Pseudo labeling
# echo ">>> Running pseudo labeling..."
# python ./pseudo_labeling/pseudo_labeling.py

# 2) Full Text Sliding Window Inference
 echo ">>> Running full-text sliding window inference..."
 python ./inference/inference_custom.py \
     --model_path ./ckpt/full_text/epoch_1.pt \
     --torch True \
     --output_csv ./submission/submission_full_text.csv
# 3) Inference_Augmentation
 echo ">>> Running inference augmentation..."
 python ./inference/inference.py \
     --model_path ./ckpt/llama/checkpoint_2 \
     --output_csv ./submission/submission_llama.csv

 python ./inference/inference.py \
     --model_path ./ckpt/gemma/checkpoint_2 \
     --output_csv ./submission/submission_gemma.csv

# 4) Inference_pseudo_labeling
echo ">>> Running inference for pseudo labeling..."
python ./inference/inference_custom.py \
    --model_path ./ckpt/train_pseudo/checkpoint_1 \
    --output_csv ./submission/submission_train_pseudo.csv

# 5) Self-Training Inference
echo ">>> Running self-training inference..."
python ./inference/inference_custom.py \
    --model_path ./ckpt/self_training/checkpoint_1 \
    --output_csv ./submission/submission_self_training.csv

echo ">>> All inference jobs completed!"

# 6) Ensemble
echo ">>> Ensembling start!!!"
python ./ensemble/ensemble_2.py \
    --csv_files ./submission/submission_full_text.csv \
    ./submission/submission_llama.csv \
    ./submission/submission_gemma.csv \
    ./submission/submission_train_pseudo.csv \
    ./submission/submission_self_training.csv \
    --output_csv ./submission/final_ensemble.csv

echo ">>> All finish!!!"