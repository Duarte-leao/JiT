#!/bin/bash
#SBATCH --job-name=dinov2-jit
#SBATCH --output=jit_output_%j.txt
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:4          # 4 GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16            # 16 CPUs (4 per GPU)
#SBATCH --time=48:00:00

# 1. Load Anaconda
module load anaconda3

# 2. Activate the NEW environment by full path
eval "$(conda shell.bash hook)"
# This points exactly to the folder you just created
conda activate /ocean/projects/cis250277p/dsilva3/JiT/jit_env

export WANDB_API_KEY="bc7bc5f8d815c5ba398c206a920fbef4582a51a6"

# 3. Navigate to your folder
cd /ocean/projects/cis250277p/dsilva3/JiT

# 4. Set Output Directory
OUT_DIR="/ocean/projects/cis250277p/dsilva3/JiT/jit_experiments/output_dinov2_decoder_and_Lpips_no_feat"
mkdir -p "$OUT_DIR"

# 5. Run Training
# Reduced batch size to 128 to prevent crashing
torchrun --nproc_per_node=4 main_jit.py \
  --model DINOv2-JiT-B/14 \
  --img_size 256 \
  --data_path /ocean/datasets/community/imagenet \
  --batch_size 8 \
  --epochs 500 \
  --output_dir "$OUT_DIR" \
  --wandb \
  --wandb_project dinov2-jit-validation \
  --wandb_run_name dinov2-jit-b14-img224_decoder_and_lpips_no_feat \
  --num_recons 16 \
  --recons_multistep_freq 1 \
  --lambda_feature 0 \
  --lambda_lpips 2.0