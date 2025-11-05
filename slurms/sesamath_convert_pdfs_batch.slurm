#!/bin/bash
#SBATCH --job-name=deepseek-ocr-sesamath-test-modif
#SBATCH --output=/home/users/lduignan/experiments/deepseek-ocr/results/logs/%x_%j.out
#SBATCH --error=/home/users/lduignan/experiments/deepseek-ocr/results/logs/%x_%j.err
#SBATCH --partition=gpuh200p,gpuh100p,gpu80G,gpu40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

#SBATCH --mail-user=liam.duignan@cea.fr
#SBATCH --mail-type=ALL

# ---- Environment ----
source ~/.bashrc
conda activate deepseek-ocr


# ---- Paths ----
INPUT_DIR="/home/users/lduignan/projects/open-edu-4-data-gen/sources/sesamath/downloads" 
OUTPUT_DIR="/home/users/lduignan/experiments/deepseek-ocr/sesamath"
PYTHON_SCRIPT="/home/users/lduignan/projects/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf_batch.py"


# ---- Processing Loop ----
echo "Starting OCR batch processing..."


python "$PYTHON_SCRIPT" --input-path "$INPUT_DIR" --output-path "$OUTPUT_DIR"

    
echo "All OCR tasks completed."