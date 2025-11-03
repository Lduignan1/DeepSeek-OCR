#!/bin/bash
#SBATCH --job-name=deepseek-ocr-sesamath
#SBATCH --output=/home/users/lduignan/experiments/deepseek-ocr/results/logs/%x_%j.out
#SBATCH --error=/home/users/lduignan/experiments/deepseek-ocr/results/logs/%x_%j.err
#SBATCH --partition=gpu40G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

#SBATCH --mail-user=liam.duignan@cea.fr
#SBATCH --mail-type=ALL

# ---- Environment ----
source ~/.bashrc
conda activate deepseek-ocr


# ---- Paths ----
INPUT_DIR="/home/users/lduignan/projects/open-edu-4-data-gen/sources/sesamath/downloads/Cahier_Sésamath_CM1" 
OUTPUT_DIR="/home/users/lduignan/experiments/deepseek-ocr/sesamath/Cahier_Sésamath_CM1"
PYTHON_SCRIPT="/home/users/lduignan/projects/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py"


# ---- Processing Loop ----
echo "Starting OCR batch processing..."


# Loop over all PDF files in this directory
shopt -s nullglob
for pdf_file in "$INPUT_DIR"/*.pdf; do
    echo "  Running OCR on $(basename "$pdf_file")"
    python "$PYTHON_SCRIPT" --input_path "$pdf_file" --output_path "$out_subdir"
done
    


echo "All OCR tasks completed."