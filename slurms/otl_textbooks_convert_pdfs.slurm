#!/bin/bash
#SBATCH --job-name=deepseek-ocr-otl
#SBATCH --output=/home/users/lduignan/experiments/deepseek-ocr/results/logs/%x_%j.out
#SBATCH --error=/home/users/lduignan/experiments/deepseek-ocr/results/logs/%x_%j.err
#SBATCH --partition=gpuh200p,gpuh100p,gpu80G,gpu40G
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
INPUT_DIR="/home/users/lduignan/data/OTL_pdfs" 
OUTPUT_DIR="/home/users/lduignan/experiments/deepseek-ocr/otl_textbooks"
PYTHON_SCRIPT="/home/users/lduignan/projects/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py"


# ---- Processing Loop ----
echo "Starting OCR batch processing..."

# ensure globs that match nothing expand to nothing
shopt -s nullglob


for entry in "$INPUT_DIR"/*; do
    if [ -d "$entry" ]; then
        subdir="$entry"
        subdir_name=$(basename "$subdir")
        out_subdir="$OUTPUT_DIR/$subdir_name"
        mkdir -p "$out_subdir"

        echo "Processing directory: $subdir_name"

        # Loop over all PDF files in this subdirectory
        for pdf_file in "$subdir"/*.pdf; do
            echo "  Running OCR on $(basename "$pdf_file")"
            python "$PYTHON_SCRIPT" --input-path "$pdf_file" --output-path "$out_subdir"
        done

    elif [ -f "$entry" ] && [[ "${entry,,}" == *.pdf ]]; then
        # Handle PDFs directly inside INPUT_DIR
        mkdir -p "$OUTPUT_DIR"
        echo "Processing PDF in root INPUT_DIR: $(basename "$entry")"
        python "$PYTHON_SCRIPT" --input-path "$entry" --output-path "$OUTPUT_DIR"
    fi
done

echo "All OCR tasks completed."