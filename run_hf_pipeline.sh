#!/bin/bash
set -e

# Configuration
GLB_DIR="data/objaverse"
OUTPUT_DIR="output_hf"
REPO_NAME="grasp-dataset-curated"
VENV_PYTHON="./venv/bin/python"

echo "----------------------------------------------------"
echo "🖐️  Grasp Dataset Scaling Pipeline"
echo "----------------------------------------------------"

# 1. Download Objects
echo "1/4 📥 Downloading high-quality objects from Objaverse-LVIS..."
$VENV_PYTHON scripts/download_objaverse.py

# 2. Generate Dataset
echo "2/4 ⚙️  Generating synthetic grasp data (renders, contacts, visibility)..."
# We exclude existing data/glb to focus on the new ones
$VENV_PYTHON generate_dataset.py --glb_dir $GLB_DIR --output_dir $OUTPUT_DIR

# 3. Export to CSV and LaTeX
echo "3/4 📊 Finalizing exports..."
$VENV_PYTHON export_to_csv.py --index $OUTPUT_DIR/dataset_index.json --output $OUTPUT_DIR/grasp_data.csv
$VENV_PYTHON generate_latex_report.py --index $OUTPUT_DIR/dataset_index.json --csv $OUTPUT_DIR/grasp_data.csv --output $OUTPUT_DIR/rapport_dataset.tex

# 4. Upload to Hugging Face
echo "4/4 🚀 Uploading to Hugging Face..."
$VENV_PYTHON scripts/upload_to_hf.py --folder $OUTPUT_DIR --repo $REPO_NAME

echo "----------------------------------------------------"
echo "✨ All done! Your dataset is now live on Hugging Face."
echo "----------------------------------------------------"
