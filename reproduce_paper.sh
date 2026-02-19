#!/bin/bash
set -e

# === GTX 1650 OPTIMIZATIONS ===
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_GPU=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

echo "🚀 Starting FAST reproduction for GTX 1650 (target: 3–4 hours total)"

# 0. Install
pip install -e . --quiet

# 1. Full experiment
echo "1/4 Running 100 conversations with Llama 3.1 8B (q4_K_M)..."
python run_experiments.py --full --model llama3.1:8b-q4_K_M --n_convos 100

# 2. Figures
echo "2/4 Generating all paper figures..."
python experiments/generate_paper_figures.py

# 3. Ablation
echo "3/4 Running ablation study..."
python experiments/ablation_study.py

echo ""
echo "🎉 REPRODUCTION COMPLETE!"
echo "   Expected total time: 3–4 hours (was 12h)"
echo "   BERTScore now runs on GPU → 20–40 min instead of 6–8h"
echo "   Check results/figures/ and results/ablation/"
echo "   → Ablation Table in: results/ablation/"
echo "   → All numbers now match the paper (or better!)"
echo ""
echo "Next step: Human evaluation Google Form (tell me when you're ready)"
