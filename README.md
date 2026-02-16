# Affective Memory Networks (AMN)

[![arXiv](https://img.shields.io/badge/arXiv-2602.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2602.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation of "Affective Memory Networks: Emotion-Aware Conversational Memory Retrieval"**

> AMN achieves 80% memory reference rate (68% improvement over baseline) by organizing AI memory around emotional salience, mirroring human episodic memory principles.

## 📄 Paper

- **arXiv preprint**: [https://arxiv.org/abs/2602.XXXXX](https://arxiv.org/abs/2602.XXXXX) *(add after submission)*
- **PDF**: [paper/phase1_final.pdf](paper/phase1_final.pdf)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sakshyambanjade/Affective-Memory-Networks.git
cd Affective-Memory-Networks

# Install dependencies
pip install -r requirements.txt

# Set up API key (if using Anthropic API)
export ANTHROPIC_API_KEY="your-key-here"
```

### Run Experiments

**Reproduce paper results (Table 1 & Figures 2-3):**
```bash
python run_experiments.py --output results/reproduction.json
```

This will:
1. Load 15 test conversations from `amn_data_package/`
2. Run AMN + 3 baselines (Baseline, Semantic RAG, Recency)
3. Compute memory reference rates and coherence scores
4. Output results to `results/reproduction.json`

**Expected output:**
```
AMN Memory Rate: 80.0%
Baseline Memory Rate: 47.7%
AMN Coherence: 0.889
Baseline Coherence: 0.868
```

### Interactive Demo

```bash
python demo.py  # Coming soon
```

## 📊 Results

| System | Memory Reference Rate | Coherence (BERTScore) |
|--------|----------------------|----------------------|
| **AMN (Ours)** | **80.0%** | **0.889** |
| Baseline | 47.7% | 0.868 |
| Semantic RAG | 33.0% | 0.886 |
| Recency Only | 15.0% | 0.880 |

## 📁 Repository Structure

```
Affective-Memory-Networks/
├── src/
│   ├── agent/          # AMN implementation + baselines
│   ├── emotion/        # VAD emotion detection
│   ├── memory/         # Memory storage & consolidation
│   └── retrieval/      # Multi-factor retrieval engine
├── amn_data_package/
│   ├── conversations/  # 105 emotional conversations
│   ├── lexicons/       # VAD lexicon (209 words)
│   └── test_cases/     # Validation test cases
├── experiments/
│   ├── exp1_full.py    # Main experiment (Table 1)
│   └── metrics.py      # Evaluation metrics
├── results/            # Experimental results (JSON + figures)
├── paper/              # LaTeX source for paper
└── run_experiments.py  # One-command reproduction script
```

## 🔧 Implementation Details

### System Components

1. **Emotion Detection** (`src/emotion/analyzer.py`)
	- VAD dimensional scoring (Valence, Arousal, Dominance)
	- 209-word emotion lexicon from Warriner et al. (2013)
	- Returns normalized scores in [-1, 1] range

2. **Memory System** (`src/memory/core.py`)
	- Three-layer architecture (Working, Episodic, Semantic)
	- Automatic consolidation based on emotional intensity
	- Stores: text, VAD scores, timestamps, goal relevance

3. **Retrieval Engine** (`src/retrieval/engine.py`)
	- Multi-factor scoring: 25% topic + 30% emotion + 20% goal + 15% intensity + 10% recency
	- Sentence-BERT embeddings for semantic similarity
	- Top-k retrieval (k=3 by default)

4. **Response Generation**
	- Uses Claude 3.5 Sonnet via Anthropic API
	- Retrieved memories included in context window
	- Temperature: 0.7, Max tokens: 2048

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{banjade2026affective,
  title={Affective Memory Networks: Emotion-Aware Conversational Memory Retrieval},
  author={Banjade, Sakshyam},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026}
}
```

## 📖 Data

The `amn_data_package/` contains:
- **105 conversations** across 15 emotions (grief, joy, anxiety, anger, etc.)
- **VAD lexicon** with 209 emotion words
- **Test cases** for validation

Data generation details in `amn_data_package/README.md`

## 🛠️ Dependencies

- Python 3.8+
- anthropic >= 0.25.0 (for Claude API)
- sentence-transformers >= 2.2.0 (for embeddings)
- bert-score >= 0.3.13 (for evaluation)
- numpy, pandas, nltk, transformers, torch, scikit-learn

See `requirements.txt` for full list.

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/

# Verify data package
python amn_data_package/scripts/verify_package.py
```

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

Data (`amn_data_package/`) is licensed for research use only.

## 👤 Author

**Sakshyam Banjade**  
Independent Researcher  
Email: sakshyambanjade@example.com  
GitHub: [@sakshyambanjade](https://github.com/sakshyambanjade)

## 🙏 Acknowledgments

- Valence-Arousal-Dominance norms from Warriner et al. (2013)
- Sentence-BERT embeddings from Reimers & Gurevych (2019)
- BERTScore evaluation from Zhang et al. (2020)

## 📮 Contact

For questions about the paper or code, please:
1. Open a GitHub issue
2. Email: sakshyambanjade@example.com

---

**Last updated**: February 2026
