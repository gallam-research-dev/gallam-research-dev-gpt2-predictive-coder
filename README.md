# How GPT-2 Learns to Be a Predictive Coder
### Evidence from Directional Activation Patching

**Jong-O Yun** | Independent Researcher, South Korea | gallam.research@gmail.com

> Paper: [arXiv link] (pending)
> Related: [Paper 1 Repository](https://github.com/gallam-research-dev/pc-transformer-interpretability)

---

## Overview

This repository contains code for the second paper in our series investigating
predictive coding (PC) tendencies in transformer-based language models.

**Key finding**: GPT-2 family models exhibit statistically significant backward
directional asymmetry in the convergence zone (p < 0.001, n=90), while a
randomly initialized control shows near-zero asymmetry (Cohen's d = 1.00).
This constitutes the first experimental evidence that partial PC structure
emerges through training, not architecture.

---

## Repository Structure

```
├── run_experiments.py    # Main experiment (directional patching)
├── analyze_results.py    # Statistical analysis (t-test + bootstrap CI)
├── merge_figures.py      # Combine individual model plots for paper
└── README.md
```

---

## How to Run

### Step 1: Run directional patching for each model

Open `run_experiments.py` in Google Colab. Change `RUN_MODEL` and run once
per model. Reset runtime between models.

```python
# Run 7 times total (reset runtime each time):
RUN_MODEL = "gpt2"                      # Run 1
RUN_MODEL = "gpt2-medium"               # Run 2
RUN_MODEL = "gpt2-large"                # Run 3
RUN_MODEL = "EleutherAI/gpt-neo-125M"   # Run 4
RUN_MODEL = "EleutherAI/pythia-160m"    # Run 5
RUN_MODEL = "EleutherAI/pythia-410m"    # Run 6
RANDOM_INIT = True                       # Run 7 (control)
```

All results accumulate in `directional_results.json`.

### Step 2: Statistical analysis

```python
# Run after all 7 experiments are complete
# Place directional_results.json in the same directory
python analyze_results.py
```

### Step 3: Merge figures for paper

```python
python merge_figures.py
```

---

## Requirements

```
transformer_lens
torch
numpy
scipy
matplotlib
pillow
```

All experiments were run on Google Colab T4 GPU.

---

## Results Summary

| Model | Mean Asymmetry | 95% CI | p | Sig |
|---|---|---|---|---|
| GPT-2 Small (117M) | +0.969 | [+0.697, +1.252] | <0.001 | *** |
| GPT-2 Med (345M) | +0.290 | [+0.167, +0.421] | <0.001 | *** |
| GPT-2 Large (774M) | +0.194 | [+0.090, +0.300] | <0.001 | *** |
| GPT-Neo (125M) | -0.345 | [-0.744, +0.067] | 0.102 | ns |
| Pythia (160M) | -0.105 | [-0.337, +0.123] | 0.380 | ns |
| Pythia (410M) | -0.128 | [-0.278, +0.021] | 0.100 | ns |
| GPT-2 Random Init | -0.008 | [-0.080, +0.063] | 0.827 | ns |

---

## Citation

```bibtex
@article{yun2026directional,
  title={How {GPT}-2 Learns to Be a Predictive Coder:
         Evidence from Directional Activation Patching},
  author={Yun, Jong-O},
  journal={arXiv preprint},
  year={2026}
}
```
