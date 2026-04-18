# i23-2518-NLP-Assignment2
## CS-4063: Natural Language Processing — Assignment 2
### FAST NUCES | Neural NLP Pipeline | PyTorch from Scratch

---

## Overview

This repository contains the full implementation of a Neural NLP Pipeline on the BBC Urdu corpus, built entirely from scratch in PyTorch. The pipeline covers:

- **Part 1:** TF-IDF, PPMI, and Skip-gram Word2Vec word embeddings
- **Part 2:** BiLSTM sequence labeller for POS tagging and NER (with CRF + Viterbi)
- **Part 3:** Custom Transformer encoder for 5-class topic classification

> No pretrained models, HuggingFace, or Gensim used. `nn.Transformer`, `nn.MultiheadAttention`, and `nn.TransformerEncoder` are not used anywhere.

---

## Repository Structure

```
i23-2518-NLP-Assignment2/
├── i23_2518_Assignment2_DS_A.ipynb   # Main notebook (all cells executed)
├── report.pdf                         # 2–3 page report (Times New Roman 12pt)
├── README.md                          # This file
│
├── embeddings/
│   ├── tfidf_matrix.npy               # TF-IDF term-document matrix (158 x 5871)
│   ├── ppmi_matrix.npy                # PPMI word-word matrix (5871 x 5871)
│   ├── embeddings_w2v.npy             # Averaged Skip-gram embeddings (5871 x 100)
│   └── word2idx.json                  # Vocabulary mapping word → index
│
├── models/
│   ├── bilstm_pos.pt                  # BiLSTM POS tagger (fine-tuned)
│   ├── bilstm_ner.pt                  # BiLSTM NER tagger (CRF, fine-tuned)
│   └── transformer_cls.pt             # Transformer topic classifier
│
└── data/
|   ├── pos_train.conll                # POS annotated training set
|   ├── pos_test.conll                 # POS annotated test set
|   ├── ner_train.conll                # NER annotated training set
|   └── ner_test.conll                 # NER annotated test set
│
└── Extra files(raw, cleaned etc), pngs
```

---

## Requirements

```
python >= 3.10
torch >= 2.0.0
numpy
scikit-learn
matplotlib
```

Install dependencies:

```bash
pip install torch numpy scikit-learn matplotlib
```

---

## Input Files

Place the following files in the **root directory** of the repository before running:

| File | Purpose |
|---|---|
| `cleaned.txt` | Primary training corpus (pipe-tokenised Urdu articles) |
| `raw.txt` | Raw corpus used for ablation baseline (C2) |
| `Metadata.json` | Article metadata with topic labels for classification |

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/i23-2518/i23-2518-NLP-Assignment2.git
cd i23-2518-NLP-Assignment2
```

### 2. Install dependencies

```bash
pip install torch numpy scikit-learn matplotlib
```

### 3. Run the notebook

Open and run all cells top-to-bottom in Jupyter:

```bash
jupyter notebook i23_2518_Assignment2_DS_A.ipynb
```

Or via JupyterLab:

```bash
jupyter lab i23_2518_Assignment2_DS_A.ipynb
```

> **Important:** Run cells sequentially from top to bottom. Each part depends on variables defined in earlier cells.

---

## Part-by-Part Guide

### Part 1 — Word Embeddings

| Cell | What it does |
|---|---|
| Setup & Imports | Loads libraries, sets seeds, defines paths |
| Data Loading | Parses `cleaned.txt` and `raw.txt` into article token lists |
| TF-IDF | Builds vocabulary (top-10K), computes TF-IDF matrix, saves `tfidf_matrix.npy` |
| PPMI | Builds co-occurrence matrix (window=5), applies PPMI, saves `ppmi_matrix.npy` |
| t-SNE | Visualises top-200 tokens; saves `tsne_ppmi.png` |
| Skip-gram Training | Trains Word2Vec (d=100, k=5, K=10, 5 epochs), saves `embeddings_w2v.npy` |
| Evaluation | Nearest neighbours, analogy tests, 4-condition MRR comparison |

**Expected outputs:**
- `embeddings/tfidf_matrix.npy`
- `embeddings/ppmi_matrix.npy`
- `embeddings/embeddings_w2v.npy`
- `embeddings/word2idx.json`
- `tsne_ppmi.png`
- `w2v_loss_curve.png`

---

### Part 2 — BiLSTM Sequence Labeling

| Cell | What it does |
|---|---|
| Sentence Selection | Selects 500 sentences stratified by topic |
| POS Tagger | Rule-based tagger using lexicon + suffix rules |
| NER Annotator | BIO annotation using seed gazetteers |
| Data Split | 70/15/15 train/val/test split; writes CoNLL files |
| BiLSTM Model | 2-layer BiLSTM with dropout=0.5; CRF for NER |
| POS Training | Trains frozen and fine-tuned modes; early stopping on val-F1 |
| NER Training | Trains with CRF + Viterbi; also trains without CRF for comparison |
| Evaluation | Accuracy, Macro-F1, confusion matrix, error analysis, ablations A1–A4 |

**Expected outputs:**
- `models/bilstm_pos.pt`
- `models/bilstm_ner.pt`
- `data/pos_train.conll`, `data/pos_test.conll`
- `data/ner_train.conll`, `data/ner_test.conll`
- `bilstm_pos_loss.png`
- `bilstm_ner_loss.png`
- `pos_confusion.png`

---

### Part 3 — Transformer Encoder

| Cell | What it does |
|---|---|
| Dataset Prep | Assigns 4-5 topic categories; pads/truncates to 256 tokens; stratified split |
| Architecture | Sinusoidal PE, scaled dot-product attention, multi-head attention (h=4), Pre-LN encoder ×4, CLS token |
| Training | AdamW (lr=5e-4, wd=0.01), cosine LR with 50 warmup steps, 20 epochs |
| Evaluation | Test accuracy, Macro-F1, 5×5 confusion matrix, attention heatmaps |
| Comparison | BiLSTM vs Transformer on all 5 assignment questions |

**Expected outputs:**
- `models/transformer_cls.pt`
- `transformer_curves.png`
- `transformer_confusion.png`
- `attention_heatmaps.png`

---

## Key Results Summary

### Part 1

| Condition | Description | MRR |
|---|---|---|
| C1 | PPMI baseline | 0.032 |
| C2 | Skip-gram on raw.txt | 0.000 |
| C3 | Skip-gram on cleaned.txt | 0.028 |
| C4 | Skip-gram, d=200 | 0.018 |

### Part 2

| Task | Mode | Accuracy | Macro-F1 |
|---|---|---|---|
| POS | Frozen embeddings | 0.8432 | 0.4278 |
| POS | Fine-tuned embeddings | **0.9462** | **0.8135** |
| NER | With CRF | — | 0.49 |
| NER | Without CRF | — | 0.49 |

### Part 3

| Model | Test Accuracy | Macro-F1 |
|---|---|---|
| BiLSTM Classifier | 75.0% | ~0.45 |
| Transformer Encoder | **87.5%** | 0.45 |

---

## Notes

- Training time on CPU: Part 1 ~5–10 min, Part 2 ~10–15 min, Part 3 ~5 min
- GPU (CUDA) will be used automatically if available
- Random seed is fixed at 42 for full reproducibility
- Urdu text requires a Unicode-compatible terminal/font for display
- All notebook cells must be run in order; re-running individual cells out of sequence may cause variable conflicts

---

## Author

**Student ID:** i23-2518  
**Section:** DS-A  
**Course:** CS-4063 Natural Language Processing  
**Semester:** Spring 2026  
**University:** FAST NUCES
