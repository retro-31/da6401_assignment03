# Transliteration Seq2Seq with Attention

This repository contains code for building, tuning, and evaluating a character-level seq2seq transliteration system (Latin → Devanagari) using PyTorch, with both vanilla and attention-based models. All experiments are tracked via Weights & Biases.

---

## 📂 Directory Structure

```
├── data/                       # raw & processed datasets
│   ├── dakshina/               # downloaded Dakshina datasets
│   ├── processed/              # pickled vocabs and splits
├── translit/                   # package source code
│   ├── vocabulary.py           # Vocab class
│   ├── dataset.py              # TransliterationDataset
│   ├── models.py               # Encoder, Decoder, Seq2Seq, Attention
├── train.py                    # train vanilla seq2seq + global best
├── sweep.py                    # wandb sweep configuration
├── eval_test.py                # evaluate vanilla model
├── train_attention.py         # train attention model + global best
├── eval_attention.py          # evaluate attention model + connectivity
├── README.md                   # this file
├── requirements.txt            # Python dependencies
└── runs/                       # checkpoints & metadata
```

---

## 🎯 Setup & Installation

1. **Clone the repo**:

   ```bash
   git clone <your-repo-url>
   cd da6401_assignment03
   ```

2. **Create & activate conda env**:

   ```bash
   conda create -n dl_seq2seq python=3.10 -y
   conda activate dl_seq2seq
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**:

   ```bash
   python data/prepare_data.py
   ```

   This will download the Google Dakshina dataset and write `data/processed/` files.

---

## 🔧 Training the Vanilla Seq2Seq

Run the basic seq2seq training (no attention):

```bash
python train.py
```

* Saves per-run best checkpoints under `runs/<run_name>_tmp.pt`.
* Maintains a global best in `runs/global_best.json` & `runs/best_model.pt`.
* All metrics (train\_loss, dev\_accuracy) are logged to W\&B.

### Hyperparameter Sweeps

Use `sweep.py` to launch a W\&B hyperparameter sweep:

```bash
wandb sweep sweep.py
wandb agent <SWEEP_ID>  # to start runs
```

Logs accuracy vs. creation time, parallel coordinates, and parameter importance automatically.

---

## 📈 Evaluating the Vanilla Model

Once `runs/best_model.pt` is ready, run:

```bash
python eval_test.py
```

This script:

* Loads `runs/global_best.json` & `runs/best_model.pt`.
* Computes exact-match accuracy on the test split.
* Saves `predictions_vanilla/all_predictions_vanilla.tsv`.
* Logs a sample table and confusion matrix to W\&B.

---

## 🔦 Training the Attention Model

To train the attention-based variant:

```bash
python train_attention.py
```

It follows the same global-best pattern, writing:

* `runs/attention_global_best.json` (best accuracy, run\_name, config)
* `runs/best_attention_model.pt`

All train/dev metrics are also on W\&B.

---

## 🔍 Evaluating the Attention Model

Once `runs/best_attention_model.pt` exists, run:

```bash
python eval_attention.py
```

This script:

* Loads the best attention config & checkpoint.
* Computes beam-search exact-match accuracy on test.
* Saves `predictions_attention/all_predictions_attention.tsv`.
* Logs a 3×3 **connectivity plot** and a 3×3 grid of **attention heatmaps** to W\&B.

---

## 📋 Requirements

* Python 3.10
* PyTorch
* pandas, numpy, sklearn, matplotlib
* wandb

Install via `pip install -r requirements.txt`.

---

## 📝 Notes

* Ensure you activate your conda env before running.
* Check `runs/` for JSON & model files between scripts.
* Use W\&B web UI to compare runs, view heatmaps, and parameter importance.

---
