# eval_test.py
import os
import json
import pickle
import random

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import wandb
from torch.utils.data import DataLoader

from translit.vocabulary import Vocab
from translit.dataset    import TransliterationDataset
from translit.models     import Encoder, Decoder, Seq2Seq

# ─── 1) Load meta & checkpoint ───────────────────────────────────────────
META_PATH   = "runs/global_best.json"
MODEL_PATH  = "runs/best_model.pt"
assert os.path.exists(META_PATH) and os.path.exists(MODEL_PATH), "Run sweep first!"

with open(META_PATH) as f:
    meta = json.load(f)

# Parse hyperparams out of run_name
parts = meta["run_name"].split("_")
cfg = {
    "cell_type":      parts[0],
    "encoder_layers": int(parts[1].replace("enc","")),
    "decoder_layers": int(parts[2].replace("dec","")),
    "hidden_dim":     int(parts[3].replace("hd","")),
    "embed_dim":      int(parts[4].replace("ed","")),
    "dropout":        int(parts[5].replace("dr",""))/100,
    "beam_size":      int(parts[6].replace("bs","")),
    "max_len":        32,
}

# ─── 2) Init W&B ────────────────────────────────────────────────────────
wandb.init(
    project="assignment-03",
    name="test_evaluation",
    config=cfg
)

# ─── 3) Load vocabs ─────────────────────────────────────────────────────
with open("data/processed/src2idx.pkl","rb") as f: src2idx = pickle.load(f)
with open("data/processed/idx2src.pkl","rb") as f: idx2src = pickle.load(f)
with open("data/processed/tgt2idx.pkl","rb") as f: tgt2idx = pickle.load(f)
with open("data/processed/idx2tgt.pkl","rb") as f: idx2tgt = pickle.load(f)

src_vocab = Vocab(src2idx, idx2src)
tgt_vocab = Vocab(tgt2idx, idx2tgt)

# ─── 4) Rebuild & load model ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = Encoder(len(src_vocab), cfg["embed_dim"], cfg["hidden_dim"],
              cfg["encoder_layers"], cfg["cell_type"], cfg["dropout"])
dec = Decoder(len(tgt_vocab), cfg["embed_dim"], cfg["hidden_dim"],
              cfg["decoder_layers"], cfg["cell_type"], cfg["dropout"])
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ─── 5) Prepare test loader ─────────────────────────────────────────────
test_ds = TransliterationDataset("test", src_vocab, tgt_vocab, cfg["max_len"])
test_loader = DataLoader(test_ds, batch_size=1)

# ─── 6) Run exact-match & gather all char-level pairs ────────────────────
total, correct = 0, 0
true_chars, pred_chars = [], []
records = []

with torch.no_grad():
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        pred_seq = model.beam_search(
            src.squeeze(0),
            sos_idx = tgt_vocab.char2idx["<sos>"],
            eos_idx = tgt_vocab.char2idx["<eos>"],
            max_len = cfg["max_len"],
            beam_size=cfg["beam_size"]
        )
        pred = tgt_vocab.decode(pred_seq[1:])         # drop <sos>
        truth = tgt_vocab.decode(tgt.squeeze(0).tolist()[1:])

        # record for grid
        input_str = "".join(src_vocab.decode(src.squeeze(0).tolist()))
        records.append((input_str, truth, pred))

        # exact match
        if pred == truth:
            correct += 1
        total += 1

        # char-level confusion
        # align both strings (pad shorter with <pad>)
        maxlen = max(len(truth), len(pred))
        truth += tgt_vocab.idx2char[tgt_vocab.char2idx["<pad>"]] * (maxlen - len(truth))
        pred  += tgt_vocab.idx2char[tgt_vocab.char2idx["<pad>"]] * (maxlen - len(pred))
        for t_char, p_char in zip(truth, pred):
            true_chars.append(tgt_vocab.char2idx.get(t_char, 1))
            pred_chars.append(tgt_vocab.char2idx.get(p_char, 1))

test_acc = correct / total
wandb.log({"test_accuracy": test_acc})

# ─── Dump all predictions to a TSV ────────────────────────────────────────
OUT_DIR = "predictions_vanilla"
os.makedirs(OUT_DIR, exist_ok=True)

# records is a list of tuples (input_str, truth, pred)
df = pd.DataFrame(records, columns=["input","reference","prediction"])
out_path = os.path.join(OUT_DIR, "all_predictions.tsv")
df.to_csv(out_path, sep="\t", index=False)
print(f"✅ Saved all predictions to {out_path}")

# If running under W&B, tell it to save that file
wandb.save(out_path)

# ─── 7) Log a random sample grid (10 rows) ──────────────────────────────
sample = random.sample(records, k=10)
table = wandb.Table(columns=["input","reference","prediction"], data=sample)
wandb.log({"sample_predictions": table})

# ─── 8) Compute & log confusion matrix ─────────────────────────────────
labels = list(tgt_vocab.idx2char)  # all characters including PAD, UNK, SOS, EOS
cm = confusion_matrix(true_chars, pred_chars, labels=range(len(labels)))

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(cm, interpolation='nearest')
ax.set_title("Character Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
plt.tight_layout()

wandb.log({"confusion_matrix": wandb.Image(fig)})

# ─── 9) Finalize ────────────────────────────────────────────────────────
print(f">>> Test Accuracy = {test_acc:.4f} ({correct}/{total})")
wandb.finish()
