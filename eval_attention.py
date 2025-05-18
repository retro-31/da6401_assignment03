# eval_attention_auto.py

import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics   import confusion_matrix

from translit.vocabulary import Vocab
from translit.dataset    import TransliterationDataset
from translit.models     import Encoder, AttentionDecoder, Seq2SeqAttention

# ─── 1) Load global best metadata & checkpoint ──────────────────────────
GLOBAL_JSON = "runs/attention_global_best.json"
BEST_CKPT   = "runs/best_attention_model.pt"
assert os.path.exists(GLOBAL_JSON) and os.path.exists(BEST_CKPT), "Run training first!"

with open(GLOBAL_JSON) as f:
    meta = json.load(f)

best_cfg   = meta["config"]
best_run   = meta["run_name"]
beam_size  = best_cfg["beam_size"]
max_len    = best_cfg["max_len"]

# ─── 2) Init W&B for evaluation ───────────────────────────────────────────
wandb.init(
    project="assignment-03",
    name="attention_test_eval_auto",
    config=best_cfg
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 3) Load vocabularies ─────────────────────────────────────────────────
with open("data/processed/src2idx.pkl","rb") as f: src2idx = pickle.load(f)
with open("data/processed/idx2src.pkl","rb") as f: idx2src = pickle.load(f)
with open("data/processed/tgt2idx.pkl","rb") as f: tgt2idx = pickle.load(f)
with open("data/processed/idx2tgt.pkl","rb") as f: idx2tgt = pickle.load(f)

src_vocab = Vocab(src2idx, idx2src)
tgt_vocab = Vocab(tgt2idx, idx2tgt)

# ─── 4) Rebuild & load the attention model ───────────────────────────────
enc = Encoder(len(src_vocab), best_cfg["embed_dim"], best_cfg["hidden_dim"],
              best_cfg["encoder_layers"], best_cfg["cell_type"], best_cfg["dropout"])
dec = AttentionDecoder(len(tgt_vocab), best_cfg["embed_dim"], best_cfg["hidden_dim"],
                       best_cfg["decoder_layers"], best_cfg["cell_type"], best_cfg["dropout"])
model = Seq2SeqAttention(enc, dec, device).to(device)
model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
model.eval()

# ─── 5) Prepare test split ────────────────────────────────────────────────
test_ds     = TransliterationDataset("test", src_vocab, tgt_vocab, max_len)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# ─── 6) Greedy decode capturing attention weights ────────────────────────
def decode_with_attention(src_tensor):
    with torch.no_grad():
        enc_outs, hidden = model.encoder(src_tensor)
        prev = torch.tensor([[tgt_vocab.char2idx["<sos>"]]], device=device)
        preds, attns = [], []
        for _ in range(max_len):
            out, hidden, attn = model.decoder(prev, hidden, enc_outs)
            tok = out.argmax(dim=-1)                # [1,1]
            preds.append(tok.item())
            attns.append(attn.squeeze(0).squeeze(0).cpu().numpy())
            prev = tok
            if tok.item() == tgt_vocab.char2idx["<eos>"]:
                break
    return preds, np.stack(attns)

# ─── 7) Evaluate test exact‐match & collect records ─────────────────────
total, correct = 0, 0
records = []

for src, tgt in test_loader:
    src, tgt = src.to(device), tgt.to(device)
    inp_str  = "".join(src_vocab.decode(src.squeeze(0).cpu().tolist()))
    true_str = tgt_vocab.decode(tgt.squeeze(0).cpu().tolist()[1:])  # drop <sos>
    pred_ids, _ = decode_with_attention(src)
    pred_str = tgt_vocab.decode(pred_ids[1:])

    records.append((inp_str, true_str, pred_str))
    if pred_str == true_str:
        correct += 1
    total += 1

test_acc = correct / total
wandb.log({"attention_test_accuracy": test_acc})

# ─── 8) Save & upload all predictions ────────────────────────────────────
os.makedirs("predictions_attention", exist_ok=True)
df_all = pd.DataFrame(records, columns=["input","reference","prediction"])
out_tsv = "predictions_attention/all_predictions_attention.tsv"
df_all.to_csv(out_tsv, sep="\t", index=False)
wandb.save(out_tsv)

# ─── 9) Log a random sample table ────────────────────────────────────────
sample = random.sample(records, k=10)
table  = wandb.Table(columns=["input","reference","prediction"], data=sample)
wandb.log({"sample_attention_predictions": table})

# ─── 10) Connectivity plot helper & 3×3 grid ─────────────────────────────
def plot_connectivity(ax, inp, out, attn):
    max_idxs = attn.argmax(axis=1)
    in_len, out_len = len(inp), len(out)
    xs_in, ys_in = np.zeros(in_len), np.arange(in_len)
    xs_out, ys_out = np.ones(out_len), np.arange(out_len)
    ax.scatter(xs_in, ys_in, s=60, color='blue')
    ax.scatter(xs_out, ys_out, s=60, color='green')
    for i,ch in enumerate(inp):
        ax.text(-0.05, ys_in[i], ch, ha='right')
    for j,ch in enumerate(out):
        ax.text(1.05, ys_out[j], ch, ha='left')
    for j in range(out_len):
        i = max_idxs[j]
        ax.plot([0,1], [ys_in[i], ys_out[j]], '-', color='gray', alpha=0.7)
    ax.axis('off')

fig, axes = plt.subplots(3,3, figsize=(9,9))
idxs = random.sample(range(len(records)), 9)
for ax, i in zip(axes.flatten(), idxs):
    inp, true_str, _ = records[i]
    pred_ids, attn = decode_with_attention(
        torch.tensor([src_vocab.encode(list(inp), add_sos_eos=True)],
                     device=device)
    )
    out_str = tgt_vocab.decode(pred_ids[1:])
    plot_connectivity(ax, inp, out_str, attn)
plt.tight_layout()
wandb.log({"connectivity_grid": wandb.Image(fig)})

# ─── 11) Finish ─────────────────────────────────────────────────────────
print(f"Attention Test Accuracy = {test_acc:.4f} ({correct}/{total})")
wandb.finish()
