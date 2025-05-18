# train_attention.py

import os
import json
import wandb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from translit.vocabulary import Vocab
from translit.dataset    import TransliterationDataset
from translit.models     import Encoder, AttentionDecoder, Seq2SeqAttention

def main():
    # ── 1) defaults & wandb init ───────────────────────────────────────────
    defaults = {
        "embed_dim":      256,
        "hidden_dim":     256,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "cell_type":      "GRU",
        "dropout":        0.2,
        "beam_size":      5,
        "max_len":        32,
        "batch_size":     64,
        "lr":             1e-3,
        "epochs":         10
    }
    wandb.init(project="assignment-03-attention", config=defaults)
    cfg = wandb.config

    # ── 2) derive run‐name from hyperparams ────────────────────────────────
    run_name = (
        f"{cfg.cell_type}"
        f"_atten_enc{cfg.encoder_layers}"
        f"_atten_dec{cfg.decoder_layers}"
        f"_hd{cfg.hidden_dim}"
        f"_ed{cfg.embed_dim}"
        f"_dr{int(cfg.dropout*100)}"
        f"_bs{cfg.beam_size}"
    )
    wandb.run.name = run_name  # type: ignore
    wandb.run.save()           # type: ignore

    # ── 3) prepare folders ────────────────────────────────────────────────
    os.makedirs("runs", exist_ok=True)
    tmp_ckpt     = f"runs/{run_name}_tmp.pt"
    global_json  = "runs/attention_global_best.json"
    global_model = "runs/best_attention_model.pt"

    # ── 4) load device, vocabs, data ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("data/processed/src2idx.pkl","rb") as f: src2idx = pickle.load(f)
    with open("data/processed/idx2src.pkl","rb") as f: idx2src = pickle.load(f)
    with open("data/processed/tgt2idx.pkl","rb") as f: tgt2idx = pickle.load(f)
    with open("data/processed/idx2tgt.pkl","rb") as f: idx2tgt = pickle.load(f)

    src_vocab = Vocab(src2idx, idx2src)
    tgt_vocab = Vocab(tgt2idx, idx2tgt)

    train_ds     = TransliterationDataset("train", src_vocab, tgt_vocab, cfg.max_len)
    dev_ds       = TransliterationDataset("dev",   src_vocab, tgt_vocab, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=1)

    # ── 5) build attention model, loss, optimizer ─────────────────────────
    enc = Encoder(len(src_vocab), cfg.embed_dim, cfg.hidden_dim,
                  cfg.encoder_layers, cfg.cell_type, cfg.dropout)
    dec = AttentionDecoder(len(tgt_vocab), cfg.embed_dim, cfg.hidden_dim,
                           cfg.decoder_layers, cfg.cell_type, cfg.dropout)
    model = Seq2SeqAttention(enc, dec, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── 6) train & track this run’s best ──────────────────────────────────
    run_best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt[:, :-1])
            loss    = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                tgt[:,1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        wandb.log({"train_loss": train_loss/len(train_loader), "epoch": epoch+1})

        # validation exact‐match
        model.eval()
        correct, total = 0, 0
        for src, tgt in dev_loader:
            src, tgt = src.to(device), tgt.to(device)
            seq = model.beam_search(
                src.squeeze(0),
                sos_idx = tgt_vocab.char2idx["<sos>"],
                eos_idx = tgt_vocab.char2idx["<eos>"],
                max_len = cfg.max_len,
                beam_size=cfg.beam_size
            )
            pred = tgt_vocab.decode(seq[1:])  # drop <sos>
            true = tgt_vocab.decode(tgt.squeeze(0).tolist()[1:])
            if pred == true: correct += 1
            total += 1
        dev_acc = correct / total
        wandb.log({"dev_accuracy": dev_acc})

        # checkpoint this run’s best
        if dev_acc > run_best_acc:
            run_best_acc = dev_acc
            torch.save(model.state_dict(), tmp_ckpt)

    # ── 7) compare to GLOBAL best ─────────────────────────────────────────
    if os.path.exists(global_json):
        with open(global_json, "r") as f:
            prev = json.load(f)
            prev_acc = prev.get("best_acc", 0.0)
    else:
        prev_acc = 0.0

    if run_best_acc > prev_acc:
        # update global record (including config if you like)
        with open(global_json, "w") as f:
            json.dump({
                "best_acc": run_best_acc,
                "run_name": run_name,
                "config": dict(cfg)
            }, f)
        # replace the global best model
        os.replace(tmp_ckpt, global_model)
        print(f"🏆 New global best attention run: {run_name} @ {run_best_acc:.4f}")
    else:
        os.remove(tmp_ckpt)
        print(f"Run {run_name} (best {run_best_acc:.4f}) did not beat global {prev_acc:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
