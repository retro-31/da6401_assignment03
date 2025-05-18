# train_attention_global_best.py

import os
import json
import pickle
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from translit.vocabulary import Vocab
from translit.dataset    import TransliterationDataset
from translit.models     import Encoder, AttentionDecoder, Seq2SeqAttention

def main():
    # ── 1) Defaults & W&B init ─────────────────────────────────────────────
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
    wandb.init(project="assignment-03", name="attention_run", config=defaults)
    cfg = wandb.config

    # ── 2) Global‐best metadata & model paths ───────────────────────────────
    os.makedirs("runs", exist_ok=True)
    GLOBAL_JSON  = "runs/attention_global_best.json"
    GLOBAL_MODEL = "runs/best_attention_model.pt"

    # load previous global best accuracy
    if os.path.exists(GLOBAL_JSON):
        with open(GLOBAL_JSON) as f:
            prev_acc = json.load(f).get("best_acc", 0.0)
    else:
        prev_acc = 0.0

    # ── 3) Data, vocabs ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("data/processed/src2idx.pkl","rb") as f: src2idx = pickle.load(f)
    with open("data/processed/idx2src.pkl","rb") as f: idx2src = pickle.load(f)
    with open("data/processed/tgt2idx.pkl","rb") as f: tgt2idx = pickle.load(f)
    with open("data/processed/idx2tgt.pkl","rb") as f: idx2tgt = pickle.load(f)
    src_vocab = Vocab(src2idx, idx2src)
    tgt_vocab = Vocab(tgt2idx, idx2tgt)

    train_loader = DataLoader(
        TransliterationDataset("train", src_vocab, tgt_vocab, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=True
    )
    dev_loader = DataLoader(
        TransliterationDataset("dev",   src_vocab, tgt_vocab, cfg.max_len),
        batch_size=1
    )

    # ── 4) Build model, loss, optimizer ────────────────────────────────────
    enc = Encoder(len(src_vocab), cfg.embed_dim, cfg.hidden_dim,
                  cfg.encoder_layers, cfg.cell_type, cfg.dropout)
    dec = AttentionDecoder(len(tgt_vocab), cfg.embed_dim, cfg.hidden_dim,
                           cfg.decoder_layers, cfg.cell_type, cfg.dropout)
    model = Seq2SeqAttention(enc, dec, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ── 5) Training: track this run’s best and keep its state_dict ────────
    run_best_acc   = 0.0
    run_best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, tgt[:, :-1])
            loss = criterion(
                out.view(-1, out.size(-1)),
                tgt[:,1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        wandb.log({"train_loss": epoch_loss/len(train_loader), "epoch": epoch+1})

        # validation exact‐match
        model.eval()
        corr, tot = 0, 0
        for src, tgt in dev_loader:
            src, tgt = src.to(device), tgt.to(device)
            seq = model.beam_search(
                src.squeeze(0),
                sos_idx = tgt_vocab.char2idx["<sos>"],
                eos_idx = tgt_vocab.char2idx["<eos>"],
                max_len = cfg.max_len,
                beam_size=cfg.beam_size
            )
            pred = tgt_vocab.decode(seq[1:])
            true = tgt_vocab.decode(tgt.squeeze(0).tolist()[1:])
            corr += int(pred == true)
            tot  += 1
        dev_acc = corr/tot
        wandb.log({"dev_accuracy": dev_acc})

        # if this epoch is new best for *this* run, stash the state_dict
        if dev_acc > run_best_acc:
            run_best_acc   = dev_acc
            run_best_state = {k:v.cpu() for k,v in model.state_dict().items()}

    # ── 6) Compare & save global best only ─────────────────────────────────
    if run_best_acc > prev_acc:
        # update metadata
        with open(GLOBAL_JSON, "w") as f:
            json.dump({
                "best_acc": run_best_acc,
                "config":   dict(cfg)
            }, f)
        # overwrite global model
        torch.save(run_best_state, GLOBAL_MODEL)
        print(f"🏆 New global best dev_accuracy={run_best_acc:.4f}")
    else:
        print(f"No improvement over global {prev_acc:.4f}; nothing saved.")

    wandb.finish()

if __name__ == "__main__":
    main()
