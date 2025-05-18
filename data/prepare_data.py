# data/prepare_data.py
import os, sys, pickle
import pandas as pd

LANG = "hi"   # change as needed
BASE_DIR = "data/dakshina_dataset_v1.0"
LEX_DIR = os.path.join(BASE_DIR, LANG, "lexicons")
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(LEX_DIR):
    sys.stderr.write(f"Missing TSVs under {LEX_DIR}\n"); sys.exit(1)

splits = {}
for split in ("train", "dev", "test"):
    path = os.path.join(LEX_DIR, f"{LANG}.translit.sampled.{split}.tsv")
    # read everything as str, then drop any nulls
    df = pd.read_csv(path, sep="\t", header=None, names=["src","tgt"], dtype=str)
    df = df.dropna(subset=["src","tgt"])
    # strip whitespace
    srcs = [s.strip() for s in df["src"]]
    tgts = [t.strip() for t in df["tgt"]]
    splits[split] = list(zip(srcs, tgts))

def build_char_vocab(pairs):
    chars = set()
    for s,t in pairs:
        chars.update(s); chars.update(t)
    idx2char = ["<pad>","<unk>","<sos>","<eos>"] + sorted(chars)
    return {c:i for i,c in enumerate(idx2char)}, idx2char

all_pairs = splits["train"] + splits["dev"]
src2idx, idx2src = build_char_vocab(all_pairs)
tgt2idx, idx2tgt = build_char_vocab(all_pairs)

# save
with open(f"{OUT_DIR}/src2idx.pkl","wb") as f: pickle.dump(src2idx, f)
with open(f"{OUT_DIR}/idx2src.pkl","wb") as f: pickle.dump(idx2src, f)
with open(f"{OUT_DIR}/tgt2idx.pkl","wb") as f: pickle.dump(tgt2idx, f)
with open(f"{OUT_DIR}/idx2tgt.pkl","wb") as f: pickle.dump(idx2tgt, f)
for split, data in splits.items():
    with open(f"{OUT_DIR}/{split}.pkl","wb") as f: pickle.dump(data, f)

print("✅ data/processed ready with train/dev/test + vocabs")

