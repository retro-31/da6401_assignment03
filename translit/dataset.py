import pickle
from torch.utils.data import Dataset
import torch

class TransliterationDataset(Dataset):
    def __init__(self, split, src_vocab, tgt_vocab, max_len=32):
        """
        split: "train", "dev", or "test"
        """
        path = f"data/processed/{split}.pkl"
        with open(path, "rb") as f:
            self.pairs = pickle.load(f)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        src_ids = self.src_vocab.encode(src)[: self.max_len]
        tgt_ids = self.tgt_vocab.encode(tgt)[: self.max_len]

        # pad
        pad = self.src_vocab.char2idx["<pad>"]
        src_ids += [pad] * (self.max_len - len(src_ids))
        tgt_ids += [pad] * (self.max_len - len(tgt_ids))

        return torch.tensor(src_ids), torch.tensor(tgt_ids)