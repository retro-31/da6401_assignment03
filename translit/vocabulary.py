class Vocab:
    def __init__(self, char2idx, idx2char):
        self.char2idx = char2idx
        self.idx2char = idx2char

    def __len__(self):
        return len(self.idx2char)

    def encode(self, sequence, add_sos_eos=True):
        ids = [self.char2idx.get(c, self.char2idx["<unk>"]) for c in sequence]
        if add_sos_eos:
            return [self.char2idx["<sos>"]] + ids + [self.char2idx["<eos>"]]
        return ids

    def decode(self, ids):
        # strip until <eos>
        res = []
        for i in ids:
            if i == self.char2idx["<eos>"]:
                break
            res.append(self.idx2char[i])
        return "".join(res)
