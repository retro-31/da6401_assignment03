import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)

    def forward(self, src):
        # src: [batch, T]
        emb = self.embedding(src)            # [batch, T, embed_dim]
        outputs, hidden = self.rnn(emb)      # hidden: tensor or (h_n, c_n)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, n_layers,
                           dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, trg, hidden):
        # trg: [batch, T]
        emb = self.embedding(trg)                # [batch, T, embed_dim]
        outputs, hidden = self.rnn(emb, hidden)  # outputs: [batch, T, hidden_dim]
        preds = self.out(outputs)                # [batch, T, vocab_size]
        return preds, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def _adapt_hidden(self, enc_hidden):
        """
        Adjust encoder hidden state to match decoder layers.
        """
        # helper to slice or pad h_n tensors
        def adapt(h):
            enc_layers, batch, hid = h.size()
            dec_layers = self.decoder.rnn.num_layers
            if enc_layers == dec_layers:
                return h
            elif enc_layers > dec_layers:
                return h[:dec_layers]
            else:
                # pad with zeros on top for additional layers
                pad = h.new_zeros(dec_layers - enc_layers, batch, hid)
                return torch.cat([h, pad], dim=0)

        if isinstance(enc_hidden, tuple):
            # LSTM: tuple of (h_n, c_n)
            h_n, c_n = enc_hidden
            return (adapt(h_n), adapt(c_n))
        else:
            # RNN or GRU: single tensor
            return adapt(enc_hidden)

    def forward(self, src, trg):
        """
        src: [batch, T_src]
        trg: [batch, T_trg] (with <sos> at trg[:,0])
        """
        enc_hidden = self.encoder(src)
        dec_hidden = self._adapt_hidden(enc_hidden)
        outputs, _ = self.decoder(trg, dec_hidden)
        return outputs

    def beam_search(self, src_seq, sos_idx, eos_idx, max_len, beam_size):
        """
        src_seq: [T_src] single sequence tensor (no batch dim)
        returns the best token-id list (including <sos> and <eos>).
        """
        src = src_seq.unsqueeze(0).to(self.device)            # [1, T_src]
        enc_hidden = self.encoder(src)
        dec_hidden = self._adapt_hidden(enc_hidden)

        # each candidate: (tokens, score, hidden)
        candidates = [([sos_idx], 0.0, dec_hidden)]
        completed  = []

        for _ in range(max_len):
            all_cand = []
            for tokens, score, hidden in candidates:
                if tokens[-1] == eos_idx:
                    completed.append((tokens, score))
                    continue
                last_tok = torch.tensor([[tokens[-1]]], device=self.device)
                out, h_new = self.decoder(last_tok, hidden)
                logp = F.log_softmax(out[0, -1], dim=-1)  # [vocab_size]
                topk = torch.topk(logp, beam_size)
                for k in range(beam_size):
                    tok = topk.indices[k].item()
                    sc  = score + topk.values[k].item()
                    all_cand.append((tokens + [tok], sc, h_new))
            # keep top beam_size
            candidates = sorted(all_cand, key=lambda x: x[1], reverse=True)[:beam_size]

        # add unfinished
        completed += [(tokens, score) for tokens,score,_ in candidates]
        best = max(completed, key=lambda x: x[1])[0]
        return best
