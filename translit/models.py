# translit/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Encoder (returns full outputs and final hidden state)
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, src):
        """
        Args:
            src: [batch_size, src_len]
        Returns:
            outputs: [batch_size, src_len, hidden_dim]
            hidden:  [n_layers, batch_size, hidden_dim]  or (h_n, c_n) tuple for LSTM
        """
        emb = self.embedding(src)              # [batch, src_len, embed_dim]
        outputs, hidden = self.rnn(emb)        # outputs & final hidden/cell
        return outputs, hidden                 # both used by attention

# ──────────────────────────────────────────────────────────────────────────────
# Decoder (vanilla seq2seq)
# ──────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, trg, hidden):
        """
        Args:
            trg:    [batch_size, trg_len]
            hidden: [n_layers, batch_size, hidden_dim]
        Returns:
            preds:  [batch_size, trg_len, vocab_size]
            hidden: updated hidden state
        """
        emb = self.embedding(trg)                   # [batch, trg_len, embed_dim]
        outputs, hidden = self.rnn(emb, hidden)     # [batch, trg_len, hidden_dim]
        preds = self.out(outputs)                   # [batch, trg_len, vocab_size]
        return preds, hidden

# ──────────────────────────────────────────────────────────────────────────────
# Seq2Seq (vanilla) with adaptive hidden sizing
# ──────────────────────────────────────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def _adapt_hidden(self, enc_hidden):
        """
        Slice or pad encoder hidden to match decoder.num_layers.
        """
        def adapt(h):
            enc_layers, batch, hid = h.size()
            dec_layers = self.decoder.rnn.num_layers
            if enc_layers == dec_layers:
                return h
            elif enc_layers > dec_layers:
                return h[:dec_layers]
            else:
                pad = h.new_zeros(dec_layers - enc_layers, batch, hid)
                return torch.cat([h, pad], dim=0)

        if isinstance(enc_hidden, tuple):
            h_n, c_n = enc_hidden
            return (adapt(h_n), adapt(c_n))
        else:
            return adapt(enc_hidden)

    def forward(self, src, trg):
        """
        src: [batch, src_len]
        trg: [batch, trg_len] (with <sos> at trg[:,0])
        """
        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = self._adapt_hidden(enc_hidden)
        preds, _ = self.decoder(trg, dec_hidden)
        return preds

    def beam_search(self, src_seq, sos_idx, eos_idx, max_len, beam_size):
        src = src_seq.unsqueeze(0).to(self.device)           # [1, src_len]
        enc_outputs, enc_hidden = self.encoder(src)
        dec_hidden = self._adapt_hidden(enc_hidden)

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
                logp = F.log_softmax(out[0, -1], dim=-1)        # [vocab_size]
                topk = torch.topk(logp, beam_size)
                for k in range(beam_size):
                    t = topk.indices[k].item()
                    s = score + topk.values[k].item()
                    all_cand.append((tokens + [t], s, h_new))
            candidates = sorted(all_cand, key=lambda x: x[1], reverse=True)[:beam_size]

        completed += [(tok, score) for tok, score, _ in candidates]
        best = max(completed, key=lambda x: x[1])[0]
        return best

# ──────────────────────────────────────────────────────────────────────────────
# Attention components
# ──────────────────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        """
        hidden:           [batch, hidden_dim]
        encoder_outputs:  [batch, src_len, hidden_dim]
        """
        batch, src_len, hid = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)    # [batch,src_len,hid]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)                      # [batch,hid,src_len]
        v = self.v.repeat(batch,1).unsqueeze(1)               # [batch,1,hid]
        scores = torch.bmm(v, energy).squeeze(1)              # [batch,src_len]
        return F.softmax(scores, dim=1)                       # [batch,src_len]

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, cell_type, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, trg, hidden, encoder_outputs):
        batch, trg_len = trg.size()
        embed = self.embedding(trg)                          # [batch, trg_len, emb]
        outputs, attentions = [], []

        for t in range(trg_len):
            emb_t = embed[:, t].unsqueeze(1)                  # [batch,1,emb]
            if isinstance(hidden, tuple):
                h_t = hidden[0][-1]                          # [batch,hid]
            else:
                h_t = hidden[-1]                             # [batch,hid]
            attn_w = self.attention(h_t, encoder_outputs)    # [batch,src_len]
            context = torch.bmm(attn_w.unsqueeze(1), encoder_outputs)  # [batch,1,hid]
            rnn_in = torch.cat((emb_t, context), dim=2)      # [batch,1,emb+hid]
            out, hidden = self.rnn(rnn_in, hidden)           # [batch,1,hid]
            concat = torch.cat((out, context), dim=2)        # [batch,1,hid*2]
            pred   = self.fc(concat)                         # [batch,1,vocab]
            outputs.append(pred)
            attentions.append(attn_w.unsqueeze(1))           # [batch,1,src_len]

        outputs    = torch.cat(outputs, dim=1)               # [batch,trg_len,vocab]
        attentions = torch.cat(attentions, dim=1)            # [batch,trg_len,src_len]
        return outputs, hidden, attentions

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, trg):
        enc_outs, enc_hidden = self.encoder(src)
        outputs, _, _ = self.decoder(trg, enc_hidden, enc_outs)
        return outputs

    def beam_search(self, src_seq, sos_idx, eos_idx, max_len, beam_size):
        src    = src_seq.unsqueeze(0).to(self.device)
        enc_outs, enc_hidden = self.encoder(src)
        beams  = [([sos_idx], 0.0, enc_hidden)]
        completed = []

        for _ in range(max_len):
            all_beams = []
            for tokens, score, hidden in beams:
                if tokens[-1] == eos_idx:
                    completed.append((tokens, score))
                    continue
                last = torch.tensor([[tokens[-1]]], device=self.device)
                out, h_new, _ = self.decoder(last, hidden, enc_outs)
                logp = F.log_softmax(out[0, -1], dim=-1)
                topk = torch.topk(logp, beam_size)
                for i in range(beam_size):
                    t = topk.indices[i].item()
                    s = score + topk.values[i].item()
                    all_beams.append((tokens + [t], s, h_new))
            beams = sorted(all_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        completed += beams
        best = max(completed, key=lambda x: x[1])[0]
        return best
