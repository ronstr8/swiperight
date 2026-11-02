"""Reference GPT-2 forward pass in numpy, used for calibration and evaluation.

Mirrors build_tflite.py op for op (right-padded fixed window, causal mask,
LM head evaluated only at the last real position). Every linear layer goes
through a pluggable `fc(x, name)` so quantization schemes can be simulated
without converting a model.
"""
import json

import numpy as np
from safetensors.numpy import load_file

PAD_ID = 50256  # <|endoftext|>; value is irrelevant under causal attention


class GPT2:
    def __init__(self, model_dir: str, seq_len: int):
        cfg = json.load(open(f"{model_dir}/config.json"))
        raw = load_file(f"{model_dir}/model.safetensors")
        self.w = {k.removeprefix("transformer."): v.astype(np.float32) for k, v in raw.items()}
        self.n_layer, self.n_head, self.d = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
        self.eps = cfg["layer_norm_epsilon"]
        self.T = seq_len
        wte = self.w["wte.weight"]
        # Softmax is invariant to adding the same value to every logit, and
        # h . (e_v - mu) = h . e_v - h . mu, so the LM head can use mean-centred
        # embeddings with identical predictions. Centring removes a shared
        # direction that otherwise dominates every row's int4 range.
        self.head_weight = (wte - wte.mean(0)).T.copy()  # [d, V]
        self.vocab = wte.shape[0]

    def fc_names(self):
        names = []
        for i in range(self.n_layer):
            for k in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
                names.append(f"h.{i}.{k}")
        return names + ["head"]

    def weight(self, name):
        """[in, out] weight and [out] bias for an FC site."""
        if name == "head":
            return self.head_weight, np.zeros(self.vocab, np.float32)
        return self.w[name + ".weight"], self.w[name + ".bias"]

    def float_fc(self, x, name):
        W, b = self.weight(name)
        return x @ W + b

    def ln(self, x, p):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + self.eps) * self.w[p + ".weight"] + self.w[p + ".bias"]

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, ids, lens, fc=None, embed=None, all_positions_head=False):
        """ids int[B, T] right-padded, lens int[B] -> logits float[B, V]."""
        fc = fc or self.float_fc
        B, T, H, d = ids.shape[0], self.T, self.n_head, self.d
        hd = d // H
        E = self.w["wte.weight"] if embed is None else embed
        h = E[ids] + self.w["wpe.weight"][:T]
        mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
        for i in range(self.n_layer):
            p = f"h.{i}."
            qkv = fc(self.ln(h, p + "ln_1"), p + "attn.c_attn")
            q, k, v = [t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, -1)]
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            h = h + fc(o, p + "attn.c_proj")
            h = h + fc(self.gelu(fc(self.ln(h, p + "ln_2"), p + "mlp.c_fc")), p + "mlp.c_proj")
        if all_positions_head:  # calibration only: LM-head inputs at every position
            return fc(self.ln(h, "ln_f"), "head")
        last = self.ln(h[np.arange(B), lens - 1], "ln_f")
        return fc(last, "head")


def windows_from_ids(token_lists, seq_len, rng, per_text=4):
    """Cut right-padded training windows (ids, lens) out of tokenized texts."""
    ids, lens = [], []
    for toks in token_lists:
        if len(toks) < 2:
            continue
        for _ in range(per_text):
            n = int(rng.integers(2, min(seq_len, len(toks)) + 1))
            start = int(rng.integers(0, len(toks) - n + 1))
            row = np.full(seq_len, PAD_ID, np.int32)
            row[:n] = toks[start:start + n]
            ids.append(row)
            lens.append(n)
    return np.stack(ids), np.array(lens)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def agreement(ref_logits, got_logits):
    """Metrics that matter for a 3-slot suggestion strip."""
    r = np.argsort(-ref_logits, 1)[:, :3]
    g = np.argsort(-got_logits, 1)[:, :3]
    pr, pq = softmax(ref_logits), softmax(got_logits)
    return {
        "top1": float((r[:, 0] == g[:, 0]).mean()),
        "top3_overlap": float(np.mean([len(set(a) & set(b)) / 3 for a, b in zip(r, g)])),
        "ref_top1_in_top3": float(np.mean([r[i, 0] in g[i] for i in range(len(r))])),
        "kl": float((pr * (np.log(pr + 1e-12) - np.log(pq + 1e-12))).sum(1).mean()),
    }
