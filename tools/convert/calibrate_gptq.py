"""Compute 4-bit weights for distilgpt2: SmoothQuant scaling + GPTQ rounding.

Why this exists: plain per-channel INT4 rounding of distilgpt2 is unusable
(top-1 agreement with float ~5%). Measured causes and fixes:

  1. LM head (tied embedding) rows share a large mean direction that eats the
     int4 range -> use mean-centred head weights (softmax-invariant, exact).
  2. A few hidden channels carry huge activations; dynamic per-row int8
     activation quantization in the hybrid FC kernel crushes everything else
     -> SmoothQuant: divide FC inputs by s, multiply weight rows by s
     (alpha = 0.5). Exact in float; folded into LayerNorm where possible.
  3. Round-to-nearest int4 error accumulates -> GPTQ (Hessian-aware rounding
     with error feedback, act-order) using a small calibration corpus.

TFLite 2.17 (the last runtime that still ships the NNAPI delegate) executes
per-channel INT4 FULLY_CONNECTED weights correctly but silently produces
garbage for block-wise INT4, so everything here is per-output-channel.

Outputs quant_params.npz with, per FC site: smooth[in], codes int8[in,out]
(values in [-7, 7]), scale float32[out]; plus the int4 per-row embedding table
used by the input GATHER. Also writes eval.npz (held-out windows + float
logits) for verify_tflite.py.

Usage: python calibrate_gptq.py <hf_model_dir> <data_dir> <out_dir> [seq_len] [--int8 site,...]
"""
import argparse
import os
import time

import numpy as np
from tokenizers import Tokenizer

from gpt2_np import GPT2, PAD_ID, agreement, softmax, windows_from_ids

ALPHA = 0.5
QMAX = 7  # symmetric int4: [-7, 7]


def gptq(W, X, bits, damp=0.01):
    """GPTQ with act-order. W [in, out], X [n, in] -> (codes int8 [in,out], scale [out])."""
    q = 2 ** (bits - 1) - 1
    W = W.astype(np.float64).copy()
    n_in = W.shape[0]
    H = X.T.astype(np.float64) @ X / len(X)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[dead] = 0
    H += damp * np.mean(np.diag(H)) * np.eye(n_in)
    perm = np.argsort(-np.diag(H))
    W, H = W[perm], H[perm][:, perm]
    scale = np.abs(W).max(0) / q
    scale[scale == 0] = 1
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T  # upper triangular
    codes = np.zeros(W.shape, np.int8)
    B = 128
    for i1 in range(0, n_in, B):
        i2 = min(i1 + B, n_in)
        W1, Hi = W[i1:i2].copy(), Hinv[i1:i2, i1:i2]
        E = np.zeros_like(W1)
        for j in range(i2 - i1):
            k = np.clip(np.round(W1[j] / scale), -q, q)
            codes[i1 + j] = k
            e = (W1[j] - k * scale) / Hi[j, j]
            W1[j:] -= np.outer(Hi[j, j:], e)
            E[j] = e
        W[i2:] -= Hinv[i1:i2, i2:].T @ E
    return codes[np.argsort(perm)], scale.astype(np.float32)


def per_row_int4(M):
    """Round-to-nearest symmetric int4 per row. M [V, d] -> (codes int8, scale [V])."""
    scale = np.abs(M).max(1) / QMAX
    scale[scale == 0] = 1
    codes = np.clip(np.round(M / scale[:, None]), -QMAX, QMAX).astype(np.int8)
    return codes, scale.astype(np.float32)


def qa_rows(x):
    """Simulate the hybrid kernel's dynamic per-row asymmetric int8 activations."""
    sh = x.shape
    x = x.reshape(-1, sh[-1])
    lo = np.minimum(x.min(1, keepdims=True), 0)
    hi = np.maximum(x.max(1, keepdims=True), 0)
    s = (hi - lo) / 255
    s[s == 0] = 1
    z = np.round(-lo / s)
    return ((np.clip(np.round(x / s) + z, 0, 255) - z) * s).reshape(sh)


def sample_windows(m, n, seq_len, seed):
    """Extra calibration diversity: short continuations sampled from the float model."""
    rng = np.random.default_rng(seed)
    ids = np.full((n, seq_len), PAD_ID, np.int32)
    ids[:, 0] = rng.integers(0, PAD_ID, n)
    for t in range(1, seq_len):
        p = softmax(m.forward(ids, np.full(n, t)) / 0.9)
        for b in range(n):
            o = np.argsort(-p[b])
            o = o[: np.searchsorted(np.cumsum(p[b][o]), 0.9) + 1]
            ids[b, t] = rng.choice(o, p=p[b][o] / p[b][o].sum())
    return ids, rng.integers(4, seq_len + 1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("data_dir"); ap.add_argument("out_dir")
    ap.add_argument("seq_len", type=int, nargs="?", default=16)
    ap.add_argument("--int8", default="", help="comma-separated FC sites to keep at int8")
    a = ap.parse_args()
    int8_sites = set(filter(None, a.int8.split(",")))
    t0 = time.time()

    m = GPT2(a.model_dir, a.seq_len)
    tok = Tokenizer.from_file(f"{a.model_dir}/tokenizer.json")
    enc = lambda path: [tok.encode(l.strip()).ids for l in open(path) if l.strip()]
    rng = np.random.default_rng(1234)

    ci, cl = windows_from_ids(enc(f"{a.data_dir}/calibration.txt"), a.seq_len, rng, per_text=8)
    gi, gl = sample_windows(m, 96, a.seq_len, seed=7)
    ci, cl = np.concatenate([ci, gi]), np.concatenate([cl, gl])
    ei, el = windows_from_ids(enc(f"{a.data_dir}/eval.txt"), a.seq_len, rng, per_text=6)
    si, sl = sample_windows(m, 64, a.seq_len, seed=99)  # held-out, generated
    ei, el = np.concatenate([ei, si]), np.concatenate([el, sl])
    print(f"calibration windows={len(ci)} eval windows={len(ei)} ({time.time()-t0:.0f}s)", flush=True)

    # Collect float inputs to every FC site. Only real (non-pad) rows count.
    X = {}
    valid = (np.arange(a.seq_len)[None, :] < cl[:, None]).reshape(-1)

    def collect(x, name):
        rows = x.reshape(-1, x.shape[-1])
        if name != "head":
            X.setdefault(name, []).append(rows[valid])
        return m.float_fc(x, name)

    m.forward(ci, cl, fc=collect)

    # The LM head sees one row per window at inference time, which is too few
    # samples for a 768-dim Hessian; calibrate it on ln_f output at every
    # real position instead (same distribution, 10x the rows).
    hx = []

    def collect_head(x, name):
        if name != "head":
            return m.float_fc(x, name)
        hx.append(x.reshape(-1, x.shape[-1])[valid])
        return np.zeros((len(x), 1), np.float32)

    m.forward(ci, cl, fc=collect_head, all_positions_head=True)
    X = {k: np.concatenate(v) for k, v in X.items()}
    X["head"] = np.concatenate(hx)

    out = {}
    for name in m.fc_names():
        W, _ = m.weight(name)
        ax = np.maximum(np.abs(X[name]).max(0), 1e-5)
        aw = np.maximum(np.abs(W).max(1), 1e-5)
        s = np.clip(ax ** ALPHA / aw ** (1 - ALPHA), 1e-2, 1e2).astype(np.float32)
        bits = 8 if name in int8_sites else 4
        codes, scale = gptq(W * s[:, None], X[name] / s, bits)
        out[f"{name}/smooth"], out[f"{name}/codes"], out[f"{name}/scale"] = s, codes, scale
        out[f"{name}/bits"] = np.int32(bits)
        print(f"  {name:18s} int{bits} rows={len(X[name])}", flush=True)
    ecodes, escale = per_row_int4(m.w["wte.weight"])
    out["embed/codes"], out["embed/scale"] = ecodes, escale

    # Simulated accuracy of exactly these parameters, before any conversion.
    def qfc(x, name):
        s, c, sc = out[f"{name}/smooth"], out[f"{name}/codes"], out[f"{name}/scale"]
        _, b = m.weight(name)
        return qa_rows(x / s) @ (c.astype(np.float32) * sc) + b

    ref = m.forward(ei, el)
    sim = m.forward(ei, el, fc=qfc, embed=ecodes.astype(np.float32) * escale[:, None])
    print("simulated vs float:", {k: round(v, 3) for k, v in agreement(ref, sim).items()})

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(f"{a.out_dir}/quant_params.npz", **out)
    np.savez(f"{a.out_dir}/eval.npz", ids=ei, lens=el, logits=ref)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
