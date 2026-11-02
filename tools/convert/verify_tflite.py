"""Check a next-token .tflite against float reference logits.

Runs in the TF 2.17 venv, i.e. the same kernels (XNNPACK hybrid FC) as the
Android `org.tensorflow:tensorflow-lite:2.17.0` CPU path. Metrics are
shift-invariant (the LM head is mean-centred, which moves all logits by a
constant):
  top1             argmax agrees with reference
  top3_overlap     |top3 ∩ ref top3| / 3
  ref_top1_in_top3 the reference's best token is among our 3 suggestions
  kl               KL(p_ref || p_model)

Usage: python verify_tflite.py <model.tflite> <ref.npz> [--min-ref1-in-top3 F] [--max-kl F]
       ref.npz holds ids int[N,T] (right-padded), lens int[N], logits float[N,V]
"""
import argparse
import time

import numpy as np
import tensorflow as tf

from gpt2_np import PAD_ID, agreement


def run(model_path, ids, lens, threads=2):
    it = tf.lite.Interpreter(model_path=model_path, num_threads=threads)
    it.allocate_tensors()
    inp = {d["name"]: d["index"] for d in it.get_input_details()}
    ids_idx = next(v for k, v in inp.items() if "input_ids" in k)
    last_idx = next(v for k, v in inp.items() if "last_index" in k)
    out_idx = it.get_output_details()[0]["index"]
    T = it.get_input_details()[0]["shape"][1]
    logits, times = [], []
    for row, n in zip(ids, lens):
        x = np.full(T, PAD_ID, np.int32)
        x[:n] = row[:n]
        it.set_tensor(ids_idx, x[None])
        it.set_tensor(last_idx, np.array([n - 1], np.int32))
        t = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - t)
        logits.append(it.get_tensor(out_idx)[0].copy())
    return np.stack(logits), float(np.median(times[1:]) * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("reference")
    ap.add_argument("--min-ref1-in-top3", type=float, default=0.0)
    ap.add_argument("--max-kl", type=float, default=float("inf"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--metrics-out")
    a = ap.parse_args()
    ref = np.load(a.reference)
    got, ms = run(a.model, ref["ids"], ref["lens"], a.threads)
    m = agreement(ref["logits"], got)
    print(f"{a.model} vs {a.reference} (n={len(got)}): "
          + " ".join(f"{k}={v:.3f}" for k, v in m.items())
          + f" | host latency {a.threads}T median {ms:.1f} ms")
    if a.metrics_out:
        import json
        json.dump({k: round(v, 4) for k, v in m.items()} | {"n": len(got), "reference": a.reference.rsplit("/", 1)[-1]},
                  open(a.metrics_out, "w"), indent=1)
    if m["ref_top1_in_top3"] < a.min_ref1_in_top3 or m["kl"] > a.max_kl:
        raise SystemExit("FAIL: model accuracy below threshold")


if __name__ == "__main__":
    main()
