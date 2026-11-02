"""Build the distilgpt2 next-token TFLite graph (float container, int4-ready).

The GPT-2 forward pass is written out by hand in TensorFlow from the Hugging
Face safetensors weights so the graph is small and predictable on device:

  * fixed shapes: input_ids int32[1, SEQ_LEN], last_index int32[1]
  * the sequence is right-padded. Attention is causal, so pad tokens after
    `last_index` cannot affect the hidden state at `last_index`; no attention
    mask input is needed
  * the last real position is selected in-graph, so the LM head runs for one
    row and the output is float32[1, VOCAB] logits (~200 KB), not [T, VOCAB]

With --quant-params (from calibrate_gptq.py) the graph is prepared for INT4:

  * every FULLY_CONNECTED weight is the GPTQ-dequantized float tensor
    (codes * scale), so quantize.py can swap in the exact int4 codes
  * SmoothQuant input scaling is folded into the preceding LayerNorm affine
    where there is one (qkv, mlp-in, LM head) and is an explicit elementwise
    MUL before attn-out / mlp-out
  * the LM head uses mean-centred embeddings (softmax-invariant)
  * the input embedding is a separate uint8 table holding two int4 codes per
    byte with a per-row scale, unpacked in-graph with integer ops. It cannot
    share the LM-head tensor because TFLite has no INT4 GATHER

Usage: python build_tflite.py <hf_model_dir> <out.tflite> [--seq-len N] [--quant-params P]
"""
import argparse

import numpy as np
import tensorflow as tf

from gpt2_np import GPT2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--quant-params")
    a = ap.parse_args()

    m = GPT2(a.model_dir, a.seq_len)
    w, T, d, H = m.w, a.seq_len, m.d, m.n_head
    hd = d // H
    qp = dict(np.load(a.quant_params)) if a.quant_params else None

    def fc_weight(name):
        """Float [in, out] weight and the 1/s input scaling (or None)."""
        if qp is None:
            W, _ = m.weight(name)
            return W, None
        W = qp[f"{name}/codes"].astype(np.float32) * qp[f"{name}/scale"]
        return W, (1.0 / qp[f"{name}/smooth"]).astype(np.float32)

    def fc_bias(name):
        return m.weight(name)[1]

    causal = tf.constant(np.triu(np.full((T, T), -1e9, np.float32), k=1))
    wpe = tf.constant(w["wpe.weight"][:T])

    if qp is None:
        wte = tf.constant(w["wte.weight"])
        embed = lambda ids: tf.gather(wte, ids)
    else:
        c = qp["embed/codes"].astype(np.int16) + 8  # [-7,7] -> [1,15]
        packed = (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)  # [V, d/2], low nibble first
        table = tf.constant(packed)
        escale = tf.constant(qp["embed/scale"])

        def embed(ids):
            b = tf.cast(tf.gather(table, ids), tf.int32)  # [T, d/2]
            lo = tf.math.floormod(b, 16)
            hi = tf.math.floordiv(b, 16)
            codes = tf.reshape(tf.stack([lo, hi], axis=-1), [T, d])
            return (tf.cast(codes, tf.float32) - 8.0) * tf.expand_dims(tf.gather(escale, ids), 1)

    def layer_norm(x, p, inv_s=None):
        g, b = w[p + ".weight"], w[p + ".bias"]
        if inv_s is not None:  # fold SmoothQuant 1/s into the LN affine (exact)
            g, b = g * inv_s, b * inv_s
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + m.eps) * tf.constant(g) + tf.constant(b)

    def linear(x, name, scaled_input=False):
        W, inv_s = fc_weight(name)
        if inv_s is not None and not scaled_input:
            x = x * tf.constant(inv_s)
        y = tf.matmul(x, tf.constant(W))
        b = fc_bias(name)
        return y + tf.constant(b) if b is not None else y

    def heads(x):  # [T, d] -> [H, T, hd]
        return tf.transpose(tf.reshape(x, [T, H, hd]), [1, 0, 2])

    @tf.function(input_signature=[
        tf.TensorSpec([1, T], tf.int32, name="input_ids"),
        tf.TensorSpec([1], tf.int32, name="last_index"),
    ])
    def next_token_logits(input_ids, last_index):
        h = embed(input_ids[0]) + wpe  # [T, d]
        for i in range(m.n_layer):
            p = f"h.{i}."
            a_ = layer_norm(h, p + "ln_1", fc_weight(p + "attn.c_attn")[1])
            q, k, v = tf.split(linear(a_, p + "attn.c_attn", scaled_input=True), 3, axis=-1)
            q, k, v = heads(q), heads(k), heads(v)
            s = tf.matmul(q, k, transpose_b=True) * (1.0 / np.sqrt(hd)) + causal
            o = tf.matmul(tf.nn.softmax(s, axis=-1), v)
            o = tf.reshape(tf.transpose(o, [1, 0, 2]), [T, d])
            h = h + linear(o, p + "attn.c_proj")
            x = layer_norm(h, p + "ln_2", fc_weight(p + "mlp.c_fc")[1])
            x = tf.nn.gelu(linear(x, p + "mlp.c_fc", scaled_input=True), approximate=True)
            h = h + linear(x, p + "mlp.c_proj")
        last = tf.gather(h, last_index)  # [1, d]
        last = layer_norm(last, "ln_f", fc_weight("head")[1])
        W, _ = fc_weight("head")
        return {"logits": tf.matmul(last, tf.constant(W))}  # [1, V]

    conv = tf.lite.TFLiteConverter.from_concrete_functions(
        [next_token_logits.get_concrete_function()], trackable_obj=None)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # Keep the SmoothQuant MUL as its own op. Fusing it into the next FC's
    # weights would move them off the int4 grid that GPTQ chose.
    conv._experimental_disable_fuse_mul_and_fc = True
    blob = conv.convert()
    open(a.out, "wb").write(blob)
    print(f"wrote {a.out}: {len(blob) / 1e6:.1f} MB, seq_len={T}, quant_params={bool(qp)}")


if __name__ == "__main__":
    main()
