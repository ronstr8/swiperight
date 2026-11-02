"""Turn the int4-ready float graph into the shipped INT4 .tflite.

Steps:
  1. ai-edge-quantizer applies dynamic-range, symmetric, per-output-channel
     INT4 to every FULLY_CONNECTED weight. This produces the tensor layout,
     packing and op versions that the TFLite 2.17 hybrid FC kernel expects.
  2. Each INT4 weight tensor is then overwritten with the GPTQ codes and
     scales from calibrate_gptq.py. The quantizer's own min/max rounding of an
     already-quantized tensor is not guaranteed to round-trip (after GPTQ
     error feedback a channel's largest code is not always +/-7), so its
     result is replaced rather than trusted.
  3. Every patched tensor is unpacked and checked to equal codes * scale
     exactly; a missing or unmatched FC site is a hard error.

FC sites are matched by weight content: build_tflite.py writes each weight as
float32(codes) * scale, so the float tensor bytes identify the site.

Usage: python quantize.py <prep.tflite> <quant_params.npz> <out.tflite>
"""
import hashlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
from ai_edge_litert.tools import flatbuffer_utils as fbu  # noqa: E402
from ai_edge_quantizer import quantizer  # noqa: E402
from ai_edge_quantizer.transformations.transformation_utils import pack_data  # noqa: E402

TFL_FULLY_CONNECTED = 9
TFL_INT4 = 17


def fc_weight_tensors(model):
    sg = model.subgraphs[0]
    for op in sg.operators:
        code = model.operatorCodes[op.opcodeIndex]
        if max(code.builtinCode, code.deprecatedBuiltinCode) == TFL_FULLY_CONNECTED:
            yield sg.tensors[op.inputs[1]]


def tensor_bytes(model, t):
    return bytes(np.asarray(model.buffers[t.buffer].data, dtype=np.uint8))


def unpack_int4(data, n):
    b = np.asarray(data, dtype=np.uint8)
    lo = (b & 0x0F).astype(np.int8)
    hi = (b >> 4).astype(np.int8)
    v = np.stack([lo, hi], axis=1).reshape(-1)[:n]
    return np.where(v > 7, v - 16, v).astype(np.int8)


def main(prep_path, params_path, out_path):
    qp = dict(np.load(params_path))
    sites = sorted({k.rsplit("/", 1)[0] for k in qp if not k.startswith("embed/")})

    # site lookup by the exact float32 bytes build_tflite.py wrote ([out, in]).
    by_hash = {}
    for s in sites:
        W = qp[f"{s}/codes"].astype(np.float32) * qp[f"{s}/scale"]  # [in, out]
        by_hash[hashlib.sha256(np.ascontiguousarray(W.T).tobytes()).hexdigest()] = s

    prep = fbu.read_model(prep_path)
    name_to_site = {}
    for t in fc_weight_tensors(prep):
        h = hashlib.sha256(tensor_bytes(prep, t)).hexdigest()
        if h not in by_hash:
            raise SystemExit(f"FC weight {t.name!r} does not match any GPTQ site "
                             "(did the converter fold an op into the weights?)")
        name_to_site[t.name] = by_hash[h]
    missing = set(sites) - set(name_to_site.values())
    if missing:
        raise SystemExit(f"GPTQ sites missing from graph: {sorted(missing)}")

    recipe = [{
        "regex": ".*", "operation": "FULLY_CONNECTED",
        "algorithm_key": "min_max_uniform_quantize",
        "op_config": {
            "weight_tensor_config": {"num_bits": 4, "symmetric": True,
                                     "granularity": "CHANNELWISE", "dtype": "INT"},
            "compute_precision": "INTEGER", "explicit_dequantize": False,
        },
    }]
    q = quantizer.Quantizer(prep_path)
    q.load_quantization_recipe(recipe)
    qblob = bytes(q.quantize().quantized_model)
    model = fbu.convert_bytearray_to_object(qblob)
    # The quantizer may append large buffers after the flatbuffer and reference
    # them by (offset, size). Inline them so they can be edited; the model is
    # far below the 2 GB flatbuffer limit.
    for b in model.buffers:
        if b.data is None and getattr(b, "offset", 0) > 1:
            b.data = np.frombuffer(qblob, np.uint8, count=b.size, offset=b.offset).copy()
            b.offset, b.size = 0, 0

    patched, changed = 0, 0
    for t in fc_weight_tensors(model):
        site = name_to_site.get(t.name.decode() if isinstance(t.name, bytes) else t.name)
        if site is None:
            site = name_to_site.get(t.name)
        if site is None:
            raise SystemExit(f"quantized FC weight {t.name!r} has no GPTQ site")
        if t.type != TFL_INT4:
            raise SystemExit(f"{site}: expected INT4 tensor, got type {t.type}")
        codes = np.ascontiguousarray(qp[f"{site}/codes"].T)  # [out, in]
        scale = qp[f"{site}/scale"].astype(np.float32)
        if tuple(t.shape) != codes.shape:
            raise SystemExit(f"{site}: shape {tuple(t.shape)} != {codes.shape}")
        old = unpack_int4(model.buffers[t.buffer].data, codes.size)
        changed += int((old != codes.reshape(-1)).sum())
        model.buffers[t.buffer].data = np.asarray(pack_data(4, codes), dtype=np.uint8)
        qz = t.quantization
        qz.scale = scale.tolist()
        qz.zeroPoint = [0] * len(scale)
        qz.quantizedDimension = 0
        back = unpack_int4(model.buffers[t.buffer].data, codes.size).reshape(codes.shape)
        assert np.array_equal(back, codes), site
        patched += 1

    if patched != len(sites):
        raise SystemExit(f"patched {patched} FC weights, expected {len(sites)}")
    blob = fbu.convert_object_to_bytearray(model)
    open(out_path, "wb").write(blob)
    print(f"wrote {out_path}: {len(blob) / 1e6:.1f} MB; patched {patched} INT4 FC weights "
          f"({changed} codes differed from plain min/max requantization)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
