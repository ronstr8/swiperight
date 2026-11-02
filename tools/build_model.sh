#!/usr/bin/env bash
# Rebuild the on-device model + tokenizer assets from scratch.
#
#   tools/build_model.sh            # full pipeline
#
# Produces app/src/main/assets/{distilgpt2.tflite,tokenizer.json,model_manifest.json}
# and app/src/test/resources/tokenizer_golden.json. Fails if the quantized model
# falls below the accuracy gate on held-out text.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"; CONV="$TOOLS/convert"
HF="$ROOT/build-cache/hf/distilgpt2"; OUT="$ROOT/build-cache/model"
ASSETS="$ROOT/app/src/main/assets"; TESTRES="$ROOT/app/src/test/resources"
SEQ_LEN=16
HF_REV=main   # pin to a commit hash for fully reproducible builds
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONWARNINGS=ignore

mkdir -p "$HF" "$OUT" "$ASSETS" "$TESTRES"

echo "== 1/7 python environments"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
[ -x "$TOOLS/.venv-tf/bin/python" ] || { uv venv -q -p 3.12 "$TOOLS/.venv-tf"; }
VIRTUAL_ENV="$TOOLS/.venv-tf" uv pip install -q -r "$TOOLS/requirements-tf.txt"
[ -x "$TOOLS/.venv-q/bin/python" ] || { uv venv -q -p 3.12 "$TOOLS/.venv-q"; }
VIRTUAL_ENV="$TOOLS/.venv-q" uv pip install -q -r "$TOOLS/requirements-quant.txt"
PYTF="$TOOLS/.venv-tf/bin/python"; PYQ="$TOOLS/.venv-q/bin/python"

echo "== 2/7 download distilgpt2 ($HF_REV)"
for f in config.json model.safetensors tokenizer.json vocab.json merges.txt; do
  [ -s "$HF/$f" ] || curl -fsSL -o "$HF/$f" "https://huggingface.co/distilbert/distilgpt2/resolve/$HF_REV/$f"
done

echo "== 3/7 golden reference (PyTorch, optional)"
if python3 -c "import torch, transformers" 2>/dev/null; then
  python3 "$CONV/make_reference.py" "$HF" "$OUT/reference.npz" "$OUT/tokenizer_golden.json"
  cp "$OUT/tokenizer_golden.json" "$TESTRES/tokenizer_golden.json"
else
  echo "   torch/transformers not found in system python; skipping HF parity reference"
fi

echo "== 4/7 calibrate: SmoothQuant + GPTQ int4"
(cd "$CONV" && "$PYTF" calibrate_gptq.py "$HF" data "$OUT" "$SEQ_LEN")

echo "== 5/7 build graph and quantize"
(cd "$CONV" && "$PYTF" build_tflite.py "$HF" "$OUT/prep.tflite" --seq-len "$SEQ_LEN" --quant-params "$OUT/quant_params.npz")
(cd "$CONV" && "$PYQ" quantize.py "$OUT/prep.tflite" "$OUT/quant_params.npz" "$OUT/distilgpt2_int4.tflite")

echo "== 6/7 verify on the TFLite 2.17 runtime (accuracy gate)"
(cd "$CONV" && "$PYTF" verify_tflite.py "$OUT/distilgpt2_int4.tflite" "$OUT/eval.npz" \
   --min-ref1-in-top3 0.90 --max-kl 0.25 --metrics-out "$OUT/metrics.json")
[ -f "$OUT/reference.npz" ] && (cd "$CONV" && "$PYTF" verify_tflite.py "$OUT/distilgpt2_int4.tflite" "$OUT/reference.npz")

echo "== 7/7 export assets"
cp "$OUT/distilgpt2_int4.tflite" "$ASSETS/distilgpt2.tflite"
python3 "$CONV/export_tokenizer.py" "$HF" "$ASSETS/tokenizer.json"
python3 "$CONV/write_manifest.py" "$ASSETS" "$OUT/metrics.json" "$SEQ_LEN"
ls -la "$ASSETS"
