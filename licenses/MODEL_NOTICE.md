# Model notice

`distilgpt2.tflite` (downloaded by `make model` from this repo's GitHub releases) and
`app/src/main/assets/tokenizer.json` are derived from
[distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2) by Hugging Face,
licensed under the Apache License, Version 2.0 (`APACHE-2.0.txt` in this folder).
Upstream ships no NOTICE file.

## Changes from the original

- Weights converted from safetensors to a hand-built TensorFlow Lite graph with fixed
  input shape `[1,16]` that returns logits for the last real position only.
- LM head mean-centred (exact under softmax).
- SmoothQuant (α = 0.5) scaling folded into the weights.
- Weights quantized to int4 with GPTQ; the input embedding is stored as a packed int4
  table with per-row scales.
- Tokenizer re-exported to a compact `tokenizer.json` (vocab, merges, special tokens).

The pipeline is `tools/build_model.sh`; `assets/model_manifest.json` records the hashes.

## Limitations

Like GPT-2, distilgpt2 was trained on web text and reflects its biases; it can suggest
offensive or stereotyped words. See the upstream model card's "Limitations and Risks".
