"""Write assets/model_manifest.json: the contract between model, tokenizer and app.

The app refuses to start the predictor if any of these disagree with what it
finds at runtime (vocab size vs model output, seq_len vs model input, file
sizes, tokenizer hash). Bump FORMAT when the graph's I/O contract changes.
Usage: python write_manifest.py <assets_dir> <eval_metrics.json> <seq_len>
"""
import datetime, hashlib, json, os, sys

FORMAT = 1

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

assets, metrics_path, seq_len = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = json.load(open(f"{assets}/tokenizer.json", encoding="utf-8"))
m = f"{assets}/distilgpt2.tflite"; t = f"{assets}/tokenizer.json"
manifest = {
    "format": FORMAT,
    "model_version": "distilgpt2-int4-gptq-sq-v1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": {
        "file": "distilgpt2.tflite", "bytes": os.path.getsize(m), "sha256": sha(m),
        "seq_len": seq_len, "vocab_size": len(tok["vocab"]),
        "inputs": ["input_ids:int32[1,%d]" % seq_len, "last_index:int32[1]"],
        "outputs": ["logits:float32[1,%d]" % len(tok["vocab"])],
        "quantization": "INT4 per-output-channel weights (GPTQ + SmoothQuant a=0.5), "
                        "dynamic int8 activations; int4 per-row embedding table",
        "runtime": "org.tensorflow:tensorflow-lite:2.17.0",
        "source": "distilbert/distilgpt2 (Apache-2.0)",
    },
    "tokenizer": {
        "file": "tokenizer.json", "bytes": os.path.getsize(t), "sha256": sha(t),
        "vocab_size": len(tok["vocab"]), "merges": len(tok["merges"]),
        "bos_id": tok["vocab"]["<|endoftext|>"], "pad_id": tok["vocab"]["<|endoftext|>"],
    },
    "eval": json.load(open(metrics_path)),
}
json.dump(manifest, open(f"{assets}/model_manifest.json", "w"), indent=2)
print(json.dumps(manifest["model"], indent=1)[:400])
