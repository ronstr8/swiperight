"""Export GPT-2's byte-level BPE into the app's tokenizer.json schema.

Schema (consumed by BpeTokenizer.kt):
  {"vocab": {token: id, ...},             # byte-level unicode token strings
   "merges": [["a", "b"], ...],           # rank = list index
   "special_tokens": ["<|endoftext|>"]}

Usage: python export_tokenizer.py <hf_model_dir> <out_tokenizer.json>
"""
import json
import sys


def main(model_dir, out):
    vocab = json.load(open(f"{model_dir}/vocab.json", encoding="utf-8"))
    merges = []
    for line in open(f"{model_dir}/merges.txt", encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#version"):
            continue
        a, b = line.split(" ")
        merges.append([a, b])
    for a, b in merges:  # vocab sync: every merge component and result must exist
        assert a in vocab and b in vocab and a + b in vocab, (a, b)
    ids = sorted(vocab.values())
    assert ids == list(range(len(ids))), "vocab ids must be dense 0..V-1"
    doc = {"vocab": vocab, "merges": merges, "special_tokens": ["<|endoftext|>"]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {out}: vocab={len(vocab)} merges={len(merges)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
