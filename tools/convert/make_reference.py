"""Produce golden token ids + next-token logits from Hugging Face (PyTorch).

Run with an interpreter that has torch + transformers. Output is consumed by
verify_tflite.py (model parity) and the Kotlin tokenizer unit test (id parity).
Usage: python make_reference.py <hf_model_dir> <out.npz> <out_tokenizer_golden.json>
"""
import json, os, sys
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROMPTS = [
    "I'm going to the", "Thank you so much for", "Can we meet at the",
    "The weather today is", "Let me know if you", "Happy birthday! Hope you have a",
    "I'll be there in 10", "What time does the movie", "She said that he would",
    "Sorry, I can't make it", "Do you want to grab", "The quick brown fox jumps over the",
    "hello", "See you", "I love", "We should probably", "It's raining cats and",
    "Our meeting has been moved to", "Please call me when you", "This is the best",
]
TOKENIZER_CASES = PROMPTS + [
    "", " ", "  leading spaces", "trailing space ", "tabs\tand\nnewlines\n\n",
    "don't won't they're I've we'll she'd it's", "numbers 12345 and 3.14159",
    "émojis 🎉🔥 and ünïcödé — dashes", "日本語のテキスト", "URL https://example.com/a?b=c",
    "MiXeD CaSe WoRdS", "punctuation!!! ??? ...", "<|endoftext|>", "a  b   c    d",
    "no\u00a0break\u3000ideographic\u2003em\u2028line", "trail\u00a0 \u00a0", "x\u0085y\x0bz\x0cw",
]

def main(model_dir, out_npz, out_json):
    tok = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir).eval()
    ids, logits = [], []
    for p in PROMPTS:
        x = tok(p)["input_ids"][-16:]
        with torch.no_grad():
            out = model(torch.tensor([x])).logits[0, -1].numpy()
        ids.append(x); logits.append(out)
    lens = np.array([len(x) for x in ids])
    padded = np.zeros((len(ids), 16), np.int32)
    for i, x in enumerate(ids): padded[i, :len(x)] = x
    np.savez(out_npz, ids=padded, lens=lens, logits=np.stack(logits), prompts=np.array(PROMPTS))
    golden = [{"text": t,
               # special-token strings typed by a user are encoded as plain text
               "ids": tok(t, add_special_tokens=False, split_special_tokens=True)["input_ids"],
               "ids_with_special": tok(t, add_special_tokens=False)["input_ids"]}
              for t in TOKENIZER_CASES]
    json.dump(golden, open(out_json, "w"), ensure_ascii=False, indent=1)
    print(f"{len(PROMPTS)} prompts, {len(golden)} tokenizer cases")

if __name__ == "__main__":
    main(*sys.argv[1:4])
