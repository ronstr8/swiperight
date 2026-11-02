"""Generate app/src/main/assets/emoji.txt from Unicode's emoji-test.txt.

Keeps fully-qualified emoji, drops skin-tone and hair-style variants and the
"Component" group. Output: one line per group, "Group name<TAB>e1 e2 e3 ...".
The app further hides emoji the device's font cannot render (Paint.hasGlyph).

Usage: python make_emoji.py <emoji-test.txt> <out emoji.txt>
  (emoji-test.txt: https://unicode.org/Public/emoji/15.1/emoji-test.txt)
"""
import sys

SKIP_CODEPOINTS = set(range(0x1F3FB, 0x1F400)) | set(range(0x1F9B0, 0x1F9B4))  # skin tones, hair

def main(src, out):
    groups, cur = {}, None
    for line in open(src, encoding="utf-8"):
        if line.startswith("# group:"):
            cur = line.split(":", 1)[1].strip()
            continue
        if not line.strip() or line.startswith("#") or cur == "Component":
            continue
        cps, rest = line.split(";", 1)
        if rest.split("#")[0].strip() != "fully-qualified":
            continue
        cp = [int(c, 16) for c in cps.split()]
        if any(c in SKIP_CODEPOINTS for c in cp) and len(cp) > 1:
            continue
        groups.setdefault(cur, []).append("".join(map(chr, cp)))
    with open(out, "w", encoding="utf-8") as f:
        for g, es in groups.items():
            f.write(g + "\t" + " ".join(es) + "\n")
    print({g: len(v) for g, v in groups.items()}, "total", sum(map(len, groups.values())))

if __name__ == "__main__":
    main(*sys.argv[1:3])
