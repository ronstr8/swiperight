#!/usr/bin/env bash
# Download the prebuilt model from the GitHub release that matches
# assets/model_manifest.json, and check it against the manifest's SHA-256.
#
#   tools/fetch_model.sh
#
# The model (61 MB) isn't committed. Its release tag is derived from the
# manifest's model_version, so a rebuilt model gets a new tag and an old
# checkout keeps fetching the model it was verified against.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="$ROOT/app/src/main/assets"
MANIFEST="$ASSETS/model_manifest.json"
REPO="${SWIPERIGHT_REPO:-ronstr8/swiperight}"

field() { python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print($1)" "$MANIFEST"; }
VERSION="$(field 'm["model_version"]')"
FILE="$(field 'm["model"]["file"]')"
WANT="$(field 'm["model"]["sha256"]')"
URL="https://github.com/$REPO/releases/download/model-$VERSION/$FILE"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

DEST="$ASSETS/$FILE"
if [ -f "$DEST" ] && [ "$(sha256 "$DEST")" = "$WANT" ]; then
  echo "$FILE is present and matches the manifest."; exit 0
fi

echo "Downloading $FILE ($VERSION)"
echo "  from $URL"
TMP="$DEST.part"
trap 'rm -f "$TMP"' EXIT
if ! curl -fL --progress-bar -o "$TMP" "$URL"; then
  echo "Download failed. Build the model locally instead: make model-build" >&2
  exit 1
fi
GOT="$(sha256 "$TMP")"
if [ "$GOT" != "$WANT" ]; then
  echo "SHA-256 mismatch: got $GOT, manifest says $WANT" >&2
  exit 1
fi
mv "$TMP" "$DEST"
echo "OK: $DEST"
