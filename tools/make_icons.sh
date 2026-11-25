#!/usr/bin/env bash
# Regenerates the README logo and the adaptive launcher icon layers.
#   docs/logo.jpg          cropped from the master artwork art/swiperight.jpg
#   ic_launcher_fg.png     rendered from art/ic_launcher_fg.svg, a simplified
#                          redraw of the logo that stays legible at launcher size
#   ic_launcher_bg.png     a gradient in the logo's colours
# Needs ImageMagick (`convert`) and Inkscape.
set -euo pipefail
cd "$(dirname "$0")/.."

res=app/src/main/res/drawable-nodpi
mkdir -p "$res" docs

# README header: the rounded tile with its wordmark, trimmed of the dark
# backdrop. Crop box is in pixels of the 2752x1536 master.
convert art/swiperight.jpg -crop 1180x1180+786+176 +repage -resize 480x480 -quality 88 docs/logo.jpg

# Layers are 108dp at xxxhdpi (432px); the SVG is drawn at that size.
convert -size 432x432 xc: -sparse-color Bilinear \
    "0,0 rgb(70,48,170) 431,0 rgb(200,50,205) 0,431 rgb(36,14,96) 431,431 rgb(110,20,160)" \
    "$res/ic_launcher_bg.png"
inkscape art/ic_launcher_fg.svg -o "$res/ic_launcher_fg.png" 2>/dev/null

echo "wrote docs/logo.jpg, $res/ic_launcher_bg.png, $res/ic_launcher_fg.png"
