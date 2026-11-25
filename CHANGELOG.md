# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- New launcher icon: a glowing keyboard with a swipe arrow and sparkles, after the project logo.

## [0.1.0] - 2026-11-17

First release.

### Added
- Android keyboard with on-device next-word prediction: distilgpt2 quantized to
  int4 (GPTQ + SmoothQuant), running on TensorFlow Lite with no network access.
- Scrolling suggestion strip of up to 9 words, best first.
- Swipe typing scored by path shape and the model's prediction, with a live trail,
  the key under the finger lit up, and the word forming as you swipe.
- Contractions (it's, don't, I'm, ...) suggested and swipeable, with or without
  the apostrophe; "i" becomes "I".
- Punctuation panel on a long press of the period key, and an emoji picker with
  recents and the Unicode 15.1 set.
- Setup screen with an on-device benchmark and an asset hash check.
- `make run` installs the keyboard on a connected phone from a fresh clone: it
  downloads the prebuilt model (hash-checked against the manifest), builds,
  installs and opens the setup screen. `make` lists the other targets.
- Licensed under Apache-2.0 (`LICENSE`); model license and notice in `licenses/`.

[Unreleased]: https://github.com/ronstr8/swiperight/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ronstr8/swiperight/releases/tag/v0.1.0
