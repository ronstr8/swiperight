# swiperight

An Android keyboard (`InputMethodService`) with **on-device next-word prediction**:
distilgpt2 with 4-bit weights, running on TensorFlow Lite. Fully offline: the app does
not request the `INTERNET` permission.

- **Suggestions:** a scrolling strip of up to 9 words, best first. Three fill the
  width; swipe the strip for the rest.
- **Swipe typing:** drag across the letters. The decoder matches the shape against
  every whole-word token in the model's vocabulary and weighs each match by how
  likely the model thinks that word comes next. Shape-identical words ("to"/"too")
  are decided by context. The swiped word stays editable: tap an alternate to
  replace it, or press backspace to remove it.
- **"i" → "I"** when typed as a whole word ("i'm" → "I'm"), and in suggestions.

```
text before cursor ─► ContextBuilder ─► BpeTokenizer ─► TFLite Interpreter ─► WordCandidates ─► top-3 words
   (last 32 chars)    (prefix/slot)      (tokenizer.json)  (distilgpt2.tflite,     (whole words,
                                                             NNAPI or CPU)         casing, dedupe)
```

## Install on your phone (wireless debugging)

1. On the phone, open **Settings › Developer options › Wireless debugging** and turn it
   on. Tap **Pair device with pairing code**.
2. On this machine:
   ```bash
   adb pair <phone-ip>:<pairing-port>      # enter the 6-digit code
   adb connect <phone-ip>:<debug-port>     # the port shown on the Wireless debugging screen
   ./gradlew installDebug                  # or installRelease (minified, debug-signed)
   ```
   `adb` is at `~/Android/Sdk/platform-tools/adb` if it's not on your PATH.
3. Open **swiperight** on the phone, tap **Enable keyboard**, then **Switch keyboard**,
   and type in the test box.
4. Tap **Run on-device benchmark** for per-backend latency (AUTO / CPU / NNAPI, 1/2/4
   threads), memory, sample predictions and a full hash check of the assets.
5. Optional on-device tests: `./gradlew connectedDebugAndroidTest`.

## Layout

| Path | What |
|---|---|
| `app/src/main/java/com/swiperight/predict/Predictor.kt` | `Predictor` class: asset validation, mmap model, NNAPI/CPU selection, `predictNextWords(text): List<String>` |
| `predict/BpeTokenizer.kt`, `TokenizerLoader.kt` | Byte-level BPE from `tokenizer.json` (`vocab`, `merges`, `special_tokens`), no dependencies |
| `predict/ContextBuilder.kt` | 32-char window, partial-word prefix, space/newline/punctuation handling, edge cases |
| `predict/WordCandidates.kt` | Logits → whole-word suggestions |
| `predict/SwipeDecoder.kt` | Gesture decoding: path shape × language-model prior |
| `ime/SwipeRightIme.kt`, `ime/KeyboardView.kt` | The keyboard: threading, commit logic, privacy, memory release |
| `ui/MainActivity.kt` | Setup + on-device benchmark |
| `tools/build_model.sh` | Rebuilds all model assets from the Hugging Face checkpoint |
| `tools/convert/` | Graph builder, GPTQ/SmoothQuant calibration, int4 packing, verification |

## Using the predictor

```kotlin
// Background thread. Loading takes ~0.3–1.5 s; reuse the instance.
val predictor = Predictor.create(context)              // Backend.AUTO: NNAPI only if measurably faster
val words: List<String> = predictor.predictNextWords("I'm going to the ")  // e.g. [store, beach, gym]
predictor.close()
```

`predict(text)` also returns the parsed context: the typed prefix, and whether the IME
must insert a space before the word.

## How the model was built (and why it isn't a stock export)

`tools/build_model.sh` runs the whole pipeline and fails if accuracy drops below a gate.

1. **Graph**: the GPT-2 forward pass is written out by hand in TensorFlow from the HF
   safetensors (`build_tflite.py`). It has fixed shapes, `input_ids[1,16]` right-padded
   plus `last_index[1]`. Causal attention means padding can't leak into the last real
   position. The last position is selected in-graph, so the output is `[1, 50257]`
   (200 KB), not `[16, 50257]`.
2. **Naive int4 doesn't work.** Plain per-channel int4 on distilgpt2 agreed with the
   float model's top-1 only **~5%** of the time. Measured causes and fixes:
   - The tied LM head rows share a big mean vector that eats the int4 range. Fix:
     **mean-centre the head**. Softmax is shift-invariant, so this is exact.
   - A few hidden channels carry huge activations, and the hybrid kernel's dynamic int8
     activation quantization crushes everything else. Fix: **SmoothQuant** (α = 0.5),
     folded into the LayerNorms or applied as an explicit MUL.
   - The int4 rounding error compounds across layers. Fix: **GPTQ** (act-order,
     calibrated on `tools/convert/data/calibration.txt` plus sampled text).
   - The input embedding is a separate int4 table (2 codes per byte, per-row scales),
     unpacked in-graph with integer ops, because TFLite has no int4 GATHER.
3. **Packing**: ai-edge-quantizer emits the int4 tensor layout TFLite expects. The exact
   GPTQ codes are then written over its output and verified bit for bit
   (`quantize.py`).
4. **Verification** on the same TFLite 2.17 kernels the app uses (`verify_tflite.py`),
   on held-out text:

   | metric (vs float distilgpt2) | int4 shipped |
   |---|---|
   | float top-1 word is in our top-3 | **94.3%** (100% on the HF prompt set) |
   | top-1 agreement | 82.0% |
   | KL(float ‖ int4) | 0.133 |
   | file size | 61.5 MB (float: 325 MB) |

   Plain int8 dynamic quantization of the same model scores 74% top-1, KL 0.106. The
   tuned int4 model is about as accurate at half the size.

## Safety notes

**Model versioning**
- `assets/model_manifest.json` records the graph format version, model and tokenizer
  SHA-256 and sizes, `seq_len`, vocab size, BOS/PAD ids, and eval metrics.
  `Predictor.create` refuses to run (`ModelMismatchException`) if the tokenizer hash,
  any vocab size, the model's input/output shapes or the model file size disagree.
  `verifyModelHash = true` (used by the benchmark and device tests) also hashes the
  model. The keyboard keeps working without suggestions instead of showing wrong ones.
- Bump `FORMAT` in `write_manifest.py` and `ModelManifest.SUPPORTED_FORMAT` together
  whenever the graph's I/O contract changes.
- The runtime is pinned to `org.tensorflow:tensorflow-lite:2.17.0`, and the model is
  verified against `tensorflow-cpu==2.17.0`. **Do not upgrade either side alone.** I
  found that block-wise int4 runs without error on 2.17 but returns garbage. Runtime
  and model must be re-verified as a pair (`build_model.sh` + the on-device tests).
- 2.17.0 is the last `org.tensorflow` release and the last to ship the NNAPI delegate.
  LiteRT 2.x (`com.google.ai.edge.litert`) drops it from the Interpreter API. Moving to
  LiteRT would allow block-wise int4 (better accuracy) but means giving up NNAPI.

**Vocabulary sync**
- `BpeTokenizerTest.matchesHuggingFaceGolden` checks the Kotlin tokenizer against the
  HF reference ids for 34 tricky strings (unicode, emoji, contractions, whitespace runs,
  special tokens).
- `export_tokenizer.py` asserts every merge's parts and result are in the vocab and
  that ids are dense. At runtime, merges that point at missing tokens are counted and
  logged (`skippedMerges`).
- Typed `<|endoftext|>` is encoded as plain text, never as a control token.

**Memory limits**
- The model is memory-mapped from an uncompressed APK entry (`noCompress "tflite"`).
  Its 61 MB is clean page cache the OS can reclaim, not Java heap.
- XNNPACK repacks the FC weights into native memory (int4 stays int4). Expect roughly
  40–70 MB of native heap while loaded; the benchmark prints the exact figure. The
  activation arena is under 1 MB, because there are 16 positions and logits for one row.
- Java heap: about 2.5 MB for the word table and about 1.5 MB for the merge table
  (primitive arrays, not `HashMap<String,…>`). Parsing `tokenizer.json` peaks at about
  15 MB once, at load.
- The IME loads the model on first use and releases it after 5 minutes hidden, on
  `onTrimMemory` pressure, and never loads it on `isLowRamDevice()` phones.

**Privacy**
- No `INTERNET` permission. Typed text is never logged or stored.
- Predictions (and text reads) are off for password, email, URL, number, phone and
  `NO_SUGGESTIONS` fields.
- GPT-2 was trained on web text and can suggest offensive words. A shipping keyboard
  should add a blocklist in `WordCandidates`.

## Known limitations
- **One inference pass per keystroke.** That keeps latency flat, but a multi-token word
  can only be offered by its first piece. Between words, a fragment such as "un" can
  occasionally show up. A second, batched pass to finish words, or a dictionary filter,
  would fix this at about 2× the cost.
- Swipe and suggestions share one vocabulary: GPT-2's single-token words, about 30k
  distinct a–z words. Common words are covered; rarer words that GPT-2 splits into
  several tokens must be typed. At the start of a field, where context is weak,
  swipe alternates can include word fragments.
- Swipe costs one model run plus shape matching: 15–35 ms on a Snapdragon 8 Gen 3.
- English only (it's GPT-2). No autocorrect for tapped typing, no learning from the user.
- NNAPI: the hybrid int4 FC layers are generally not accepted by NNAPI drivers, so on
  most phones AUTO measures NNAPI and then chooses CPU (XNNPACK). NNAPI is also
  deprecated from Android 15. The choice is measured per device and cached.
- Model rebuilds need `uv` and about 2 GB of Python wheels. HF parity references need a
  system Python with `torch` and `transformers` (optional; skipped otherwise).
