package com.swiperight.predict

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * On-device next-word predictor: distilgpt2 (INT4 weights) on TensorFlow Lite.
 *
 * - Fully offline. The model is memory-mapped from the APK, so its 61 MB are
 *   clean, reclaimable page cache rather than Java heap.
 * - Fixed shapes: input_ids int32[1, seqLen] right-padded plus last_index
 *   int32[1], output float32[1, vocab]. Every call does the same work, and
 *   all I/O buffers are allocated once.
 * - Backend: NNAPI if it is available and measurably faster, otherwise CPU
 *   (XNNPACK). With this model's hybrid INT4 layers NNAPI usually cannot take
 *   the heavy ops, so expect CPU; the choice is measured, not assumed. It is
 *   cached per (model hash, device build) in SharedPreferences.
 *
 * Thread-safety: all public methods are synchronized; inference is serialized.
 * Call from a background thread. Loading takes 0.3-1.5 s, inference tens of ms.
 */
class Predictor private constructor(
    private val manifest: ModelManifest,
    private val tokenizer: BpeTokenizer,
    private val candidates: WordCandidates,
    private var interpreter: Interpreter?,
    private var delegate: NnApiDelegate?,
    private val modelBuffer: MappedByteBuffer,
    val info: Info,
) : Closeable {

    enum class Backend { AUTO, CPU, NNAPI }

    data class Info(
        val backend: Backend,
        val numThreads: Int,
        val loadMillis: Long,
        val modelVersion: String,
        val backendTimingsMs: Map<Backend, Double>,
    )

    data class Prediction(
        val words: List<String>,
        val context: PredictionContext,
        val tokenCount: Int,
        val inferenceMillis: Double,
    )

    private val seqLen = manifest.seqLen
    private val vocab = manifest.modelVocabSize
    private val idsIndex: Int
    private val lastIndex: Int
    private val idsBuf: ByteBuffer = ByteBuffer.allocateDirect(4 * seqLen).order(ByteOrder.nativeOrder())
    private val lastBuf: ByteBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
    private val outBuf: ByteBuffer = ByteBuffer.allocateDirect(4 * vocab).order(ByteOrder.nativeOrder())
    private val logits = FloatArray(vocab)
    private val inputs = arrayOfNulls<Any>(2)
    private val outputs = HashMap<Int, Any>(1)

    init {
        val itp = requireNotNull(interpreter)
        val (ids, last) = resolveInputs(itp)
        idsIndex = ids
        lastIndex = last
        outputs[0] = outBuf
    }

    /** The spec's helper: top-3 next-word suggestions for the text before the cursor. */
    fun predictNextWords(text: String): List<String> = predict(text).words

    @Synchronized
    fun predict(textBeforeCursor: CharSequence, k: Int = 3): Prediction {
        val ctx = ContextBuilder.build(textBeforeCursor)
        if (ctx.suppress) return Prediction(emptyList(), ctx, 0, 0.0)
        val tokens = contextTokens(ctx)
        var ms = infer(tokens, logits)
        val words = ranker.rank(logits, ctx, k) { baseId -> ms += infer(tokens + baseId, logits2); logits2 }
        return Prediction(words, ctx, tokens.size, ms)
    }

    /**
     * Decodes a swipe into up to [k] words, best first, as the *next* word
     * after [textBeforeCursor]. If the text ends mid-word or right after
     * punctuation, the caller inserts the separating space on commit; the
     * context is built as if that space were already there.
     */
    @Synchronized
    fun decodeSwipe(textBeforeCursor: CharSequence, swipe: SwipeInput, k: Int = 9): Prediction {
        val text = textBeforeCursor.toString()
        val needsSpace = text.isNotEmpty() && ContextBuilder.build(text).let { it.prefix.isNotEmpty() || it.insertSpace }
        val ctx = ContextBuilder.build(if (needsSpace) "$text " else text).copy(suppress = false, prefix = "")
        val tokens = contextTokens(ctx)
        val t0 = SystemClock.elapsedRealtimeNanos()
        infer(tokens, logits)
        val words = swipeDecoder.decode(swipe, logits, ctx.spaceBefore, k) { baseId -> infer(tokens + baseId, logits2); logits2 }
        val totalMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        return Prediction(words, ctx.copy(insertSpace = needsSpace), tokens.size, totalMs)
    }

    // Built at load (background thread), not on the first swipe: ~0.3 s on device.
    private val contractions = Contractions(tokenizer)
    private val ranker = SuggestionRanker(candidates, contractions)
    private val swipeDecoder = SwipeDecoder(candidates, contractions).also { it.warmUp(vocab) }
    /** Second-pass logits (contraction suffixes), so the first pass stays intact. */
    private val logits2 = FloatArray(vocab)

    /** Runs the model on [tokens] (most recent [seqLen] kept) into [dest]. Returns inference ms. */
    private fun infer(tokens: IntArray, dest: FloatArray): Double {
        val itp = interpreter ?: throw IllegalStateException("Predictor is closed")
        val t = if (tokens.size <= seqLen) tokens else tokens.copyOfRange(tokens.size - seqLen, tokens.size)
        idsBuf.clear()
        for (i in 0 until seqLen) idsBuf.putInt(if (i < t.size) t[i] else manifest.padId)
        idsBuf.rewind()
        lastBuf.clear()
        lastBuf.putInt(t.size - 1).rewind()
        outBuf.clear()
        inputs[idsIndex] = idsBuf
        inputs[lastIndex] = lastBuf

        val t0 = SystemClock.elapsedRealtimeNanos()
        itp.runForMultipleInputsOutputs(inputs, outputs)
        val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
        outBuf.rewind()
        outBuf.asFloatBuffer().get(dest)
        return ms
    }

    /** BOS (at document start) + context tokens, keeping the most recent [seqLen]. */
    internal fun contextTokens(ctx: PredictionContext): IntArray {
        val ids = tokenizer.encode(ctx.context)
        val withBos = if (ctx.docStart || ids.isEmpty()) intArrayOf(manifest.bosId) + ids else ids
        return if (withBos.size <= seqLen) withBos else withBos.copyOfRange(withBos.size - seqLen, withBos.size)
    }

    @Synchronized
    override fun close() {
        interpreter?.close()
        interpreter = null
        delegate?.close()
        delegate = null
    }

    val isClosed: Boolean @Synchronized get() = interpreter == null

    companion object {
        private const val TAG = "Predictor"
        const val MANIFEST_ASSET = "model_manifest.json"
        private const val PREFS = "predictor"

        /**
         * Loads and validates the assets and builds an interpreter.
         *
         * @param verifyModelHash also SHA-256 the 61 MB model (~0.2-0.5 s). The
         *   size and tokenizer-hash checks always run. Use this in diagnostics,
         *   not on the typing path.
         * @throws ModelMismatchException if model, tokenizer and manifest disagree.
         */
        fun create(
            context: Context,
            backend: Backend = Backend.AUTO,
            numThreads: Int = defaultThreads(),
            verifyModelHash: Boolean = false,
        ): Predictor {
            val t0 = SystemClock.elapsedRealtime()
            val assets = context.assets
            val manifest = ModelManifest.parse(assets.open(MANIFEST_ASSET).bufferedReader().use { it.readText() })
            check(manifest.format == ModelManifest.SUPPORTED_FORMAT) {
                "manifest format ${manifest.format}, app supports ${ModelManifest.SUPPORTED_FORMAT}"
            }

            // Tokenizer: hash-checked against the manifest, then parsed.
            val tokBytes = assets.open(manifest.tokenizerFile).use { it.readBytes() }
            mismatchIf(tokBytes.size.toLong() != manifest.tokenizerBytes) { "tokenizer size ${tokBytes.size} != manifest ${manifest.tokenizerBytes}" }
            mismatchIf(sha256(tokBytes) != manifest.tokenizerSha256) { "tokenizer.json hash does not match manifest" }
            val tokenizer = TokenizerLoader.parse(String(tokBytes, Charsets.UTF_8))
            mismatchIf(tokenizer.vocabSize != manifest.tokenizerVocabSize || tokenizer.vocabSize != manifest.modelVocabSize) {
                "vocab sizes differ: tokenizer=${tokenizer.vocabSize} manifest.tokenizer=${manifest.tokenizerVocabSize} " +
                    "manifest.model=${manifest.modelVocabSize}"
            }
            if (tokenizer.skippedMerges > 0) Log.w(TAG, "${tokenizer.skippedMerges} merges reference tokens not in vocab")

            // Model: memory-mapped straight from the (uncompressed) APK entry.
            val fd = try {
                assets.openFd(manifest.modelFile)
            } catch (e: java.io.FileNotFoundException) {
                throw ModelMismatchException("${manifest.modelFile} missing or compressed in APK (needs noCompress): ${e.message}")
            }
            val modelBuffer = fd.use {
                mismatchIf(it.declaredLength != manifest.modelBytes) { "model size ${it.declaredLength} != manifest ${manifest.modelBytes}" }
                FileInputStream(it.fileDescriptor).channel.use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)
                }
            }
            if (verifyModelHash) {
                mismatchIf(sha256(modelBuffer.duplicate()) != manifest.modelSha256) { "model hash does not match manifest" }
            }

            val candidates = WordCandidates(tokenizer)
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val cacheKey = "backend:${manifest.modelSha256.take(16)}:${Build.FINGERPRINT}:$numThreads"

            val timings = LinkedHashMap<Backend, Double>()
            val wanted = when {
                backend != Backend.AUTO -> backend
                else -> prefs.getString(cacheKey, null)?.let { runCatching { Backend.valueOf(it) }.getOrNull() } ?: Backend.AUTO
            }

            var chosen: Built? = null
            if (wanted == Backend.NNAPI || wanted == Backend.AUTO) {
                chosen = tryBuild(modelBuffer, manifest, Backend.NNAPI, numThreads)
                if (chosen != null && wanted == Backend.AUTO) timings[Backend.NNAPI] = chosen.benchmark(manifest)
            }
            if (wanted == Backend.CPU || wanted == Backend.AUTO || chosen == null) {
                if (wanted == Backend.AUTO && chosen != null) {
                    val cpu = tryBuild(modelBuffer, manifest, Backend.CPU, numThreads)
                        ?: throw IllegalStateException("CPU interpreter failed to initialize")
                    timings[Backend.CPU] = cpu.benchmark(manifest)
                    // Prefer CPU unless NNAPI wins clearly; partitioned graphs are fragile.
                    if (timings.getValue(Backend.CPU) <= timings.getValue(Backend.NNAPI) * 1.15) {
                        chosen.close(); chosen = cpu
                    } else {
                        cpu.close()
                    }
                } else {
                    chosen?.close()
                    chosen = tryBuild(modelBuffer, manifest, Backend.CPU, numThreads)
                        ?: throw IllegalStateException("CPU interpreter failed to initialize")
                    if (wanted == Backend.AUTO) timings[Backend.CPU] = chosen.benchmark(manifest)
                }
                if (wanted == Backend.AUTO) prefs.edit().putString(cacheKey, chosen.backend.name).apply()
            }
            val built = chosen!!
            val loadMs = SystemClock.elapsedRealtime() - t0
            Log.i(TAG, "loaded ${manifest.modelVersion} backend=${built.backend} threads=$numThreads in ${loadMs}ms $timings")
            return Predictor(
                manifest, tokenizer, candidates, built.interpreter, built.delegate, modelBuffer,
                Info(built.backend, numThreads, loadMs, manifest.modelVersion, timings),
            )
        }

        /**
         * Two threads. Mid-tier SoCs (Pixel 4a: 2x A76 + 6x A55) have two big
         * cores; spreading XNNPACK onto little cores usually adds latency.
         */
        fun defaultThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 2)

        private class Built(val interpreter: Interpreter, val delegate: NnApiDelegate?, val backend: Backend) {
            fun close() {
                interpreter.close(); delegate?.close()
            }

            /** Median of a few warm runs on a typical-length input. */
            fun benchmark(m: ModelManifest): Double {
                val ids = ByteBuffer.allocateDirect(4 * m.seqLen).order(ByteOrder.nativeOrder())
                for (i in 0 until m.seqLen) ids.putInt(if (i < 9) 464 + i else m.padId)
                ids.rewind()
                val last = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).putInt(8)
                last.rewind()
                val out = ByteBuffer.allocateDirect(4 * m.modelVocabSize).order(ByteOrder.nativeOrder())
                val (ii, li) = resolveInputs(interpreter)
                val ins = arrayOfNulls<Any>(2).also { it[ii] = ids; it[li] = last }
                val times = DoubleArray(7)
                repeat(2) { interpreter.runForMultipleInputsOutputs(ins, mapOf(0 to out)); out.clear() }
                for (r in times.indices) {
                    val t = SystemClock.elapsedRealtimeNanos()
                    interpreter.runForMultipleInputsOutputs(ins, mapOf(0 to out)); out.clear()
                    times[r] = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
                }
                times.sort()
                return times[times.size / 2]
            }
        }

        private fun tryBuild(model: MappedByteBuffer, m: ModelManifest, backend: Backend, threads: Int): Built? {
            var delegate: NnApiDelegate? = null
            return try {
                val opts = Interpreter.Options().setNumThreads(threads)
                if (backend == Backend.NNAPI) {
                    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) return null
                    delegate = NnApiDelegate(
                        NnApiDelegate.Options()
                            .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                            .setAllowFp16(true)
                            // The NNAPI reference CPU driver is far slower than XNNPACK.
                            .setUseNnapiCpu(false),
                    )
                    opts.addDelegate(delegate)
                }
                val itp = Interpreter(model, opts)
                validateShapes(itp, m)
                Built(itp, delegate, backend)
            } catch (e: ModelMismatchException) {
                delegate?.close()
                throw e
            } catch (e: Throwable) {
                // NNAPI unavailable, delegate rejected the graph, OOM, etc.
                Log.w(TAG, "backend $backend unavailable: ${e.message}")
                delegate?.close()
                if (backend == Backend.CPU) throw e
                null
            }
        }

        private fun resolveInputs(itp: Interpreter): Pair<Int, Int> {
            var ids = -1
            var last = -1
            for (i in 0 until itp.inputTensorCount) {
                val name = itp.getInputTensor(i).name()
                when {
                    "input_ids" in name -> ids = i
                    "last_index" in name -> last = i
                }
            }
            if (ids < 0 || last < 0) throw ModelMismatchException("model inputs must be input_ids and last_index")
            return ids to last
        }

        private fun validateShapes(itp: Interpreter, m: ModelManifest) {
            val (ii, li) = resolveInputs(itp)
            val idsT = itp.getInputTensor(ii)
            val lastT = itp.getInputTensor(li)
            val outT = itp.getOutputTensor(0)
            mismatchIf(!idsT.shape().contentEquals(intArrayOf(1, m.seqLen)) || idsT.dataType() != DataType.INT32) {
                "input_ids is ${idsT.shape().contentToString()} ${idsT.dataType()}, manifest says [1, ${m.seqLen}] INT32"
            }
            mismatchIf(!lastT.shape().contentEquals(intArrayOf(1)) || lastT.dataType() != DataType.INT32) {
                "last_index is ${lastT.shape().contentToString()} ${lastT.dataType()}"
            }
            mismatchIf(!outT.shape().contentEquals(intArrayOf(1, m.modelVocabSize)) || outT.dataType() != DataType.FLOAT32) {
                "logits are ${outT.shape().contentToString()} ${outT.dataType()}, tokenizer vocab is ${m.modelVocabSize}"
            }
        }

        private inline fun mismatchIf(cond: Boolean, msg: () -> String) {
            if (cond) throw ModelMismatchException(msg())
        }

        private fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

        private fun sha256(buf: ByteBuffer): String {
            val md = MessageDigest.getInstance("SHA-256")
            md.update(buf)
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
