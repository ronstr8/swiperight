package com.swiperight.predict

import java.util.Locale

/**
 * Turns next-token logits into whole-word suggestions.
 *
 * GPT-2 predicts subword tokens. A token is offered as a word only if, once
 * decoded, it is made of letters (one optional inner apostrophe allowed), and
 * its leading-space flag fits the slot: "Ġthe" after a space, "the" after a
 * newline, an opening bracket, or the start of the field.
 *
 * A single inference pass decides everything, which keeps latency
 * predictable. The cost: a word GPT-2 splits into several tokens can only be
 * suggested by its first piece. Mid-word, its first piece only matches if it
 * extends the typed prefix, so such words simply don't show up. Between words,
 * a piece like "Ġun" can occasionally surface. See README "Known limitations".
 *
 * The per-token classification is computed once (about 2.5 MB for 50k tokens)
 * and is read-only, so it can be shared across threads.
 */
class WordCandidates(private val tokenizer: BpeTokenizer) {

    private val size = tokenizer.vocabSize
    internal val vocabSize: Int get() = size

    /** Decoded word for each token id (without its leading space), or null if not word-like. */
    private val wordOf = arrayOfNulls<String>(size)
    private val lowerOf = arrayOfNulls<String>(size)
    private val leadingSpace = BooleanArray(size)

    init {
        for (id in 0 until size) {
            if (tokenizer.isSpecial(id)) continue
            val s = tokenizer.decodeTokenStrict(id) ?: continue
            val space = s.startsWith(' ')
            val word = if (space) s.substring(1) else s
            if (!isWordLike(word)) continue
            wordOf[id] = word
            lowerOf[id] = word.lowercase(Locale.ROOT)
            leadingSpace[id] = space
        }
    }

    internal fun wordAt(id: Int): String? = wordOf[id]
    internal fun lowerAt(id: Int): String? = lowerOf[id]
    internal fun hasLeadingSpace(id: Int): Boolean = leadingSpace[id]

    /**
     * Top [k] distinct words (case-insensitive) for [ctx], best first.
     * [logits] has one entry per vocab id. It is read, never modified.
     */
    fun select(logits: FloatArray, ctx: PredictionContext, k: Int = 3): List<String> =
        selectScored(logits, ctx, k).map { format(it.first, ctx.prefix) }

    /**
     * Top [k] distinct words (case-insensitive) with their raw logits, best
     * first. Words are in the model's casing; see [format].
     */
    fun selectScored(logits: FloatArray, ctx: PredictionContext, k: Int): List<Pair<String, Float>> {
        require(logits.size >= size) { "logits (${logits.size}) shorter than vocab ($size)" }
        if (ctx.suppress) return emptyList()
        val prefixLower = ctx.prefix.lowercase(Locale.ROOT)
        val pool = maxOf(POOL, k * 3)

        val poolIds = IntArray(pool) { -1 }
        val poolScores = FloatArray(pool) { Float.NEGATIVE_INFINITY }
        var minIdx = 0
        for (id in 0 until size) {
            val lw = lowerOf[id] ?: continue
            if (leadingSpace[id] != ctx.spaceBefore) continue
            if (prefixLower.isNotEmpty()) {
                if (!lw.startsWith(prefixLower)) continue
            } else if (lw.length == 1 && lw != "a" && lw != "i") {
                continue // stray single letters are almost never the next word
            }
            val sc = logits[id]
            if (sc.isNaN() || sc <= poolScores[minIdx]) continue
            poolIds[minIdx] = id
            poolScores[minIdx] = sc
            minIdx = 0
            for (j in 1 until pool) if (poolScores[j] < poolScores[minIdx]) minIdx = j
        }

        val order = (0 until pool).filter { poolIds[it] >= 0 }.sortedByDescending { poolScores[it] }
        val seen = HashSet<String>()
        val out = ArrayList<Pair<String, Float>>(k)
        for (j in order) {
            val id = poolIds[j]
            if (!seen.add(lowerOf[id]!!)) continue
            out += wordOf[id]!! to poolScores[j]
            if (out.size == k) break
        }
        return out
    }

    companion object {
        /** Candidate pool kept during the vocab scan; headroom for case dedupe of up to 9 results. */
        private const val POOL = 32

        /** Final display form: typed casing, then the "I" rule. */
        fun format(word: String, prefix: String): String = fixStandaloneI(applyCase(word, prefix))

        /** English capitalizes the pronoun: "i" is always "I". */
        internal fun fixStandaloneI(w: String): String = if (w == "i") "I" else w

        internal fun isWordLike(w: String): Boolean {
            if (w.isEmpty() || w.length > 30) return false
            var apostrophes = 0
            for ((i, c) in w.withIndex()) {
                when {
                    c.isLetter() -> {}
                    (c == '\'' || c == '’') && i > 0 && i < w.length - 1 -> apostrophes++
                    else -> return false
                }
            }
            return apostrophes <= 1
        }

        /**
         * Typed capitals win: "Ho" + "home" -> "Home", "HO" -> "HOME". With a
         * lowercase prefix the model's casing is kept, so "lon" can still
         * become "London".
         */
        internal fun applyCase(word: String, prefix: String): String {
            if (prefix.isEmpty()) return word
            if (prefix.length > 1 && prefix.all { it.isUpperCase() }) return word.uppercase(Locale.ROOT)
            if (prefix[0].isUpperCase()) return word.replaceFirstChar { it.titlecase(Locale.ROOT) }
            return word
        }
    }
}
