package com.swiperight.predict

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * A finger path across the letter keys, in any pixel space. [keyX]/[keyY] are
 * the centres of the a..z keys (index = letter - 'a', NaN if absent), and
 * [keyWidth] is the width of one key in the same units.
 */
class SwipeInput(
    val xs: FloatArray,
    val ys: FloatArray,
    val keyX: FloatArray,
    val keyY: FloatArray,
    val keyWidth: Float,
) {
    init {
        require(xs.size == ys.size && xs.isNotEmpty()) { "empty or ragged path" }
        require(keyX.size == 26 && keyY.size == 26) { "need 26 key centres" }
        require(keyWidth > 0f)
    }
}

/**
 * Gesture ("swipe") typing decoder: shape matching plus the language model.
 *
 * The lexicon is the set of whole-word tokens the predictor already knows
 * (about 30k distinct a-z words from GPT-2's vocabulary), so it needs no
 * separate dictionary. Each candidate gets two scores:
 *
 *   shape: the user's path and the word's ideal path (key centre to key
 *          centre, repeated letters collapsed), both resampled to N points.
 *          Mean point distance in key widths, scored as a Gaussian log-likelihood.
 *   LM:    log P(word | context) from the same logits that drive the
 *          suggestion strip.
 *
 * Words whose first or last key is far from where the path starts or ends are
 * pruned before shape scoring, so a decode touches a few hundred words, not
 * 30k. Shape-identical words ("to"/"too", "for"/"four") are decided by context.
 *
 * Limitation: as with suggestions, only words that are a single GPT-2 token
 * are swipeable. Multi-token words have to be typed.
 */
class SwipeDecoder(words: WordCandidates) {

    private class Entry(val word: String, val spaced: IntArray, val bare: IntArray, val letters: IntArray)

    /** Lexicon bucketed by first letter. */
    private val buckets: Array<ArrayList<Entry>>

    init {
        val spaced = HashMap<String, MutableList<Int>>()
        val bare = HashMap<String, MutableList<Int>>()
        for (id in 0 until words.vocabSize) {
            val lw = words.lowerAt(id) ?: continue
            if (lw.length < 2 || lw.length > MAX_LEN || !lw.all { it in 'a'..'z' }) continue
            (if (words.hasLeadingSpace(id)) spaced else bare).getOrPut(lw) { ArrayList(2) } += id
        }
        buckets = Array(26) { ArrayList() }
        for (w in spaced.keys + bare.keys) {
            val letters = collapse(w)
            if (letters.size < 2) continue // "aa": no path to draw
            buckets[w[0] - 'a'] += Entry(
                w,
                spaced[w]?.toIntArray() ?: IntArray(0),
                bare[w]?.toIntArray() ?: IntArray(0),
                letters,
            )
        }
    }

    private val wordAt: (Int) -> String = { words.wordAt(it)!! }

    /**
     * Best [k] words for [input], given next-token [logits] for the context the
     * word will be inserted into. [spaceBefore] selects which token variant
     * ("Ġword" vs "word") the context calls for; the other variant is still
     * allowed, with a penalty.
     */
    fun decode(input: SwipeInput, logits: FloatArray, spaceBefore: Boolean, k: Int = 9): List<String> {
        val kw = input.keyWidth
        val ux = FloatArray(input.xs.size) { input.xs[it] / kw }
        val uy = FloatArray(input.ys.size) { input.ys[it] / kw }
        val (rx, ry) = resample(ux, uy, N)
        val sx = rx[0]; val sy = ry[0]; val ex = rx[N - 1]; val ey = ry[N - 1]
        val cx = FloatArray(26) { input.keyX[it] / kw }
        val cy = FloatArray(26) { input.keyY[it] / kw }

        val lse = logSumExp(logits)
        val scored = ArrayList<Pair<Float, String>>()
        val tx = FloatArray(32); val ty = FloatArray(32)
        for (first in 0 until 26) {
            if (cx[first].isNaN() || dist(cx[first], cy[first], sx, sy) > PRUNE) continue
            for (e in buckets[first]) {
                val last = e.letters.last()
                if (cx[last].isNaN() || dist(cx[last], cy[last], ex, ey) > PRUNE) continue
                if (e.letters.any { cx[it].isNaN() }) continue
                for ((i, l) in e.letters.withIndex()) { tx[i] = cx[l]; ty[i] = cy[l] }
                val (wx, wy) = resample(tx.copyOf(e.letters.size), ty.copyOf(e.letters.size), N)
                var d = 0f
                for (i in 0 until N) d += dist(rx[i], ry[i], wx[i], wy[i])
                d /= N
                if (d > MAX_MEAN_DIST) continue
                val shape = -(d * d) / (2 * SIGMA * SIGMA)

                // Best-scoring token variant for this slot, and its surface form.
                val (preferred, other) = if (spaceBefore) e.spaced to e.bare else e.bare to e.spaced
                var bestLp = Float.NEGATIVE_INFINITY
                var form = e.word
                for (id in preferred) { val lp = logits[id] - lse; if (lp > bestLp) { bestLp = lp; form = wordAt(id) } }
                for (id in other) { val lp = logits[id] - lse - WRONG_SLOT_PENALTY; if (lp > bestLp) { bestLp = lp; form = wordAt(id) } }
                if (bestLp.isNaN()) continue
                scored += (shape + LM_WEIGHT * max(bestLp, LM_FLOOR)) to form
            }
        }
        scored.sortByDescending { it.first }
        val seen = HashSet<String>()
        val out = ArrayList<String>(k)
        val best = scored.firstOrNull()?.first ?: return out
        for ((score, w) in scored) {
            if (best - score > ALT_MARGIN) break // far-fetched alternates are noise, not choices
            if (seen.add(w.lowercase())) out += WordCandidates.fixStandaloneI(w)
            if (out.size == k) break
        }
        return out
    }

    /**
     * Runs a few throwaway decodes so the first real swipe doesn't pay for
     * ART compiling this code (about 110 ms cold vs 20-35 ms warm on device).
     * Call on a background thread at load.
     */
    fun warmUp(vocabSize: Int) {
        val kx = FloatArray(26) { Float.NaN }; val ky = FloatArray(26) { Float.NaN }
        listOf("qwertyuiop" to 0f, "asdfghjkl" to 0.5f, "zxcvbnm" to 1.5f).forEachIndexed { r, (keys, off) ->
            keys.forEachIndexed { i, c -> kx[c - 'a'] = off + i + 0.5f; ky[c - 'a'] = r * 1.4f + 0.7f }
        }
        val path = "there".map { it - 'a' }
        val swipe = SwipeInput(FloatArray(path.size) { kx[path[it]] }, FloatArray(path.size) { ky[path[it]] }, kx, ky, 1f)
        val logits = FloatArray(vocabSize)
        repeat(3) { decode(swipe, logits, spaceBefore = true) }
    }

    companion object {
        /** Alternates scoring more than this (nats) below the best are dropped. */
        private const val ALT_MARGIN = 7f
        private const val N = 40
        private const val MAX_LEN = 20
        /** Max distance (key widths) between path start/end and the word's first/last key. */
        private const val PRUNE = 1.4f
        private const val MAX_MEAN_DIST = 1.6f
        private const val SIGMA = 0.45f
        private const val LM_WEIGHT = 0.6f
        private const val LM_FLOOR = -25f
        private const val WRONG_SLOT_PENALTY = 3f

        /** Letter indices with consecutive repeats removed: "hello" -> h e l o. */
        internal fun collapse(w: String): IntArray {
            val out = ArrayList<Int>(w.length)
            for (c in w) { val i = c - 'a'; if (out.isEmpty() || out.last() != i) out += i }
            return out.toIntArray()
        }

        private fun dist(x1: Float, y1: Float, x2: Float, y2: Float): Float {
            val dx = x1 - x2; val dy = y1 - y2
            return sqrt(dx * dx + dy * dy)
        }

        /** Resample a polyline to [n] points evenly spaced along its length. */
        internal fun resample(xs: FloatArray, ys: FloatArray, n: Int): Pair<FloatArray, FloatArray> {
            val ox = FloatArray(n); val oy = FloatArray(n)
            var total = 0f
            for (i in 1 until xs.size) total += dist(xs[i - 1], ys[i - 1], xs[i], ys[i])
            if (xs.size == 1 || total == 0f) { ox.fill(xs[0]); oy.fill(ys[0]); return ox to oy }
            val step = total / (n - 1)
            ox[0] = xs[0]; oy[0] = ys[0]
            var seg = 1
            var segStart = 0f // distance along path at xs[seg-1]
            for (j in 1 until n - 1) {
                val target = j * step
                while (seg < xs.size - 1 && segStart + dist(xs[seg - 1], ys[seg - 1], xs[seg], ys[seg]) < target) {
                    segStart += dist(xs[seg - 1], ys[seg - 1], xs[seg], ys[seg]); seg++
                }
                val len = dist(xs[seg - 1], ys[seg - 1], xs[seg], ys[seg])
                val t = if (len == 0f) 0f else ((target - segStart) / len).coerceIn(0f, 1f)
                ox[j] = xs[seg - 1] + t * (xs[seg] - xs[seg - 1])
                oy[j] = ys[seg - 1] + t * (ys[seg] - ys[seg - 1])
            }
            ox[n - 1] = xs.last(); oy[n - 1] = ys.last()
            return ox to oy
        }

        private fun logSumExp(v: FloatArray): Float {
            var m = Float.NEGATIVE_INFINITY
            for (x in v) if (x > m) m = x
            var s = 0.0
            for (x in v) if (!x.isNaN()) s += exp((x - m).toDouble())
            return m + ln(s).toFloat()
        }
    }
}
