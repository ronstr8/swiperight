package com.swiperight.predict

import java.util.Locale
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Final suggestion list: single-token words from [WordCandidates] merged with
 * two-token contractions from [Contractions], all ranked by log-probability.
 *
 * At most one extra model pass per prediction: for the most probable
 * contraction base in this slot ("it", "don", "I", ...) it asks for the
 * next-token distribution after that base. That pass yields the exact
 * probability of every contraction sharing the base ("it's", "it'll", "it'd").
 * The plain base word keeps only the leftover probability, P(base) x
 * (1 - sum of suffix probs), because P("Ġit") includes every "it's".
 *
 * The extra pass runs only when its result could reach the visible strip: the
 * user is typing a word that could be a contraction, or the base itself ranks
 * in the top 3. Otherwise latency is unchanged.
 */
class SuggestionRanker(private val words: WordCandidates, private val contractions: Contractions) {

    private class Cand(val word: String, var lp: Float)

    private val typoForms: Set<String> =
        contractions.entries.map { it.stripped }.toSet() - Contractions.REAL_WORDS

    /**
     * @param nextLogits logits for the token after context + the given base
     *   token id. Called at most once.
     */
    fun rank(
        logits: FloatArray,
        ctx: PredictionContext,
        k: Int,
        nextLogits: (baseId: Int) -> FloatArray,
    ): List<String> {
        if (ctx.suppress) return emptyList()
        val lse = logSumExp(logits)
        val cands = words.selectScored(logits, ctx, k + EXTRA).map { Cand(it.first, it.second - lse) }.toMutableList()

        val hits = contractions.matching(ctx.prefix).mapNotNull { e ->
            contractions.bestVariant(e, ctx.spaceBefore, logits)?.let { e to it }
        }
        if (hits.isNotEmpty()) {
            val baseId = hits.maxBy { logits[it.second.baseId] }.second.baseId
            val baseLp = logits[baseId] - lse
            val visibleCut = cands.getOrNull(2)?.lp ?: Float.NEGATIVE_INFINITY
            if (ctx.prefix.isNotEmpty() || baseLp >= visibleCut) {
                val next = nextLogits(baseId)
                val lse2 = logSumExp(next)
                var mass = 0.0
                for ((e, v) in hits) {
                    if (v.baseId != baseId) continue
                    val lp2 = next[e.suffixId] - lse2
                    mass += exp(lp2.toDouble())
                    cands += Cand(v.surface, baseLp + lp2)
                }
                val baseLower = words.lowerAt(baseId)
                val rest = ln(max(1e-6, 1.0 - mass)).toFloat()
                for (c in cands) if (c.word.lowercase(Locale.ROOT) == baseLower) c.lp += rest
            }
        }

        cands.removeAll { it.word.lowercase(Locale.ROOT) in typoForms }
        cands.sortByDescending { it.lp }
        val seen = HashSet<String>()
        val out = ArrayList<String>(k)
        for (c in cands) {
            if (!seen.add(c.word.lowercase(Locale.ROOT))) continue
            out += WordCandidates.format(c.word, ctx.prefix)
            if (out.size == k) break
        }
        return out
    }

    companion object {
        /** Extra regular candidates fetched so removals/re-ranking still leave k. */
        private const val EXTRA = 6

        internal fun logSumExp(v: FloatArray): Float {
            var m = Float.NEGATIVE_INFINITY
            for (x in v) if (x > m) m = x
            var s = 0.0
            for (x in v) if (!x.isNaN()) s += exp((x - m).toDouble())
            return m + ln(s).toFloat()
        }
    }
}
