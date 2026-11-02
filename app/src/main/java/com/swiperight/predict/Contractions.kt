package com.swiperight.predict

import java.util.Locale

/**
 * Contractions the model can only produce as two tokens: "it's" = "Ġit" + "'s",
 * "don't" = "Ġdon" + "'t". A single-pass predictor never sees "it's" as a
 * candidate, so these are scored explicitly as
 *
 *     log P(contraction) = log P(base | context) + log P(suffix | context, base)
 *
 * where the second term comes from one extra model pass (see [Predictor]).
 *
 * Every entry is validated against the tokenizer at load: a surface form is
 * kept only if it encodes to exactly [base, suffix]. Forms GPT-2 splits
 * differently (e.g. "couldn't" with no leading space: could+n+'t) are dropped
 * rather than mis-scored.
 */
class Contractions(tok: BpeTokenizer) {

    /** One spelling of a contraction that is exactly two tokens. */
    class Variant(val surface: String, val baseId: Int, val spaced: Boolean)

    class Entry(
        /** Canonical spelling, e.g. "it's", "I'm". */
        val word: String,
        /** Lowercase without apostrophe ("its", "im"): what an apostrophe-less prefix or a swipe spells. */
        val stripped: String,
        val suffixId: Int,
        val variants: List<Variant>,
    )

    val entries: List<Entry>

    init {
        val out = ArrayList<Entry>()
        for (w in WORDS) {
            val apo = w.indexOf('\'')
            val suffix = tok.tokenId(w.substring(apo)) ?: continue
            val forms = LinkedHashSet<String>().apply {
                add(w)
                if (!w.startsWith("I'")) add(w.replaceFirstChar { it.titlecase(Locale.ROOT) })
            }
            val variants = ArrayList<Variant>()
            for (f in forms) for (spaced in listOf(true, false)) {
                val ids = tok.encode(if (spaced) " $f" else f)
                if (ids.size == 2 && ids[1] == suffix) variants += Variant(f, ids[0], spaced)
            }
            if (variants.isNotEmpty()) out += Entry(w, strip(w), suffix, variants)
        }
        entries = out
    }

    /** Entries compatible with a typed [prefix] (may contain an apostrophe or not). */
    fun matching(prefix: String): List<Entry> {
        if (prefix.isEmpty()) return entries
        val p = normalize(prefix)
        return if ('\'' in p) entries.filter { it.word.lowercase(Locale.ROOT).startsWith(p) }
        else entries.filter { it.stripped.startsWith(p) }
    }

    /** The most probable spelling of [e] for this slot, or null if none fits it. */
    fun bestVariant(e: Entry, spaceBefore: Boolean, logits: FloatArray): Variant? =
        e.variants.filter { it.spaced == spaceBefore }.maxByOrNull { logits[it.baseId] }

    companion object {
        /**
         * Apostrophe-less spellings that are real words in their own right. Any
         * other single token equal to a contraction's stripped form ("dont",
         * "im", "youre") is a typo the model learned from the web, and is
         * hidden in favour of the contraction.
         */
        val REAL_WORDS = setOf("its", "cant", "wont", "well", "were", "hell", "shell", "shed", "wed", "ill", "id", "lets")

        fun normalize(s: String) = s.lowercase(Locale.ROOT).replace('’', '\'')

        fun strip(s: String) = normalize(s).replace("'", "")

        /** Common English contractions (Unicode apostrophes are normalized on input). */
        val WORDS = listOf(
            "I'm", "I'll", "I've", "I'd",
            "it's", "it'll", "it'd", "that's", "that'll", "there's", "there'll", "here's",
            "what's", "what'll", "where's", "who's", "who'll", "how's", "let's",
            "he's", "he'll", "he'd", "she's", "she'll", "she'd",
            "we're", "we'll", "we've", "we'd", "you're", "you'll", "you've", "you'd",
            "they're", "they'll", "they've", "they'd",
            "don't", "doesn't", "didn't", "can't", "couldn't", "won't", "wouldn't", "shouldn't",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "ain't",
            "everyone's", "everybody's", "someone's", "nobody's", "today's",
        )
    }
}
