package com.swiperight.predict

/**
 * What the model should condition on, and what kind of word should come next.
 *
 * @property context text fed to the tokenizer. Excludes a partially typed word
 *   and a single trailing space, because GPT-2 attaches the space to the
 *   *following* token ("Ġword").
 * @property prefix letters of the word being typed ("" when between words).
 * @property spaceBefore the suggested word follows a space, so only "Ġ…"
 *   tokens qualify. False after a newline, an opening bracket or quote, or at
 *   the start of the document.
 * @property insertSpace the field does not end in a space yet but the word
 *   needs one before it (e.g. right after "end."). The IME adds it on commit.
 * @property docStart the whole field fits in the window, so a BOS token is
 *   prepended. GPT-2 was trained with `<|endoftext|>` between documents.
 * @property suppress nothing sensible to suggest: a "word" too long to be one
 *   (a URL or hash) or glued to digits.
 */
data class PredictionContext(
    val context: String,
    val prefix: String,
    val spaceBefore: Boolean,
    val insertSpace: Boolean,
    val docStart: Boolean,
    val suppress: Boolean = false,
)

object ContextBuilder {
    /** The spec's character window: only the last N characters are tokenized. */
    const val WINDOW_CHARS = 32

    /** Upper bound on a partially typed word we try to complete. */
    private const val MAX_PREFIX = 24

    private val OPENERS = setOf('(', '[', '{', '“', '‘', '«', '¿', '¡', '/', '@', '#')

    /**
     * @param textBeforeCursor text before the cursor. Pass more than
     *   [WINDOW_CHARS] characters when available; that is how a cut window is
     *   told apart from the true start of the field.
     */
    fun build(textBeforeCursor: CharSequence): PredictionContext {
        val full = textBeforeCursor.toString()
        val docStart = full.length <= WINDOW_CHARS
        var w = if (docStart) full else full.substring(full.length - WINDOW_CHARS)

        if (!docStart) {
            // Never start on half a surrogate pair.
            if (w.isNotEmpty() && Character.isLowSurrogate(w[0])) w = w.substring(1)
            // The window cut probably split a word. Drop that fragment, unless
            // the window is a single long word with no whitespace at all.
            val firstWs = w.indexOfFirst { it.isWhitespace() }
            if (firstWs >= 0 && firstWs < w.length - 1 && !full[full.length - WINDOW_CHARS - 1].isWhitespace()) {
                w = w.substring(firstWs + 1)
            }
        }

        // Partially typed word: trailing letters, possibly with one apostrophe
        // inside or at the end ("it'", "don't", "you’r"): a contraction being typed.
        var p = w.length
        var apostrophes = 0
        while (p > 0) {
            val c = w[p - 1]
            if (c.isLetter()) { p--; continue }
            if ((c == '\'' || c == '’') && apostrophes == 0 && p - 1 > 0 && w[p - 2].isLetter()) { apostrophes++; p--; continue }
            break
        }
        val prefix = w.substring(p)
        val before = w.substring(0, p)
        val glued = p > 0 && w[p - 1].isLetterOrDigit() // e.g. "abc123de"
        if (prefix.length > MAX_PREFIX || glued) {
            return PredictionContext(w, prefix, true, false, docStart, suppress = true)
        }
        if (prefix.isNotEmpty()) {
            return when {
                before.endsWith(' ') ->
                    PredictionContext(before.dropLast(1), prefix, spaceBefore = true, insertSpace = false, docStart = docStart)
                before.isEmpty() ->
                    PredictionContext("", prefix, spaceBefore = !docStart, insertSpace = false, docStart = docStart)
                else -> // glued to punctuation / newline: "(hel", "\nhel", "word,hel"
                    PredictionContext(before, prefix, spaceBefore = false, insertSpace = false, docStart = docStart)
            }
        }

        // Between words.
        if (w.isEmpty()) {
            return PredictionContext("", "", spaceBefore = !docStart, insertSpace = false, docStart = docStart)
        }
        val last = w.last()
        return when {
            last == ' ' -> {
                val ctx = w.dropLast(1)
                // " " at the very start of a field behaves like the start.
                if (ctx.isBlank() && docStart) PredictionContext("", "", false, false, true)
                else PredictionContext(ctx, "", spaceBefore = true, insertSpace = false, docStart = docStart)
            }
            last == '\n' || last == '\t' || last.isWhitespace() ->
                PredictionContext(w, "", spaceBefore = false, insertSpace = false, docStart = docStart)
            last in OPENERS || isOpeningQuote(w) ->
                PredictionContext(w, "", spaceBefore = false, insertSpace = false, docStart = docStart)
            else -> // after "." "," "!" digits, closing quotes: next word needs a space
                PredictionContext(w, "", spaceBefore = true, insertSpace = true, docStart = docStart)
        }
    }

    /** A straight quote counts as opening when at the start or after whitespace/opener. */
    private fun isOpeningQuote(w: String): Boolean {
        val q = w.last()
        if (q != '"' && q != '\'') return false
        if (w.length == 1) return true
        val prev = w[w.length - 2]
        return prev.isWhitespace() || prev in OPENERS
    }
}
