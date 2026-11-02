package com.swiperight.predict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Ranker logic with a fake model: [first] for the context, [second] after the base token. */
class SuggestionRankerTest {
    private val tok = TestAssets.tokenizer
    private val wc = WordCandidates(tok)
    private val contractions = Contractions(tok)
    private val ranker = SuggestionRanker(wc, contractions)

    private fun logits(vararg pairs: Pair<String, Float>) = FloatArray(tok.vocabSize) { -20f }.also { l ->
        for ((t, v) in pairs) l[tok.tokenId(t) ?: error("no token $t")] = v
    }

    private fun ctx(prefix: String = "", spaceBefore: Boolean = true) =
        PredictionContext("I think", prefix, spaceBefore, insertSpace = false, docStart = false)

    @Test
    fun contractionsAreValidatedAgainstTokenizer() {
        val words = contractions.entries.map { it.word }
        assertTrue("it's" in words && "don't" in words && "I'm" in words && "you're" in words)
        val its = contractions.entries.first { it.word == "it's" }
        assertEquals("its", its.stripped)
        assertTrue(its.variants.any { it.spaced && tok.token(it.baseId) == "Ġit" })
    }

    @Test
    fun itsBeatsItWhenTheModelExpectsApostropheS() {
        var passes = 0
        val first = logits("Ġit" to 5f, "Ġthat" to 4f, "Ġthe" to 3.5f, "Ġwe" to 3f)
        val second = logits("'s" to 6f, "Ġis" to 2f, "'ll" to 3f)
        val out = ranker.rank(first, ctx(), 9) { id -> passes++; assertEquals("Ġit", tok.token(id)); second }
        assertEquals(1, passes)
        assertEquals("it's", out[0])
        assertTrue("it'll" in out)
        assertTrue(out.indexOf("it") > out.indexOf("it's")) // "it" keeps only the leftover probability
    }

    @Test
    fun apostropheLessPrefixFindsContraction() {
        val first = logits("Ġdon" to 6f, "Ġdone" to 2f, "Ġdont" to 5f)
        val second = logits("'t" to 8f)
        val out = ranker.rank(first, ctx("dont"), 9) { second }
        assertEquals("don't", out[0])
        assertFalse("dont" in out) // web-typo token hidden in favour of the contraction
    }

    @Test
    fun apostrophePrefixOnlyOffersContractions() {
        val first = logits("Ġit" to 5f, "Ġitem" to 4f)
        val second = logits("'s" to 5f, "'ll" to 4f, "'d" to 1f)
        assertEquals(listOf("it's", "it'll", "it'd"), ranker.rank(first, ctx("it'"), 9) { second })
    }

    @Test
    fun iContractionsAreCapitalized() {
        val first = logits("ĠI" to 6f, "Ġyou" to 2f)
        val second = logits("'m" to 6f, "'ll" to 3f)
        val out = ranker.rank(first, ctx("i"), 9) { second }
        assertEquals("I'm", out[0])
        assertTrue("I" in out && "I'll" in out)
    }

    @Test
    fun noSecondPassWhenNoContractionBaseIsVisible() {
        val first = logits("Ġmovie" to 9f, "Ġstore" to 8f, "Ġpark" to 7f, "Ġit" to -5f)
        var passes = 0
        val out = ranker.rank(first, ctx(), 3) { passes++; first }
        assertEquals(0, passes)
        assertEquals(listOf("movie", "store", "park"), out)
    }
}
