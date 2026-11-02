package com.swiperight.predict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WordCandidatesTest {
    private val tok = TestAssets.tokenizer
    private val wc = WordCandidates(tok)

    private fun logits(vararg ranked: String): FloatArray {
        val l = FloatArray(tok.vocabSize) { -10f }
        ranked.forEachIndexed { i, t -> l[tok.tokenId(t) ?: error("no token $t")] = 10f - i }
        return l
    }

    private fun ctx(prefix: String = "", spaceBefore: Boolean = true) =
        PredictionContext("x", prefix, spaceBefore, insertSpace = false, docStart = false)

    @Test
    fun nextWordPicksSpacedWordTokensOnly() {
        // "," and "Ġ123" are not words; "ing" lacks the leading space.
        val l = logits(",", "Ġ123", "ing", "Ġmovie", "Ġstore", "Ġpark", "Ġbeach")
        assertEquals(listOf("movie", "store", "park"), wc.select(l, ctx()))
    }

    @Test
    fun prefixCompletionAndCaseDedupe() {
        val l = logits("ĠHome", "Ġhome", "Ġhouse", "Ġtomorrow", "Ġhospital")
        assertEquals(listOf("Home", "house", "hospital"), wc.select(l, ctx("ho")))
        assertEquals(listOf("Home", "House", "Hospital"), wc.select(l, ctx("Ho")))
        assertEquals(listOf("HOME", "HOUSE", "HOSPITAL"), wc.select(l, ctx("HO")))
    }

    @Test
    fun noSpaceSlotUsesBareTokens() {
        val l = logits("ĠThe", "The", "I", "We")
        assertEquals(listOf("The", "I", "We"), wc.select(l, ctx(spaceBefore = false)))
    }

    @Test
    fun suppressedContextGivesNothing() {
        val l = logits("Ġthe")
        assertTrue(wc.select(l, ctx().copy(suppress = true)).isEmpty())
    }

    @Test
    fun fewerThanThreeMatchesIsFine() {
        val l = logits("Ġelephant")
        val out = wc.select(l, ctx("eleph"))
        assertEquals("elephant", out.first())
        assertTrue(out.size <= 3)
    }

    @Test
    fun nanLogitsIgnored() {
        val l = logits("Ġmovie", "Ġstore")
        l[tok.tokenId("Ġpark")!!] = Float.NaN
        assertFalse(wc.select(l, ctx()).contains("park"))
    }

    @Test
    fun standaloneIIsCapitalized() {
        val l = logits("Ġi", "Ġit", "Ġif")
        assertEquals(listOf("I", "it", "if"), wc.select(l, ctx("i")))
        assertEquals("I", WordCandidates.fixStandaloneI("i"))
        assertEquals("in", WordCandidates.fixStandaloneI("in"))
    }

    @Test
    fun nineSuggestions() {
        val l = logits("Ġthe", "Ġa", "Ġmy", "Ġyour", "Ġthis", "Ġthat", "Ġour", "Ġhis", "Ġher", "Ġtheir")
        assertEquals(listOf("the", "a", "my", "your", "this", "that", "our", "his", "her"), wc.select(l, ctx(), k = 9))
    }

    @Test
    fun wordLikeRules() {
        assertTrue(WordCandidates.isWordLike("don't"))
        assertTrue(WordCandidates.isWordLike("café"))
        assertFalse(WordCandidates.isWordLike("'t"))
        assertFalse(WordCandidates.isWordLike("a1"))
        assertFalse(WordCandidates.isWordLike(""))
    }
}
