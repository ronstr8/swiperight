package com.swiperight.predict

import org.json.JSONArray
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BpeTokenizerTest {
    private val tok = TestAssets.tokenizer

    /** Token ids must match the Hugging Face GPT-2 tokenizer exactly (vocabulary sync). */
    @Test
    fun matchesHuggingFaceGolden() {
        val text = javaClass.classLoader!!.getResource("tokenizer_golden.json")!!.readText()
        val cases = JSONArray(text)
        assertTrue(cases.length() > 20)
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val s = c.getString("text")
            val want = c.getJSONArray("ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
            assertArrayEquals("plain: \"$s\"", want, tok.encode(s))
            val wantSpecial = c.getJSONArray("ids_with_special").let { a -> IntArray(a.length()) { a.getInt(it) } }
            assertArrayEquals("special: \"$s\"", wantSpecial, tok.encode(s, allowSpecial = true))
        }
    }

    @Test
    fun roundTripsArbitraryUnicode() {
        for (s in listOf("hello world", "naïve café — “quotes”", "🎉🔥👍🏽", "日本語", "tabs\tand\nnewlines", "  ")) {
            assertEquals(s, tok.decode(tok.encode(s)))
        }
    }

    @Test
    fun typedSpecialTokenIsPlainText() {
        val ids = tok.encode("<|endoftext|>")
        assertTrue(ids.size > 1)
        assertTrue(ids.none { tok.isSpecial(it) })
    }

    @Test
    fun emptyInput() {
        assertEquals(0, tok.encode("").size)
    }

    @Test
    fun vocabMatchesModelContract() {
        assertEquals(50257, tok.vocabSize)
        assertEquals(0, tok.skippedMerges)
        assertEquals(50256, tok.tokenId("<|endoftext|>"))
    }

    /** A custom vocab missing byte symbols maps them to <unk> (or drops them) instead of crashing. */
    @Test
    fun unknownBytesUseUnkOrAreDropped() {
        val vocab = mapOf("h" to 0, "i" to 1, "hi" to 2, "<unk>" to 3)
        val withUnk = BpeTokenizer(vocab, listOf("h" to "i"), setOf("<unk>"))
        assertArrayEquals(intArrayOf(2, 3, 3), withUnk.encode("hi!?").let { it })
        val noUnk = BpeTokenizer(vocab - "<unk>", listOf("h" to "i"), emptySet())
        assertArrayEquals(intArrayOf(2), noUnk.encode("hi!"))
        assertEquals(1, noUnk.droppedBytes)
    }
}
