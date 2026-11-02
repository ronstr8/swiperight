package com.swiperight.predict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class SwipeDecoderTest {
    private val tok = TestAssets.tokenizer
    private val decoder = SwipeDecoder(WordCandidates(tok))

    // Synthetic QWERTY geometry in key widths; keys 1.4 wide-units tall.
    private val kx = FloatArray(26) { Float.NaN }
    private val ky = FloatArray(26) { Float.NaN }

    init {
        listOf("qwertyuiop" to 0f, "asdfghjkl" to 0.5f, "zxcvbnm" to 1.5f).forEachIndexed { row, (keys, off) ->
            keys.forEachIndexed { i, c -> kx[c - 'a'] = (off + i + 0.5f) * W; ky[c - 'a'] = (row * 1.4f + 0.7f) * W }
        }
    }

    /** A sloppy finger path through [word]'s keys (pixel units, W px per key). */
    private fun swipe(word: String, noise: Float = 0.18f, seed: Int = 1): SwipeInput {
        val rnd = Random(seed)
        val xs = ArrayList<Float>(); val ys = ArrayList<Float>()
        val pts = word.map { kx[it - 'a'] to ky[it - 'a'] }
        for (i in 0 until pts.size - 1) {
            val (x0, y0) = pts[i]; val (x1, y1) = pts[i + 1]
            for (s in 0 until 8) {
                val t = s / 8f
                xs += x0 + t * (x1 - x0) + (rnd.nextFloat() - 0.5f) * 2 * noise * W
                ys += y0 + t * (y1 - y0) + (rnd.nextFloat() - 0.5f) * 2 * noise * W
            }
        }
        xs += pts.last().first; ys += pts.last().second
        return SwipeInput(xs.toFloatArray(), ys.toFloatArray(), kx, ky, W)
    }

    private val flat = FloatArray(tok.vocabSize)

    private fun boost(vararg tokens: Pair<String, Float>): FloatArray {
        val l = FloatArray(tok.vocabSize)
        for ((t, v) in tokens) l[tok.tokenId(t)!!] = v
        return l
    }

    @Test
    fun decodesCleanAndSloppySwipes() {
        for (w in listOf("hello", "world", "keyboard", "please", "thanks", "tomorrow", "meeting", "great")) {
            for (seed in 1..3) {
                val out = decoder.decode(swipe(w, seed = seed), flat, spaceBefore = true)
                assertTrue("$w (seed $seed) -> $out", w in out.take(3).map { it.lowercase() })
            }
        }
    }

    @Test
    fun languageModelBreaksShapeTies() {
        // "to" and "too" draw the same path; context decides.
        val s = swipe("to", noise = 0.05f)
        assertEquals("too", decoder.decode(s, boost("Ġtoo" to 8f, "Ġto" to 2f), true)[0])
        assertEquals("to", decoder.decode(s, boost("Ġto" to 8f, "Ġtoo" to 2f), true)[0])
    }

    @Test
    fun returnsUpToNineDistinctWords() {
        val out = decoder.decode(swipe("the"), flat, true, k = 9)
        assertTrue(out.size in 1..9)
        assertEquals(out.size, out.map { it.lowercase() }.toSet().size)
    }

    @Test
    fun keepsModelCasingForProperNouns() {
        val out = decoder.decode(swipe("london", noise = 0.05f), boost("ĠLondon" to 10f), true)
        assertEquals("London", out[0])
    }

    @Test
    fun collapseRepeats() {
        assertEquals(listOf('h', 'e', 'l', 'o'), SwipeDecoder.collapse("hello").map { 'a' + it })
    }

    @Test
    fun resampleKeepsEndpoints() {
        val (x, y) = SwipeDecoder.resample(floatArrayOf(0f, 10f), floatArrayOf(0f, 0f), 11)
        assertEquals(0f, x[0], 1e-6f); assertEquals(10f, x[10], 1e-6f); assertEquals(5f, x[5], 1e-5f)
        assertTrue(y.all { it == 0f })
    }

    companion object { private const val W = 100f }
}
