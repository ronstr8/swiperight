package com.swiperight

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.swiperight.predict.Predictor
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device checks: real model, real interpreter, real latency.
 *   ./gradlew connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class PredictorDeviceTest {

    companion object {
        private lateinit var p: Predictor

        @BeforeClass @JvmStatic
        fun load() {
            val ctx = InstrumentationRegistry.getInstrumentation().targetContext
            p = Predictor.create(ctx, verifyModelHash = true)
            Log.i("PredictorDeviceTest", "info=${p.info}")
        }

        @AfterClass @JvmStatic
        fun close() = p.close()
    }

    @Test
    fun commonPhrasesGiveThreeWords() {
        for (s in listOf("I'm going to the ", "Thank you so much for ", "Let me know if you ")) {
            val w = p.predictNextWords(s)
            assertEquals("$s -> $w", 3, w.size)
            assertTrue(w.all { it.isNotBlank() && it.none(Char::isWhitespace) })
        }
    }

    @Test
    fun obviousContinuationsRankHigh() {
        assertTrue(p.predictNextWords("Thank you so much for ").any { it.lowercase() in setOf("the", "your", "all", "coming", "being", "this") })
        assertTrue("tomorrow" in p.predictNextWords("See you tomor").map { it.lowercase() })
    }

    @Test
    fun prefixIsRespected() {
        val w = p.predictNextWords("What time does the mo")
        assertTrue(w.toString(), w.isNotEmpty() && w.all { it.lowercase().startsWith("mo") })
    }

    @Test
    fun edgeCasesDoNotCrash() {
        val inputs = listOf(
            "", " ", "\n", "a", "🎉🎉🎉", "日本語のテキスト", "<|endoftext|>", "don'",
            "x".repeat(5000), "word ".repeat(2000), "1234567890", "!!!???...", "\u0000\u0001",
            "\uD83D", // lone surrogate
        )
        for (s in inputs) {
            val w = p.predictNextWords(s)
            assertTrue("$s -> $w", w.size <= 3)
        }
    }

    /** Not an assertion: logs suggestions for eyeballing (`adb logcat -s PredictorSamples`). */
    @Test
    fun logSamples() {
        for (s in listOf(
            "", "I'm going to the ", "Thank you so much for ", "Can we meet at the ",
            "Let me know if you ", "I'll be there in ", "Happy birthday! Hope you have a ",
            "What time does the mo", "See you tomor", "Sorry, I can't make it to",
            "The meeting has been moved to next ", "Do you want to grab ", "Please call me when you get ",
            "We should probably leave early to avoid the ", "Hello\n", "He said (", "This is the end.",
            "I think ", "I think it", "I think its", "Yes, it'", "I don", "I dont", "Sorry, I ", "Well, you", "Trust me, ",
        )) {
            val r = p.predict(s)
            Log.i("PredictorSamples", "%-48s -> %s (%.1f ms)".format("\"$s\"", r.words, r.inferenceMillis))
        }
    }

    private fun syntheticSwipe(word: String): com.swiperight.predict.SwipeInput {
        val w = 100f
        val kx = FloatArray(26) { Float.NaN }; val ky = FloatArray(26) { Float.NaN }
        listOf("qwertyuiop" to 0f, "asdfghjkl" to 0.5f, "zxcvbnm" to 1.5f).forEachIndexed { r, (keys, off) ->
            keys.forEachIndexed { i, c -> kx[c - 'a'] = (off + i + 0.5f) * w; ky[c - 'a'] = (r * 1.4f + 0.7f) * w }
        }
        val rnd = java.util.Random(7)
        val xs = ArrayList<Float>(); val ys = ArrayList<Float>()
        for (i in 0 until word.length - 1) {
            val a = word[i] - 'a'; val b = word[i + 1] - 'a'
            for (s in 0 until 8) {
                val t = s / 8f
                xs += kx[a] + t * (kx[b] - kx[a]) + (rnd.nextFloat() - 0.5f) * 30
                ys += ky[a] + t * (ky[b] - ky[a]) + (rnd.nextFloat() - 0.5f) * 30
            }
        }
        xs += kx[word.last() - 'a']; ys += ky[word.last() - 'a']
        return com.swiperight.predict.SwipeInput(xs.toFloatArray(), ys.toFloatArray(), kx, ky, w)
    }

    @Test
    fun swipeDecodesInContext() {
        // Warm-up: the first decode runs before ART has compiled the decoder.
        val warm = (0 until 10).map { p.decodeSwipe("I'm going to the ", syntheticSwipe("store")).inferenceMillis }
        Log.i("PredictorSamples", "swipe warm-up ms: " + warm.joinToString { "%.1f".format(it) })
        for ((ctx, word) in listOf(
            "I'm going to the " to "store", "Thank you so " to "much", "See you " to "tomorrow",
            "" to "hello", "Can we meet at the " to "office", "I love" to "you",
            "I think " to "its", "Sorry, I " to "dont", "Honestly " to "im",
        )) {
            val r = p.decodeSwipe(ctx, syntheticSwipe(word))
            Log.i("PredictorSamples", "swipe %-28s + '%s' -> %s (%.1f ms)".format("\"$ctx\"", word, r.words, r.inferenceMillis))
            val want = mapOf("its" to setOf("its", "it's"), "dont" to setOf("don't"), "im" to setOf("i'm"))[word] ?: setOf(word)
            assertTrue("$ctx + $word -> ${r.words}", r.words.take(3).any { it.lowercase() in want })
            assertTrue(r.inferenceMillis < 100.0)
        }
    }

    @Test
    fun latencyUnderBudget() {
        val samples = listOf("I'm going to the ", "Can we meet at the ", "See you tomor", "")
        repeat(5) { p.predict(samples[it % samples.size]) }
        val times = (0 until 40).map {
            val t = System.nanoTime()
            p.predict(samples[it % samples.size])
            (System.nanoTime() - t) / 1e6
        }.sorted()
        val p50 = times[times.size / 2]
        val p90 = times[(times.size * 9) / 10]
        Log.i("PredictorDeviceTest", "end-to-end latency p50=$p50 ms p90=$p90 ms backend=${p.info.backend}")
        assertTrue("p90 $p90 ms over the 100 ms budget", p90 < 100.0)
    }
}
