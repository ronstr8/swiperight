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
