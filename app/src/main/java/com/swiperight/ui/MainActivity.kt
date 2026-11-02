package com.swiperight.ui

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Debug
import android.provider.Settings
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView
import com.swiperight.BuildConfig
import com.swiperight.R
import com.swiperight.predict.ModelMismatchException
import com.swiperight.predict.Predictor
import java.util.concurrent.Executors

/**
 * Setup + diagnostics: enable/select the keyboard, a test field, and an
 * on-device benchmark that measures per-backend latency against the 100 ms
 * budget, shows sample predictions and checks every asset hash.
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private lateinit var status: TextView
    private lateinit var output: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        status = findViewById(R.id.status)
        output = findViewById(R.id.output)
        findViewById<Button>(R.id.enable).setOnClickListener {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        findViewById<Button>(R.id.choose).setOnClickListener {
            (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker()
        }
        findViewById<Button>(R.id.bench).setOnClickListener { runBenchmark(it as Button) }
    }

    override fun onResume() {
        super.onResume()
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        val enabled = imm.enabledInputMethodList.any { it.packageName == packageName }
        val current = Settings.Secure.getString(contentResolver, Settings.Secure.DEFAULT_INPUT_METHOD)
            ?.startsWith("$packageName/") == true
        status.text = getString(
            R.string.status_fmt,
            if (enabled) "✓" else "✗",
            if (current) "✓" else "✗",
            BuildConfig.VERSION_NAME,
        )
    }

    private fun runBenchmark(button: Button) {
        button.isEnabled = false
        output.text = getString(R.string.bench_running)
        bg.execute {
            val report = StringBuilder()
            fun log(s: String) {
                report.appendLine(s)
                val snapshot = report.toString()
                runOnUiThread { output.text = snapshot }
            }
            try {
                benchmark(::log)
            } catch (e: ModelMismatchException) {
                log("MODEL/TOKENIZER MISMATCH: ${e.message}")
            } catch (e: Throwable) {
                log("FAILED: $e")
            }
            runOnUiThread { button.isEnabled = true }
        }
    }

    private fun benchmark(log: (String) -> Unit) {
        val nativeBefore = Debug.getNativeHeapAllocatedSize()
        val auto = Predictor.create(this, Predictor.Backend.AUTO, verifyModelHash = true)
        val nativeAfter = Debug.getNativeHeapAllocatedSize()
        log("model ${auto.info.modelVersion}: hashes OK")
        log("AUTO picked ${auto.info.backend} (${auto.info.numThreads} threads), load ${auto.info.loadMillis} ms")
        if (auto.info.backendTimingsMs.isNotEmpty()) {
            log("  probe medians: " + auto.info.backendTimingsMs.entries.joinToString { "${it.key}=${"%.1f".format(it.value)} ms" })
        }
        log("  native heap +${(nativeAfter - nativeBefore) / (1 shl 20)} MB (XNNPACK packed weights + arena)")
        log("")
        log("Samples (text before cursor → suggestions):")
        for (s in SAMPLES) {
            val p = auto.predict(s)
            log("  \"${s.takeLast(32)}\" → ${p.words}  [${"%.1f".format(p.inferenceMillis)} ms, ${p.tokenCount} tok]")
        }
        log("")
        log(latencyLine(auto, "end-to-end ${auto.info.backend}"))
        auto.close()

        for (b in listOf(Predictor.Backend.CPU, Predictor.Backend.NNAPI)) {
            try {
                val p = Predictor.create(this, b)
                if (p.info.backend != b) {
                    log("$b: unavailable on this device (fell back to ${p.info.backend})")
                } else {
                    log(latencyLine(p, b.name))
                }
                p.close()
            } catch (e: Throwable) {
                log("$b: failed: ${e.message}")
            }
        }
        for (threads in listOf(1, 4)) {
            val p = Predictor.create(this, Predictor.Backend.CPU, numThreads = threads)
            log(latencyLine(p, "CPU ${threads}T"))
            p.close()
        }
        log("")
        log("Budget: < 100 ms per prediction (tokenize + inference + word selection).")
    }

    /** End-to-end predict() latency (tokenize + inference + selection). */
    private fun latencyLine(p: Predictor, label: String): String {
        repeat(3) { p.predict(SAMPLES[it % SAMPLES.size]) }
        val times = DoubleArray(ROUNDS)
        for (i in 0 until ROUNDS) {
            val t = System.nanoTime()
            p.predict(SAMPLES[i % SAMPLES.size])
            times[i] = (System.nanoTime() - t) / 1e6
        }
        times.sort()
        fun pct(q: Double) = times[((times.size - 1) * q).toInt()]
        return "%-18s p50 %.1f ms  p90 %.1f ms  max %.1f ms".format(label, pct(0.5), pct(0.9), times.last())
    }

    override fun onDestroy() {
        bg.shutdownNow()
        super.onDestroy()
    }

    companion object {
        private const val ROUNDS = 40
        private val SAMPLES = listOf(
            "",
            "I'm going to the ",
            "Thank you so much for ",
            "Can we meet at the ",
            "Let me know if you ",
            "I'll be there in ",
            "Happy birthday! Hope you have a ",
            "What time does the mo",
            "See you tomor",
            "Sorry, I can't make it to",
            "The meeting has been moved to next ",
            "Do you want to grab ",
            "Please call me when you get ",
            "We should probably leave early to avoid the ",
            "Hello\n",
            "He said (",
            "This is the end.",
            "don'",
        )
    }
}
