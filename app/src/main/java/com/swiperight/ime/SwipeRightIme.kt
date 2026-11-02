package com.swiperight.ime

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Process
import android.text.InputType
import android.text.TextUtils
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import com.swiperight.predict.ContextBuilder
import com.swiperight.predict.Predictor
import com.swiperight.predict.SwipeInput

/**
 * The keyboard. Prediction work happens on one background thread that owns
 * the [Predictor]. The UI thread only sends text snapshots and shows results.
 *
 * Latency hygiene: each request carries a generation number. The worker skips
 * any request that is no longer the newest, and the UI ignores results for
 * text that has changed since. Fast typing therefore never queues up stale
 * inferences.
 *
 * Privacy: no prediction (and no text read) in password, number, phone,
 * date or `TYPE_TEXT_FLAG_NO_SUGGESTIONS` fields. Typed text is never logged
 * or stored.
 *
 * Memory: the model is loaded on first use. It is released when the keyboard
 * has been hidden for [IDLE_RELEASE_MS], on memory pressure, and never loaded
 * at all on low-RAM devices.
 */
class SwipeRightIme : InputMethodService(), KeyboardView.Listener {

    private val main = Handler(Looper.getMainLooper())
    private lateinit var workerThread: HandlerThread
    private lateinit var worker: Handler

    // Owned by the worker thread.
    private var predictor: Predictor? = null
    private var loadFailed = false

    @Volatile private var generation = 0L
    private var keyboard: KeyboardView? = null
    private var predictionsEnabled = false
    private var swipeEnabled = false
    private var lastCommitWasSuggestion = false

    /** A swiped word is in the editor's composing region; alternates are on the strip. */
    private var composingSwipe = false
    /** Bumped on every key and swipe; a swipe result is applied only if nothing happened since. */
    private var inputSerial = 0L

    private val releaseRunnable = Runnable { worker.post { releasePredictor("idle") } }

    override fun onCreate() {
        super.onCreate()
        workerThread = HandlerThread("swiperight-predictor", Process.THREAD_PRIORITY_DEFAULT).apply { start() }
        worker = Handler(workerThread.looper)
    }

    override fun onCreateInputView(): View = KeyboardView(this, this).also { keyboard = it }

    override fun onStartInputView(info: EditorInfo, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        main.removeCallbacks(releaseRunnable)
        predictionsEnabled = wantsPredictions(info) && !isLowRam()
        swipeEnabled = allowsSwipe(info) && !isLowRam()
        lastCommitWasSuggestion = false
        composingSwipe = false
        keyboard?.enterLabel = enterLabelFor(info)
        keyboard?.setSuggestions(emptyList())
        updateAutoShift()
        if (predictionsEnabled || swipeEnabled) {
            worker.post { ensurePredictor() } // warm up while the user starts typing
            requestPrediction()
        }
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        super.onFinishInputView(finishingInput)
        generation++
        keyboard?.setSuggestions(emptyList())
        main.postDelayed(releaseRunnable, IDLE_RELEASE_MS)
    }

    override fun onUpdateSelection(
        oldSelStart: Int, oldSelEnd: Int, newSelStart: Int, newSelEnd: Int,
        candidatesStart: Int, candidatesEnd: Int,
    ) {
        super.onUpdateSelection(oldSelStart, oldSelEnd, newSelStart, newSelEnd, candidatesStart, candidatesEnd)
        updateAutoShift()
        if (composingSwipe) {
            // Keep showing swipe alternates while the swiped word is still composing.
            if (candidatesStart >= 0 && newSelStart == candidatesEnd) return
            composingSwipe = false // the editor or the user moved on
        }
        if (newSelStart == newSelEnd) requestPrediction() else keyboard?.setSuggestions(emptyList())
    }

    // ---- KeyboardView.Listener -------------------------------------------------

    override fun onText(text: String) {
        val ic = currentInputConnection ?: return
        inputSerial++
        if (composingSwipe) {
            // Accept the swiped word. A letter starts a new word, so it gets a space.
            ic.finishComposingText()
            composingSwipe = false
            when {
                text[0].isLetterOrDigit() -> ic.commitText(" $text", 1)
                text.length == 1 && text[0] in ".,!?;:" -> ic.commitText("$text ", 1)
                else -> ic.commitText(text, 1)
            }
            lastCommitWasSuggestion = false
            return
        }
        if (text.length == 1 && text[0] in I_TERMINATORS) capitalizeStandaloneI(ic)
        if (lastCommitWasSuggestion && text.length == 1 && text[0] in ".,!?;:") {
            // "word |" + "." -> "word. |": swap the auto-inserted space.
            if (ic.getTextBeforeCursor(1, 0) == " ") ic.deleteSurroundingText(1, 0)
            ic.commitText("$text ", 1)
        } else {
            ic.commitText(text, 1)
        }
        lastCommitWasSuggestion = false
    }

    override fun onBackspace() {
        val ic = currentInputConnection ?: return
        inputSerial++
        lastCommitWasSuggestion = false
        if (composingSwipe) {
            // Backspace right after a swipe removes the whole swiped word.
            ic.commitText("", 1)
            composingSwipe = false
            return
        }
        val sel = ic.getSelectedText(0)
        if (!sel.isNullOrEmpty()) ic.commitText("", 1)
        else sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL) // editor handles surrogates/graphemes
    }

    override fun onEnter() {
        val ic = currentInputConnection ?: return
        inputSerial++
        lastCommitWasSuggestion = false
        if (composingSwipe) {
            ic.finishComposingText()
            composingSwipe = false
        } else {
            capitalizeStandaloneI(ic)
        }
        val info = currentInputEditorInfo
        val action = info?.imeOptions?.and(EditorInfo.IME_MASK_ACTION) ?: EditorInfo.IME_ACTION_NONE
        val noEnterAction = (info?.imeOptions ?: 0) and EditorInfo.IME_FLAG_NO_ENTER_ACTION != 0
        val multiLine = (info?.inputType ?: 0) and InputType.TYPE_TEXT_FLAG_MULTI_LINE != 0
        if (!multiLine && !noEnterAction && action != EditorInfo.IME_ACTION_NONE && action != EditorInfo.IME_ACTION_UNSPECIFIED) {
            ic.performEditorAction(action)
        } else {
            sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
        }
    }

    override fun onSuggestion(word: String) {
        val ic = currentInputConnection ?: return
        inputSerial++
        if (composingSwipe) {
            // Tapped a swipe alternate: it replaces the swiped word.
            ic.beginBatchEdit()
            ic.setComposingText(word, 1)
            ic.finishComposingText()
            ic.commitText(" ", 1)
            ic.endBatchEdit()
            composingSwipe = false
            lastCommitWasSuggestion = true
            return
        }
        // Re-derive the slot from the *current* text: the user may have typed
        // more since these suggestions were computed.
        val before = ic.getTextBeforeCursor(CONTEXT_CHARS, 0) ?: ""
        val ctx = ContextBuilder.build(before)
        ic.beginBatchEdit()
        if (ctx.prefix.isNotEmpty()) ic.deleteSurroundingText(ctx.prefix.length, 0)
        ic.commitText((if (ctx.insertSpace) " " else "") + word + " ", 1)
        ic.endBatchEdit()
        lastCommitWasSuggestion = true
    }

    override fun onSwipe(swipe: SwipeInput, shift: KeyboardView.Shift) {
        if (!swipeEnabled) return
        val ic = currentInputConnection ?: return
        if (composingSwipe) {
            ic.finishComposingText() // the previous swiped word is accepted as is
            composingSwipe = false
        }
        lastCommitWasSuggestion = false
        val serial = ++inputSerial
        val text = ic.getTextBeforeCursor(CONTEXT_CHARS, 0)?.toString() ?: ""
        generation++ // drop any pending suggestion result
        worker.post {
            val p = ensurePredictor() ?: return@post
            val result = try {
                p.decodeSwipe(text, swipe, KeyboardView.MAX_SUGGESTIONS)
            } catch (e: Exception) {
                Log.e(TAG, "swipe decode failed", e)
                return@post
            }
            main.post {
                if (serial != inputSerial || result.words.isEmpty()) return@post
                val conn = currentInputConnection ?: return@post
                val words = result.words.map { applyShift(it, shift) }
                generation++
                conn.beginBatchEdit()
                if (result.context.insertSpace) conn.commitText(" ", 1)
                conn.setComposingText(words[0], 1)
                conn.endBatchEdit()
                composingSwipe = true
                keyboard?.setSuggestions(words)
            }
        }
    }

    // ---- prediction ------------------------------------------------------------

    private fun requestPrediction() {
        if (!predictionsEnabled) return
        val ic = currentInputConnection ?: return
        val text = ic.getTextBeforeCursor(CONTEXT_CHARS, 0)?.toString() ?: return
        val gen = ++generation
        worker.post {
            if (gen != generation) return@post // superseded by a newer keystroke
            val p = ensurePredictor() ?: return@post
            val result = try {
                p.predict(text, KeyboardView.MAX_SUGGESTIONS)
            } catch (e: Exception) {
                Log.e(TAG, "prediction failed", e)
                return@post
            }
            main.post { if (gen == generation) keyboard?.setSuggestions(result.words) }
        }
    }

    /** Worker thread only. */
    private fun ensurePredictor(): Predictor? {
        predictor?.let { if (!it.isClosed) return it }
        if (loadFailed) return null
        return try {
            Predictor.create(applicationContext).also { predictor = it }
        } catch (e: Throwable) {
            // Missing/mismatched assets or OOM: keep typing working without suggestions.
            Log.e(TAG, "predictor unavailable; suggestions disabled", e)
            loadFailed = true
            null
        }
    }

    /** Worker thread only. */
    private fun releasePredictor(reason: String) {
        predictor?.let {
            it.close()
            Log.i(TAG, "released model ($reason)")
        }
        predictor = null
    }

    @Deprecated("Deprecated in Java")
    override fun onTrimMemory(level: Int) {
        @Suppress("DEPRECATION")
        super.onTrimMemory(level)
        // Hidden keyboard (UI_HIDDEN and up) or the system running out of memory.
        @Suppress("DEPRECATION")
        if (level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN || level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW) {
            if (!isInputViewShown || level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL) {
                worker.post { releasePredictor("trim level $level") }
            }
        }
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        worker.post { releasePredictor("destroy") }
        workerThread.quitSafely()
        super.onDestroy()
    }

    // ---- helpers ---------------------------------------------------------------

    private fun updateAutoShift() {
        val kb = keyboard ?: return
        if (kb.shift == KeyboardView.Shift.LOCKED) return
        val ic = currentInputConnection ?: return
        val info = currentInputEditorInfo ?: return
        val caps = info.inputType != InputType.TYPE_NULL && ic.getCursorCapsMode(info.inputType) != 0
        kb.shift = if (caps) KeyboardView.Shift.ONCE else KeyboardView.Shift.OFF
    }

    /** "i" typed as a whole word becomes "I" once the word ends (space, punctuation, enter). */
    private fun capitalizeStandaloneI(ic: android.view.inputmethod.InputConnection) {
        val before = ic.getTextBeforeCursor(2, 0)?.toString() ?: return
        val standalone = before == "i" ||
            (before.length == 2 && before[1] == 'i' && (before[0].isWhitespace() || before[0] in "(\"“‘[{"))
        if (!standalone) return
        ic.beginBatchEdit()
        ic.deleteSurroundingText(1, 0)
        ic.commitText("I", 1)
        ic.endBatchEdit()
    }

    private fun applyShift(w: String, shift: KeyboardView.Shift): String = when (shift) {
        KeyboardView.Shift.OFF -> w
        KeyboardView.Shift.ONCE -> w.replaceFirstChar { it.titlecase() }
        KeyboardView.Shift.LOCKED -> w.uppercase()
    }

    private fun isLowRam(): Boolean =
        (getSystemService(ACTIVITY_SERVICE) as ActivityManager).isLowRamDevice

    companion object {
        private const val TAG = "SwipeRightIme"
        /** Read a bit more than the model window so ContextBuilder can tell a cut from the start. */
        private const val CONTEXT_CHARS = ContextBuilder.WINDOW_CHARS * 2
        private const val IDLE_RELEASE_MS = 5 * 60 * 1000L

        /** Characters that end a word and trigger "i" -> "I". */
        private const val I_TERMINATORS = " .,!?;:'’)\n"

        /** Swipe needs letters: any text field except passwords. */
        fun allowsSwipe(info: EditorInfo): Boolean {
            val type = info.inputType
            if (type and InputType.TYPE_MASK_CLASS != InputType.TYPE_CLASS_TEXT) return false
            val v = type and InputType.TYPE_MASK_VARIATION
            return v != InputType.TYPE_TEXT_VARIATION_PASSWORD &&
                v != InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD &&
                v != InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD
        }

        fun wantsPredictions(info: EditorInfo): Boolean {
            val type = info.inputType
            if (type == InputType.TYPE_NULL) return false
            if (type and InputType.TYPE_MASK_CLASS != InputType.TYPE_CLASS_TEXT) return false
            val variation = type and InputType.TYPE_MASK_VARIATION
            if (variation == InputType.TYPE_TEXT_VARIATION_PASSWORD ||
                variation == InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
                variation == InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD ||
                variation == InputType.TYPE_TEXT_VARIATION_URI ||
                variation == InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS ||
                variation == InputType.TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS
            ) return false
            if (type and InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS != 0) return false
            return true
        }

        fun enterLabelFor(info: EditorInfo): String {
            if (info.inputType and InputType.TYPE_TEXT_FLAG_MULTI_LINE != 0) return "⏎"
            if (!TextUtils.isEmpty(info.actionLabel)) return info.actionLabel.toString()
            return when (info.imeOptions and EditorInfo.IME_MASK_ACTION) {
                EditorInfo.IME_ACTION_GO -> "Go"
                EditorInfo.IME_ACTION_SEARCH -> "🔍"
                EditorInfo.IME_ACTION_SEND -> "Send"
                EditorInfo.IME_ACTION_NEXT -> "Next"
                EditorInfo.IME_ACTION_DONE -> "Done"
                else -> "⏎"
            }
        }
    }
}
