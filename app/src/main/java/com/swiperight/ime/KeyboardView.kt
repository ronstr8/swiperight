package com.swiperight.ime

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.content.res.ColorStateList
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import com.swiperight.R

/**
 * A plain QWERTY + symbols keyboard with a three-slot suggestion strip, built
 * in code. It has no prediction logic of its own; it only reports keys to
 * [Listener].
 */
@SuppressLint("ViewConstructor")
class KeyboardView(context: Context, private val listener: Listener) : LinearLayout(context) {

    interface Listener {
        fun onText(text: String)
        fun onBackspace()
        fun onEnter()
        fun onSuggestion(word: String)
    }

    enum class Shift { OFF, ONCE, LOCKED }

    private val handler = Handler(Looper.getMainLooper())
    private val suggestionViews = ArrayList<TextView>(3)
    private val letterKeys = ArrayList<Pair<TextView, String>>()
    private var shiftKey: TextView? = null
    private val keyRows = LinearLayout(context).apply { orientation = VERTICAL }
    private var symbols = false
    private var lastShiftTap = 0L

    var shift: Shift = Shift.OFF
        set(value) {
            field = value
            refreshLetters()
        }

    var enterLabel: String = "⏎"
        set(value) {
            field = value
            buildRows()
        }

    private val keyColor = context.getColor(R.color.key)
    private val modKeyColor = context.getColor(R.color.key_mod)
    private val textColor = context.getColor(R.color.key_text)
    private val accent = context.getColor(R.color.accent)

    init {
        orientation = VERTICAL
        setBackgroundColor(context.getColor(R.color.keyboard_bg))
        val pad = dp(4)
        setPadding(pad, pad, pad, dp(8))
        addView(buildSuggestionStrip())
        addView(keyRows)
        buildRows()
    }

    /** Show up to three words; the middle slot holds the top suggestion. */
    fun setSuggestions(words: List<String>) {
        // Order on screen: [2nd] [1st] [3rd], like most mobile keyboards.
        val slots = arrayOf(words.getOrNull(1), words.getOrNull(0), words.getOrNull(2))
        for ((i, v) in suggestionViews.withIndex()) {
            val w = slots[i]
            v.text = w ?: ""
            v.tag = w
            v.isEnabled = w != null
        }
    }

    private fun buildSuggestionStrip(): View {
        val strip = LinearLayout(context).apply {
            orientation = HORIZONTAL
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(44))
        }
        repeat(3) { i ->
            val tv = TextView(context).apply {
                gravity = Gravity.CENTER
                setTextColor(textColor)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
                if (i == 1) setTypeface(typeface, Typeface.BOLD)
                maxLines = 1
                ellipsize = android.text.TextUtils.TruncateAt.END
                background = ripple(null)
                setOnClickListener {
                    val w = tag as? String ?: return@setOnClickListener
                    performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                    listener.onSuggestion(w)
                }
            }
            strip.addView(tv, LayoutParams(0, LayoutParams.MATCH_PARENT, 1f))
            suggestionViews += tv
            if (i < 2) strip.addView(View(context).apply { setBackgroundColor(modKeyColor) },
                LayoutParams(dp(1), LayoutParams.MATCH_PARENT).apply { setMargins(0, dp(10), 0, dp(10)) })
        }
        return strip
    }

    private fun buildRows() {
        keyRows.removeAllViews()
        letterKeys.clear()
        shiftKey = null
        if (!symbols) {
            row(chars("qwertyuiop"))
            row(chars("asdfghjkl"), sidePad = 0.5f)
            row(listOf(Key.Shift) + chars("zxcvbnm") + Key.Delete)
        } else {
            row(chars("1234567890"))
            row(chars("@#$%&-+()/"))
            row(listOf(Key.Text("=", 1f)) + chars("*\"':;!?") + Key.Delete)
        }
        row(listOf(Key.Mode, Key.Text(",", 1f), Key.Space, Key.Text(".", 1f), Key.Enter))
        refreshLetters()
    }

    private sealed class Key(val weight: Float) {
        class Text(val s: String, w: Float = 1f) : Key(w)
        object Shift : Key(1.5f)
        object Delete : Key(1.5f)
        object Mode : Key(1.5f)
        object Space : Key(5f)
        object Enter : Key(1.5f)
    }

    private fun chars(s: String) = s.map { Key.Text(it.toString()) }

    private fun row(keys: List<Key>, sidePad: Float = 0f) {
        val r = LinearLayout(context).apply { orientation = HORIZONTAL }
        if (sidePad > 0) r.addView(View(context), LayoutParams(0, dp(KEY_H), sidePad))
        for (k in keys) r.addView(makeKey(k), LayoutParams(0, dp(KEY_H), k.weight))
        if (sidePad > 0) r.addView(View(context), LayoutParams(0, dp(KEY_H), sidePad))
        keyRows.addView(r)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun makeKey(k: Key): View {
        val tv = TextView(context).apply {
            gravity = Gravity.CENTER
            setTextColor(textColor)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, if (k is Key.Text) 22f else 16f)
            isSoundEffectsEnabled = false
        }
        val frame = LinearLayout(context).apply {
            setPadding(dp(3), dp(4), dp(3), dp(4))
            addView(tv, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        }
        val mod = k !is Key.Text && k !is Key.Space
        tv.background = ripple(if (k is Key.Enter) accent else if (mod) modKeyColor else keyColor)
        when (k) {
            is Key.Text -> {
                tv.text = k.s
                if (k.s.length == 1 && k.s[0].isLetter()) letterKeys += tv to k.s
                tv.setOnClickListener {
                    tap(tv)
                    val out = if (shift != Shift.OFF && k.s[0].isLetter()) k.s.uppercase() else k.s
                    listener.onText(out)
                    if (shift == Shift.ONCE && k.s[0].isLetter()) shift = Shift.OFF
                }
            }
            Key.Shift -> {
                shiftKey = tv
                tv.setOnClickListener {
                    tap(tv)
                    val now = System.currentTimeMillis()
                    shift = when {
                        shift == Shift.OFF -> if (now - lastShiftTap < DOUBLE_TAP_MS) Shift.LOCKED else Shift.ONCE
                        shift == Shift.ONCE && now - lastShiftTap < DOUBLE_TAP_MS -> Shift.LOCKED
                        else -> Shift.OFF
                    }
                    lastShiftTap = now
                }
            }
            Key.Delete -> {
                tv.text = "⌫"
                // Hold to repeat.
                val repeat = object : Runnable {
                    override fun run() {
                        listener.onBackspace()
                        handler.postDelayed(this, REPEAT_MS)
                    }
                }
                tv.setOnTouchListener { v, e ->
                    when (e.actionMasked) {
                        MotionEvent.ACTION_DOWN -> {
                            v.isPressed = true
                            tap(tv)
                            listener.onBackspace()
                            handler.postDelayed(repeat, REPEAT_DELAY_MS)
                        }
                        MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                            v.isPressed = false
                            handler.removeCallbacks(repeat)
                        }
                    }
                    true
                }
            }
            Key.Mode -> {
                tv.text = if (symbols) "ABC" else "?123"
                tv.setOnClickListener {
                    tap(tv)
                    symbols = !symbols
                    buildRows()
                }
            }
            Key.Space -> {
                tv.text = context.getString(R.string.app_name)
                tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                tv.alpha = 0.8f
                tv.setOnClickListener { tap(tv); listener.onText(" ") }
            }
            Key.Enter -> {
                tv.text = enterLabel
                tv.setOnClickListener { tap(tv); listener.onEnter() }
            }
        }
        return frame
    }

    private fun refreshLetters() {
        for ((tv, s) in letterKeys) tv.text = if (shift != Shift.OFF) s.uppercase() else s
        shiftKey?.text = when (shift) {
            Shift.OFF -> "⇧"
            Shift.ONCE -> "⬆"
            Shift.LOCKED -> "⇪"
        }
    }

    private fun tap(v: View) = v.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)

    private fun ripple(fill: Int?): RippleDrawable {
        val shape = fill?.let {
            GradientDrawable().apply {
                cornerRadius = dp(6).toFloat()
                setColor(it)
            }
        }
        val mask = GradientDrawable().apply { cornerRadius = dp(6).toFloat(); setColor(0xFFFFFFFF.toInt()) }
        return RippleDrawable(ColorStateList.valueOf(0x33FFFFFF), shape, mask)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onDetachedFromWindow() {
        handler.removeCallbacksAndMessages(null)
        super.onDetachedFromWindow()
    }

    companion object {
        private const val KEY_H = 54
        private const val DOUBLE_TAP_MS = 300L
        private const val REPEAT_DELAY_MS = 400L
        private const val REPEAT_MS = 50L
    }
}
