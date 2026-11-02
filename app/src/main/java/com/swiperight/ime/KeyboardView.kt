package com.swiperight.ime

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import com.swiperight.R
import com.swiperight.predict.SwipeInput
import kotlin.math.hypot

/**
 * QWERTY + symbols keyboard with a horizontally scrolling strip of up to nine
 * suggestions and gesture (swipe) input over the letter keys. It has no
 * prediction logic of its own; it only reports to [Listener].
 */
@SuppressLint("ViewConstructor")
class KeyboardView(context: Context, private val listener: Listener) : LinearLayout(context) {

    interface Listener {
        fun onText(text: String)
        fun onBackspace()
        fun onEnter()
        fun onSuggestion(word: String)
        /** A completed swipe across letter keys; [shift] is the shift state when it started. */
        fun onSwipe(swipe: SwipeInput, shift: Shift)
    }

    enum class Shift { OFF, ONCE, LOCKED }

    private val handler = Handler(Looper.getMainLooper())
    private val suggestionViews = ArrayList<TextView>(MAX_SUGGESTIONS)
    private lateinit var stripScroll: HorizontalScrollView
    private val letterKeys = ArrayList<Pair<TextView, String>>()
    private var shiftKey: TextView? = null
    private val keyRows = SwipeLayer(context)
    private var symbols = false
    private var lastShiftTap = 0L

    var shift: Shift = Shift.OFF
        set(value) {
            field = value
            refreshLetters()
        }

    var enterLabel: String = "⏎"
        set(value) {
            if (field == value) return
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

    /**
     * Shows up to nine words, best first, left to right. Three fill the
     * visible width; the rest scroll in from the right.
     */
    fun setSuggestions(words: List<String>) {
        for ((i, v) in suggestionViews.withIndex()) {
            val w = words.getOrNull(i)
            v.text = w ?: ""
            v.tag = w
            v.visibility = if (w != null || i < 3) View.VISIBLE else View.GONE
            v.isEnabled = w != null
        }
        stripScroll.scrollTo(0, 0)
    }

    private fun buildSuggestionStrip(): View {
        stripScroll = HorizontalScrollView(context).apply {
            isHorizontalScrollBarEnabled = false
            overScrollMode = OVER_SCROLL_NEVER
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(44))
        }
        val row = LinearLayout(context).apply { orientation = HORIZONTAL }
        stripScroll.addView(row, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.MATCH_PARENT))
        // A third of the strip's width per slot, so slots 1-3 fill the screen.
        val slotWidth = (resources.displayMetrics.widthPixels - 2 * dp(4)) / 3
        repeat(MAX_SUGGESTIONS) { i ->
            val tv = TextView(context).apply {
                gravity = Gravity.CENTER
                setTextColor(textColor)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
                if (i == 0) setTypeface(typeface, Typeface.BOLD)
                maxLines = 1
                ellipsize = android.text.TextUtils.TruncateAt.END
                setPadding(dp(6), 0, dp(6), 0)
                background = ripple(null)
                setOnClickListener {
                    val w = tag as? String ?: return@setOnClickListener
                    performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                    listener.onSuggestion(w)
                }
            }
            row.addView(tv, LayoutParams(slotWidth, LayoutParams.MATCH_PARENT))
            suggestionViews += tv
        }
        return stripScroll
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
                if (k.s.length == 1 && k.s[0] in 'a'..'z') letterKeys += tv to k.s
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

    /**
     * Holds the key rows. A touch that starts on a letter key and travels more
     * than [SWIPE_START_KEYS] key widths is taken over from the key (the key
     * gets ACTION_CANCEL, so no letter is typed). It is then drawn as a trail
     * and reported as a swipe when the finger lifts. Taps are unaffected.
     */
    private inner class SwipeLayer(context: Context) : LinearLayout(context) {
        private val xs = ArrayList<Float>(256)
        private val ys = ArrayList<Float>(256)
        private var downX = 0f
        private var downY = 0f
        private var downOnLetter = false
        private var swiping = false
        private var startShift = Shift.OFF
        private val trail = Path()
        private val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = accent
            style = Paint.Style.STROKE
            strokeWidth = dp(5).toFloat()
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
            alpha = 170
        }
        private val touchSlop = ViewConfiguration.get(context).scaledTouchSlop

        init {
            orientation = VERTICAL
            setWillNotDraw(false)
        }

        override fun onInterceptTouchEvent(e: MotionEvent): Boolean {
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = e.x; downY = e.y
                    downOnLetter = !symbols && letterAt(e.x, e.y) != null
                    swiping = false
                    xs.clear(); ys.clear()
                    xs += e.x; ys += e.y
                    startShift = shift
                }
                MotionEvent.ACTION_MOVE -> {
                    if (!downOnLetter || e.pointerCount > 1) return false
                    for (h in 0 until e.historySize) { xs += e.getHistoricalX(h); ys += e.getHistoricalY(h) }
                    xs += e.x; ys += e.y
                    val kw = keyWidth()
                    if (hypot(e.x - downX, e.y - downY) > maxOf(touchSlop.toFloat(), kw * SWIPE_START_KEYS)) {
                        swiping = true
                        trail.reset()
                        trail.moveTo(xs[0], ys[0])
                        for (i in 1 until xs.size) trail.lineTo(xs[i], ys[i])
                        invalidate()
                        return true
                    }
                }
            }
            return false
        }

        @SuppressLint("ClickableViewAccessibility")
        override fun onTouchEvent(e: MotionEvent): Boolean {
            if (!swiping) return false
            when (e.actionMasked) {
                MotionEvent.ACTION_MOVE -> {
                    for (h in 0 until e.historySize) {
                        xs += e.getHistoricalX(h); ys += e.getHistoricalY(h)
                        trail.lineTo(e.getHistoricalX(h), e.getHistoricalY(h))
                    }
                    xs += e.x; ys += e.y
                    trail.lineTo(e.x, e.y)
                    invalidate()
                }
                MotionEvent.ACTION_UP -> {
                    swiping = false
                    trail.reset()
                    invalidate()
                    buildSwipe()?.let { listener.onSwipe(it, startShift) }
                }
                MotionEvent.ACTION_CANCEL -> {
                    swiping = false
                    trail.reset()
                    invalidate()
                }
            }
            return true
        }

        override fun dispatchDraw(canvas: Canvas) {
            super.dispatchDraw(canvas)
            if (swiping) canvas.drawPath(trail, trailPaint)
        }

        private fun buildSwipe(): SwipeInput? {
            if (xs.size < 2 || letterKeys.isEmpty()) return null
            val kx = FloatArray(26) { Float.NaN }
            val ky = FloatArray(26) { Float.NaN }
            for ((tv, s) in letterKeys) {
                val (cx, cy) = centerOf(tv)
                kx[s[0] - 'a'] = cx; ky[s[0] - 'a'] = cy
            }
            return SwipeInput(xs.toFloatArray(), ys.toFloatArray(), kx, ky, keyWidth())
        }

        private fun keyWidth(): Float = letterKeys.firstOrNull()?.first?.let { (it.parent as View).width.toFloat() } ?: 1f

        /** Centre of a key in this layer's coordinates. */
        private fun centerOf(v: View): Pair<Float, Float> {
            val frame = v.parent as View
            val row = frame.parent as View
            return (row.left + frame.left + frame.width / 2f) to (row.top + frame.top + frame.height / 2f)
        }

        private fun letterAt(x: Float, y: Float): String? {
            for ((tv, s) in letterKeys) {
                val frame = tv.parent as View
                val row = frame.parent as View
                val l = row.left + frame.left; val t = row.top + frame.top
                if (x >= l && x < l + frame.width && y >= t && y < t + frame.height) return s
            }
            return null
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
        const val MAX_SUGGESTIONS = 9
        private const val KEY_H = 54
        private const val DOUBLE_TAP_MS = 300L
        private const val REPEAT_DELAY_MS = 400L
        private const val REPEAT_MS = 50L
        /** Travel (in key widths) before a touch on a letter becomes a swipe. */
        private const val SWIPE_START_KEYS = 0.6f
    }
}
