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
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.PopupWindow
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
        /** The path so far, each time the finger enters a new key. Answer with [showSwipePreview]. */
        fun onSwipeProgress(swipe: SwipeInput, shift: Shift) {}
    }

    enum class Shift { OFF, ONCE, LOCKED }

    private val handler = Handler(Looper.getMainLooper())
    private val suggestionViews = ArrayList<TextView>(MAX_SUGGESTIONS)
    private lateinit var stripScroll: HorizontalScrollView
    private val letterKeys = ArrayList<Pair<TextView, String>>()
    private var shiftKey: TextView? = null
    private enum class Mode { LETTERS, SYMBOLS, EMOJI }
    private var mode = Mode.LETTERS
    private val symbols get() = mode != Mode.LETTERS
    private lateinit var traceLabel: TextView
    private var punctPopup: PopupWindow? = null
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
    // Created after the colours above: SwipeLayer's trail paint reads `accent` at construction.
    private val keyRows = SwipeLayer(context)

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
        // The word the in-progress swipe currently decodes to, shown over the suggestions.
        traceLabel = TextView(context).apply {
            gravity = Gravity.CENTER
            setTextColor(accent)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            setTypeface(typeface, Typeface.BOLD)
            letterSpacing = 0.12f
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.START
            setBackgroundColor(context.getColor(R.color.keyboard_bg))
            visibility = View.GONE
        }
        return FrameLayout(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(44))
            addView(stripScroll, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
            addView(traceLabel, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        }
    }

    private var swipeActive = false

    private fun showTrace(active: Boolean) {
        swipeActive = active
        traceLabel.text = ""
        traceLabel.visibility = if (active) View.VISIBLE else View.GONE
    }

    /**
     * Live decode of the swipe in progress. Ignored once the finger has
     * lifted, so a late result can't overwrite the final word. An empty
     * result keeps the previous word rather than flickering.
     */
    fun showSwipePreview(word: String?) {
        if (!swipeActive || word.isNullOrEmpty()) return
        traceLabel.text = word
    }

    private fun buildRows() {
        keyRows.removeAllViews()
        letterKeys.clear()
        shiftKey = null
        when (mode) {
            Mode.LETTERS -> {
                row(chars("qwertyuiop"))
                row(chars("asdfghjkl"), sidePad = 0.5f)
                row(listOf(Key.Shift) + chars("zxcvbnm") + Key.Delete)
            }
            Mode.SYMBOLS -> {
                row(chars("1234567890"))
                row(chars("@#$%&-+()/"))
                row(listOf(Key.Text("=", 1f)) + chars("*\"':;!?") + Key.Delete)
            }
            Mode.EMOJI -> {
                keyRows.addView(EmojiPanel(context, dp(KEY_H) * 3) { listener.onText(it) },
                    LayoutParams(LayoutParams.MATCH_PARENT, dp(KEY_H) * 3))
                row(listOf(Key.Mode, Key.Text(",", 1f), Key.Space, Key.Text(".", 1f), Key.Delete))
                refreshLetters()
                return
            }
        }
        row(listOf(Key.Mode, Key.Emoji, Key.Text(",", 1f), Key.Space, Key.Text(".", 1f), Key.Enter))
        refreshLetters()
    }

    private sealed class Key(val weight: Float) {
        class Text(val s: String, w: Float = 1f) : Key(w)
        object Shift : Key(1.5f)
        object Delete : Key(1.5f)
        object Mode : Key(1.5f)
        object Space : Key(4f)
        object Emoji : Key(1f)
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
                if (k.s == "." && mode != Mode.SYMBOLS) {
                    attachPunctuationPanel(tv)
                    return frame
                }
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
                tv.text = if (mode == Mode.LETTERS) "?123" else "ABC"
                tv.setOnClickListener {
                    tap(tv)
                    mode = if (mode == Mode.LETTERS) Mode.SYMBOLS else Mode.LETTERS
                    buildRows()
                }
            }
            Key.Emoji -> {
                tv.text = "😊"
                tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)
                tv.contentDescription = "Emoji"
                tv.setOnClickListener {
                    tap(tv)
                    mode = Mode.EMOJI
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

    /**
     * Period key: tap for ".", hold to open a punctuation panel. While holding,
     * slide onto a character and release to type it; release elsewhere and
     * the panel stays open for a tap (touch outside to dismiss).
     */
    @SuppressLint("ClickableViewAccessibility")
    private fun attachPunctuationPanel(key: TextView) {
        var opened = false
        var cells: List<TextView> = emptyList()
        var hot: TextView? = null
        val open = Runnable {
            opened = true
            tap(key)
            cells = showPunctuationPanel(key)
        }
        fun cellAt(rawX: Float, rawY: Float): TextView? = cells.firstOrNull { c ->
            val loc = IntArray(2).also { c.getLocationOnScreen(it) }
            rawX >= loc[0] && rawX < loc[0] + c.width && rawY >= loc[1] && rawY < loc[1] + c.height
        }
        key.setOnTouchListener { v, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    opened = false; hot = null
                    v.isPressed = true
                    handler.postDelayed(open, ViewConfiguration.getLongPressTimeout().toLong())
                }
                MotionEvent.ACTION_MOVE -> if (opened) {
                    val c = cellAt(e.rawX, e.rawY)
                    if (c !== hot) {
                        hot?.isPressed = false
                        c?.isPressed = true
                        hot = c
                    }
                }
                MotionEvent.ACTION_UP -> {
                    v.isPressed = false
                    handler.removeCallbacks(open)
                    if (!opened) {
                        tap(key); listener.onText(".")
                    } else {
                        val c = cellAt(e.rawX, e.rawY)
                        if (c != null) { listener.onText(c.text.toString()); punctPopup?.dismiss() }
                    }
                }
                MotionEvent.ACTION_CANCEL -> {
                    v.isPressed = false
                    handler.removeCallbacks(open)
                }
            }
            true
        }
    }

    private fun showPunctuationPanel(anchor: View): List<TextView> {
        punctPopup?.dismiss()
        val cellW = dp(42); val cellH = dp(48)
        val panel = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(dp(4), dp(4), dp(4), dp(4))
            background = GradientDrawable().apply {
                cornerRadius = dp(10).toFloat()
                setColor(modKeyColor)
            }
            elevation = dp(6).toFloat()
        }
        val cells = ArrayList<TextView>()
        for (rowChars in PUNCTUATION) {
            val r = LinearLayout(context).apply { orientation = HORIZONTAL }
            for (ch in rowChars) {
                val c = TextView(context).apply {
                    text = ch
                    gravity = Gravity.CENTER
                    setTextColor(textColor)
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
                    background = ripple(keyColor)
                    setOnClickListener {
                        tap(this); listener.onText(ch); punctPopup?.dismiss()
                    }
                }
                r.addView(c, LayoutParams(cellW, cellH).apply { setMargins(dp(2), dp(2), dp(2), dp(2)) })
                cells += c
            }
            panel.addView(r)
        }
        val popup = PopupWindow(panel, LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT, false).apply {
            isOutsideTouchable = true
            setOnDismissListener { punctPopup = null }
        }
        panel.measure(MeasureSpec.UNSPECIFIED, MeasureSpec.UNSPECIFIED)
        val loc = IntArray(2).also { anchor.getLocationInWindow(it) }
        val myLoc = IntArray(2).also { getLocationInWindow(it) }
        // Above the key, right-aligned with it, kept inside the keyboard.
        val x = (loc[0] + anchor.width - panel.measuredWidth).coerceIn(myLoc[0], myLoc[0] + width - panel.measuredWidth)
        val y = (loc[1] - panel.measuredHeight - dp(4)).coerceAtLeast(myLoc[1])
        popup.showAtLocation(anchor, Gravity.NO_GRAVITY, x, y)
        punctPopup = popup
        return cells
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
        private var lastKey: String? = null
        private var litKey: TextView? = null

        /** Light the key under the finger; on entering a new key, ask for a live decode. */
        private fun track(x: Float, y: Float, report: Boolean = true) {
            val hit = letterKeyAt(x, y)
            if (hit != null && hit.second != lastKey) {
                lastKey = hit.second
                if (report && xs.size >= 2) buildSwipe()?.let { listener.onSwipeProgress(it, startShift) }
            }
            val tv = hit?.first
            if (tv !== litKey) {
                litKey?.isPressed = false
                tv?.isPressed = true
                litKey = tv
            }
        }

        private fun endTrace() {
            litKey?.isPressed = false
            litKey = null
            lastKey = null
            showTrace(false)
        }

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
                        showTrace(true)
                        lastKey = null
                        for (i in xs.indices) track(xs[i], ys[i], report = false)
                        buildSwipe()?.let { listener.onSwipeProgress(it, startShift) }
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
                    track(e.x, e.y)
                    invalidate()
                }
                MotionEvent.ACTION_UP -> {
                    swiping = false
                    endTrace()
                    trail.reset()
                    invalidate()
                    buildSwipe()?.let { listener.onSwipe(it, startShift) }
                }
                MotionEvent.ACTION_CANCEL -> {
                    swiping = false
                    endTrace()
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

        private fun letterAt(x: Float, y: Float): String? = letterKeyAt(x, y)?.second

        private fun letterKeyAt(x: Float, y: Float): Pair<TextView, String>? {
            for (pair in letterKeys) {
                val frame = pair.first.parent as View
                val row = frame.parent as View
                val l = row.left + frame.left; val t = row.top + frame.top
                if (x >= l && x < l + frame.width && y >= t && y < t + frame.height) return pair
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

    /** Back to letters (e.g. when a new field gets focus). */
    fun resetMode() {
        punctPopup?.dismiss()
        if (mode != Mode.LETTERS) { mode = Mode.LETTERS; buildRows() }
    }

    override fun onDetachedFromWindow() {
        punctPopup?.dismiss()
        handler.removeCallbacksAndMessages(null)
        super.onDetachedFromWindow()
    }

    companion object {
        const val MAX_SUGGESTIONS = 9
        private const val KEY_H = 54
        private const val DOUBLE_TAP_MS = 300L
        private const val REPEAT_DELAY_MS = 400L
        private const val REPEAT_MS = 50L
        /** Period-key long-press panel. */
        private val PUNCTUATION = listOf(
            listOf("!", "?", ",", "'", "\"", ":", ";"),
            listOf("-", "(", ")", "@", "&", "/", "…"),
        )
        /** Travel (in key widths) before a touch on a letter becomes a swipe. */
        private const val SWIPE_START_KEYS = 0.6f
    }
}
