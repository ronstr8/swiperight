package com.swiperight.ime

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Paint
import android.graphics.Typeface
import android.util.TypedValue
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.GridView
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import com.swiperight.R

/**
 * Emoji picker: category tabs, a scrolling grid and recently used emoji.
 *
 * The emoji list (assets/emoji.txt) comes from Unicode's emoji-test.txt
 * (tools/make_emoji.py). Emoji this device's font cannot draw are hidden
 * (Paint.hasGlyph), so older phones never show tofu boxes. Recents are kept
 * in SharedPreferences; nothing else is stored.
 */
@SuppressLint("ViewConstructor")
class EmojiPanel(
    context: Context,
    private val heightPx: Int,
    private val onEmoji: (String) -> Unit,
) : LinearLayout(context) {

    private class Group(val name: String, val icon: String, val emoji: List<String>)

    private val prefs = context.getSharedPreferences("emoji", Context.MODE_PRIVATE)
    private val textColor = context.getColor(R.color.key_text)
    private val accent = context.getColor(R.color.accent)
    private val grid = GridView(context)
    private val tabs = ArrayList<TextView>()
    private val adapter = EmojiAdapter()
    private val groups: List<Group> = loadGroups(context)
    private var selected = 0

    init {
        orientation = VERTICAL
        val tabRow = LinearLayout(context).apply { orientation = HORIZONTAL }
        val tabScroll = HorizontalScrollView(context).apply {
            isHorizontalScrollBarEnabled = false
            addView(tabRow)
        }
        addView(tabScroll, LayoutParams(LayoutParams.MATCH_PARENT, dp(TAB_H)))
        groups.forEachIndexed { i, g ->
            val tv = TextView(context).apply {
                text = g.icon
                gravity = Gravity.CENTER
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)
                contentDescription = g.name
                setOnClickListener { select(i) }
            }
            tabRow.addView(tv, LayoutParams(dp(44), LayoutParams.MATCH_PARENT))
            tabs += tv
        }
        grid.apply {
            numColumns = GridView.AUTO_FIT
            columnWidth = dp(46)
            stretchMode = GridView.STRETCH_COLUMN_WIDTH
            isVerticalScrollBarEnabled = false
            adapter = this@EmojiPanel.adapter
            setOnItemClickListener { _, v, pos, _ ->
                val e = this@EmojiPanel.adapter.getItem(pos)
                v.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                remember(e)
                onEmoji(e)
            }
        }
        addView(grid, LayoutParams(LayoutParams.MATCH_PARENT, heightPx - dp(TAB_H)))
        select(if (recents().isEmpty()) 1.coerceAtMost(groups.size - 1) else 0)
    }

    private fun select(i: Int) {
        selected = i
        tabs.forEachIndexed { j, t ->
            t.alpha = if (j == i) 1f else 0.45f
            t.setBackgroundColor(if (j == i) (accent and 0x00FFFFFF) or 0x33000000 else 0)
        }
        adapter.items = if (i == 0) recents() else groups[i].emoji
        adapter.notifyDataSetChanged()
        grid.setSelection(0)
    }

    private fun recents(): List<String> =
        prefs.getString(KEY_RECENT, "")!!.split(' ').filter { it.isNotEmpty() }

    private fun remember(e: String) {
        val list = (listOf(e) + recents().filter { it != e }).take(MAX_RECENT)
        prefs.edit().putString(KEY_RECENT, list.joinToString(" ")).apply()
        if (selected == 0) { adapter.items = list; adapter.notifyDataSetChanged() }
    }

    private inner class EmojiAdapter : BaseAdapter() {
        var items: List<String> = emptyList()
        override fun getCount() = items.size
        override fun getItem(position: Int) = items[position]
        override fun getItemId(position: Int) = position.toLong()
        override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {
            val tv = (convertView as? TextView) ?: TextView(context).apply {
                gravity = Gravity.CENTER
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 26f)
                setTextColor(textColor)
                typeface = Typeface.DEFAULT
                layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(46))
            }
            tv.text = items[position]
            return tv
        }
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    companion object {
        private const val TAB_H = 38
        private const val KEY_RECENT = "recent"
        private const val MAX_RECENT = 32

        private val ICONS = mapOf(
            "Smileys & Emotion" to "😀", "People & Body" to "👋", "Animals & Nature" to "🐻",
            "Food & Drink" to "🍔", "Travel & Places" to "🚗", "Activities" to "⚽",
            "Objects" to "💡", "Symbols" to "❤️", "Flags" to "🏳️",
        )

        /** Parsed once per process: ~1.9k strings, filtered to what the font can draw. */
        @Volatile private var cache: List<Group>? = null

        private fun loadGroups(context: Context): List<Group> {
            cache?.let { return it }
            val paint = Paint()
            val out = ArrayList<Group>()
            out += Group("Recent", "🕘", emptyList())
            context.assets.open("emoji.txt").bufferedReader().useLines { lines ->
                for (line in lines) {
                    val tab = line.indexOf('\t')
                    if (tab < 0) continue
                    val name = line.substring(0, tab)
                    val emoji = line.substring(tab + 1).split(' ').filter { it.isNotEmpty() && paint.hasGlyph(it) }
                    if (emoji.isNotEmpty()) out += Group(name, ICONS[name] ?: emoji[0], emoji)
                }
            }
            return out.also { cache = it }
        }
    }
}
