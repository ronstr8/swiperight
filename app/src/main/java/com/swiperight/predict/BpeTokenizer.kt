package com.swiperight.predict

import java.util.regex.Pattern

/**
 * Byte-level BPE tokenizer (GPT-2 scheme) built from `vocab`, `merges` and
 * `special_tokens`.
 *
 * Pipeline: optional special-token split -> GPT-2 pre-tokenizer regex -> UTF-8
 * bytes -> printable byte alphabet (`Ġ` = space) -> greedy lowest-rank merges ->
 * ids.
 *
 * Memory: symbols are vocab ids during merging, and the merge table is a
 * primitive open-addressing map keyed by (leftId, rightId). That is about
 * 1.5 MB for GPT-2's 50k merges, versus roughly 6 MB for a
 * `HashMap<String, Int>`.
 *
 * Unknown input: byte-level BPE can encode any byte, so with GPT-2's vocab
 * nothing is unknown. For a custom vocab that lacks some byte symbols, a
 * missing byte maps to the unk token if `special_tokens` defines one in the
 * vocab, and is otherwise dropped. Dropped bytes are counted in [droppedBytes].
 *
 * Not thread-safe (merge cache). Confine each instance to one thread.
 */
class BpeTokenizer(
    vocab: Map<String, Int>,
    merges: List<Pair<String, String>>,
    specialTokens: Set<String>,
) {
    val vocabSize: Int
    private val idToToken: Array<String>
    private val tokenToId: HashMap<String, Int> = HashMap(vocab)
    private val byteToId = IntArray(256) { -1 }
    private val mergeTable: MergeTable
    private val specials: List<String>
    private val specialPattern: Pattern?
    private val unkId: Int

    /** Bytes that could not be represented and were dropped (custom vocabs only). */
    var droppedBytes: Long = 0
        private set

    /** Merges skipped at load because a component/result is missing from vocab. */
    val skippedMerges: Int

    init {
        require(vocab.isNotEmpty()) { "empty vocab" }
        vocabSize = (vocab.values.maxOrNull() ?: -1) + 1
        val inv = arrayOfNulls<String>(vocabSize)
        for ((tok, id) in vocab) {
            require(id >= 0) { "negative id for $tok" }
            require(inv[id] == null) { "duplicate id $id (${inv[id]} / $tok)" }
            inv[id] = tok
        }
        idToToken = Array(vocabSize) { inv[it] ?: "" }

        for (b in 0 until 256) {
            byteToId[b] = vocab[BYTE_TO_UNICODE[b].toString()] ?: -1
        }

        mergeTable = MergeTable(merges.size)
        var skipped = 0
        merges.forEachIndexed { rank, (a, b) ->
            val ia = vocab[a]
            val ib = vocab[b]
            val im = vocab[a + b]
            if (ia == null || ib == null || im == null) skipped++ else mergeTable.putIfAbsent(ia, ib, rank, im)
        }
        skippedMerges = skipped

        specials = specialTokens.filter { it.isNotEmpty() && vocab.containsKey(it) }
            .sortedByDescending { it.length }
        specialPattern = if (specials.isEmpty()) null
        else Pattern.compile(specials.joinToString("|") { Pattern.quote(it) })
        unkId = UNK_NAMES.firstNotNullOfOrNull { n -> if (n in specialTokens) vocab[n] else null } ?: -1
    }

    fun tokenId(token: String): Int? = tokenToId[token]

    fun isSpecial(id: Int): Boolean = id in 0 until vocabSize && idToToken[id] in specials

    /** Raw vocab string for [id], in the byte-level alphabet (e.g. "Ġhello"). */
    fun token(id: Int): String = idToToken[id]

    /**
     * Encodes [text]. Special-token strings typed by the user are encoded as
     * plain text unless [allowSpecial] is set, so typed "<|endoftext|>" cannot
     * act as a control token.
     */
    fun encode(text: String, allowSpecial: Boolean = false): IntArray {
        val out = IntList()
        if (allowSpecial && specialPattern != null) {
            val m = specialPattern.matcher(text)
            var last = 0
            while (m.find()) {
                encodeOrdinary(text.substring(last, m.start()), out)
                out.add(tokenToId.getValue(m.group()))
                last = m.end()
            }
            encodeOrdinary(text.substring(last), out)
        } else {
            encodeOrdinary(text, out)
        }
        return out.toArray()
    }

    /** Decodes ids to text. Invalid UTF-8 (e.g. half an emoji) becomes U+FFFD. */
    fun decode(ids: IntArray): String {
        val bytes = java.io.ByteArrayOutputStream()
        for (id in ids) {
            if (id !in 0 until vocabSize) continue
            for (ch in idToToken[id]) {
                val b = UNICODE_TO_BYTE[ch.code]
                if (b >= 0) bytes.write(b) else bytes.write(ch.toString().toByteArray(Charsets.UTF_8))
            }
        }
        return bytes.toString(Charsets.UTF_8.name())
    }

    /** Decoded bytes of a single token, or null if it is not valid UTF-8 on its own. */
    fun decodeTokenStrict(id: Int): String? {
        val tok = idToToken[id]
        val buf = ByteArray(tok.length)
        for ((i, ch) in tok.withIndex()) {
            val b = if (ch.code < UNICODE_TO_BYTE.size) UNICODE_TO_BYTE[ch.code] else -1
            if (b < 0) return null
            buf[i] = b.toByte()
        }
        val dec = Charsets.UTF_8.newDecoder()
            .onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
            .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
        return try {
            dec.decode(java.nio.ByteBuffer.wrap(buf)).toString()
        } catch (e: java.nio.charset.CharacterCodingException) {
            null
        }
    }

    private fun encodeOrdinary(text: String, out: IntList) {
        if (text.isEmpty()) return
        val m = PRETOKENIZE.matcher(text)
        while (m.find()) {
            val piece = m.group()
            val cached = cache[piece]
            if (cached != null) {
                out.addAll(cached)
                continue
            }
            val ids = bpe(piece.toByteArray(Charsets.UTF_8))
            if (piece.length <= MAX_CACHED_PIECE) cache[piece] = ids
            out.addAll(ids)
        }
    }

    private fun bpe(bytes: ByteArray): IntArray {
        // Symbols as vocab ids; -2 marks a byte with no vocab entry.
        var syms = IntArray(bytes.size) { byteToId[bytes[it].toInt() and 0xFF].let { id -> if (id < 0) -2 else id } }
        var n = syms.size
        while (n >= 2) {
            var bestRank = Int.MAX_VALUE
            var bestMerged = -1
            var bestLeft = -1
            var bestRight = -1
            for (i in 0 until n - 1) {
                val a = syms[i]
                val b = syms[i + 1]
                if (a < 0 || b < 0) continue
                val packed = mergeTable.get(a, b)
                if (packed != MergeTable.MISSING) {
                    val rank = (packed ushr 32).toInt()
                    if (rank < bestRank) {
                        bestRank = rank
                        bestMerged = packed.toInt()
                        bestLeft = a
                        bestRight = b
                    }
                }
            }
            if (bestMerged < 0) break
            // Merge every occurrence of the best pair, left to right.
            val next = IntArray(n)
            var j = 0
            var i = 0
            while (i < n) {
                if (i < n - 1 && syms[i] == bestLeft && syms[i + 1] == bestRight) {
                    next[j++] = bestMerged
                    i += 2
                } else {
                    next[j++] = syms[i++]
                }
            }
            syms = next
            n = j
        }
        val out = IntList()
        for (k in 0 until n) {
            val s = syms[k]
            when {
                s >= 0 -> out.add(s)
                unkId >= 0 -> out.add(unkId)
                else -> droppedBytes++
            }
        }
        return out.toArray()
    }

    private val cache = object : LinkedHashMap<String, IntArray>(256, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, IntArray>?) = size > CACHE_SIZE
    }

    /** Open-addressing (leftId,rightId) -> (rank<<32 | mergedId), linear probing. */
    private class MergeTable(expected: Int) {
        private val capacity = Integer.highestOneBit(maxOf(4, expected * 2 - 1)) shl 1
        private val keys = LongArray(capacity) { EMPTY }
        private val values = LongArray(capacity)
        private val mask = capacity - 1

        fun putIfAbsent(a: Int, b: Int, rank: Int, merged: Int) {
            val key = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
            var i = mix(key) and mask
            while (keys[i] != EMPTY) {
                if (keys[i] == key) return // first (lowest-rank) definition wins
                i = (i + 1) and mask
            }
            keys[i] = key
            values[i] = (rank.toLong() shl 32) or (merged.toLong() and 0xFFFFFFFFL)
        }

        fun get(a: Int, b: Int): Long {
            val key = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
            var i = mix(key) and mask
            while (true) {
                val k = keys[i]
                if (k == key) return values[i]
                if (k == EMPTY) return MISSING
                i = (i + 1) and mask
            }
        }

        private fun mix(k: Long): Int {
            var h = k * -0x61c8864680b583ebL
            h = h xor (h ushr 29)
            return h.toInt()
        }

        companion object {
            const val EMPTY = Long.MIN_VALUE
            const val MISSING = Long.MIN_VALUE
        }
    }

    companion object {
        private const val CACHE_SIZE = 4096
        private const val MAX_CACHED_PIECE = 32
        private val UNK_NAMES = listOf("<unk>", "<|unk|>", "[UNK]")

        /**
         * Unicode White_Space, written out. `\s` means ASCII-only whitespace on
         * the JVM but Unicode whitespace on Android (ICU), and Android rejects
         * Pattern.UNICODE_CHARACTER_CLASS. An explicit class behaves the same
         * on both and matches the reference tokenizer.
         */
        private const val WS = "\\t\\n\\x0B\\f\\r \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

        /** GPT-2 pre-tokenizer. */
        val PRETOKENIZE: Pattern = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^$WS\\p{L}\\p{N}]+|[$WS]+(?![^$WS])|[$WS]+",
        )

        /** GPT-2 bytes_to_unicode: printable bytes map to themselves, others to U+0100+. */
        val BYTE_TO_UNICODE: CharArray = CharArray(256).also { table ->
            val direct = BooleanArray(256)
            for (b in '!'.code..'~'.code) direct[b] = true
            for (b in '¡'.code..'¬'.code) direct[b] = true
            for (b in '®'.code..'ÿ'.code) direct[b] = true
            var n = 0
            for (b in 0 until 256) {
                table[b] = if (direct[b]) b.toChar() else (256 + n++).toChar()
            }
        }

        val UNICODE_TO_BYTE: IntArray = IntArray(512) { -1 }.also { inv ->
            for (b in 0 until 256) inv[BYTE_TO_UNICODE[b].code] = b
        }

        /** The byte-level alphabet's space marker. */
        const val SPACE_MARK = 'Ġ'
    }
}

/** Minimal growable int list (avoids boxing on the hot path). */
internal class IntList(cap: Int = 16) {
    private var a = IntArray(cap)
    var size = 0
        private set

    fun add(v: Int) {
        if (size == a.size) a = a.copyOf(a.size * 2)
        a[size++] = v
    }

    fun addAll(vs: IntArray) {
        for (v in vs) add(v)
    }

    fun toArray(): IntArray = a.copyOf(size)
}
