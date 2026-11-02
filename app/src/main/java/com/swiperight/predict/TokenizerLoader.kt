package com.swiperight.predict

import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream

/**
 * Parses tokenizer.json:
 * `{"vocab": {token: id}, "merges": [["a","b"], ...], "special_tokens": [..]}`.
 *
 * Also accepts merges as Hugging Face-style "a b" strings. Uses org.json, which
 * is part of the Android platform, so no extra dependency is needed.
 * Parsing GPT-2's 1.4 MB file peaks at around 15 MB of transient heap. It runs
 * once, on the predictor thread.
 */
object TokenizerLoader {

    fun load(input: InputStream): BpeTokenizer {
        val text = input.bufferedReader(Charsets.UTF_8).use { it.readText() }
        return parse(text)
    }

    fun parse(json: String): BpeTokenizer {
        val root = JSONObject(json)
        val vocabObj = root.getJSONObject("vocab")
        val vocab = HashMap<String, Int>(vocabObj.length() * 4 / 3 + 1)
        val keys = vocabObj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            vocab[k] = vocabObj.getInt(k)
        }

        val mergesArr = root.optJSONArray("merges") ?: JSONArray()
        val merges = ArrayList<Pair<String, String>>(mergesArr.length())
        for (i in 0 until mergesArr.length()) {
            when (val m = mergesArr.get(i)) {
                is JSONArray -> {
                    require(m.length() == 2) { "merge #$i must have 2 parts, has ${m.length()}" }
                    merges += m.getString(0) to m.getString(1)
                }
                is String -> {
                    val sp = m.indexOf(' ')
                    require(sp > 0) { "merge #$i is not 'a b': $m" }
                    merges += m.substring(0, sp) to m.substring(sp + 1)
                }
                else -> throw IllegalArgumentException("merge #$i has unexpected type")
            }
        }

        val specials = LinkedHashSet<String>()
        root.optJSONArray("special_tokens")?.let { arr ->
            for (i in 0 until arr.length()) specials += arr.getString(i)
        }
        return BpeTokenizer(vocab, merges, specials)
    }
}
