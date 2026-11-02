package com.swiperight.predict

import org.json.JSONObject

/**
 * assets/model_manifest.json, written by tools/convert/write_manifest.py.
 *
 * It is the version contract between the three assets. [Predictor] checks it
 * at load time and refuses to run on a mismatch. A keyboard showing confidently
 * wrong words is worse than a keyboard with no suggestions.
 */
data class ModelManifest(
    val format: Int,
    val modelVersion: String,
    val modelFile: String,
    val modelBytes: Long,
    val modelSha256: String,
    val seqLen: Int,
    val modelVocabSize: Int,
    val tokenizerFile: String,
    val tokenizerBytes: Long,
    val tokenizerSha256: String,
    val tokenizerVocabSize: Int,
    val bosId: Int,
    val padId: Int,
) {
    companion object {
        /** Graph I/O contract version this app understands. */
        const val SUPPORTED_FORMAT = 1

        fun parse(json: String): ModelManifest {
            val r = JSONObject(json)
            val m = r.getJSONObject("model")
            val t = r.getJSONObject("tokenizer")
            return ModelManifest(
                format = r.getInt("format"),
                modelVersion = r.getString("model_version"),
                modelFile = m.getString("file"),
                modelBytes = m.getLong("bytes"),
                modelSha256 = m.getString("sha256"),
                seqLen = m.getInt("seq_len"),
                modelVocabSize = m.getInt("vocab_size"),
                tokenizerFile = t.getString("file"),
                tokenizerBytes = t.getLong("bytes"),
                tokenizerSha256 = t.getString("sha256"),
                tokenizerVocabSize = t.getInt("vocab_size"),
                bosId = t.getInt("bos_id"),
                padId = t.getInt("pad_id"),
            )
        }
    }
}

/** Thrown when the assets disagree with each other or with this app version. */
class ModelMismatchException(message: String) : IllegalStateException(message)
