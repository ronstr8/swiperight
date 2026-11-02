package com.swiperight.predict

import java.io.File

/** Local unit tests run with the module directory as working directory. */
object TestAssets {
    val tokenizer: BpeTokenizer by lazy {
        val f = File("src/main/assets/tokenizer.json")
        check(f.exists()) { "missing ${f.absolutePath}; run tools/build_model.sh" }
        TokenizerLoader.parse(f.readText())
    }
}
