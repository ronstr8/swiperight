plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.swiperight"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.swiperight"
        // NNAPI delegate needs API 27; below that the predictor runs on CPU.
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "0.1.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // TFLite ships 4 ABIs (~4 MB each). Every phone in scope is arm64;
            // x86_64 is kept for the emulator.
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            // No release keystore in this repo; sign with the debug key so
            // `installRelease` works for on-device testing.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    androidResources {
        // The model is memory-mapped straight out of the APK. That only works
        // for uncompressed assets; a compressed one would have to be inflated
        // onto the Java heap (60+ MB).
        noCompress += listOf("tflite")
    }

    buildFeatures {
        buildConfig = true
    }

    testOptions {
        unitTests.all {
            it.testLogging { showStandardStreams = true }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

// Fail early with a useful message instead of crashing on device.
val checkModelAssets by tasks.registering {
    val assets = layout.projectDirectory.dir("src/main/assets")
    doLast {
        listOf("distilgpt2.tflite", "tokenizer.json", "model_manifest.json").forEach {
            if (!assets.file(it).asFile.exists()) {
                throw GradleException("Missing app/src/main/assets/$it. Run tools/build_model.sh first.")
            }
        }
    }
}
tasks.named("preBuild") { dependsOn(checkModelAssets) }

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.tflite)

    testImplementation(libs.junit)
    // org.json is stubbed in android.jar for local unit tests; use the real one.
    testImplementation(libs.org.json)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.junit)
}
