.DEFAULT_GOAL := help
.PHONY: help build build-release test device-test lint clean model model-build model-publish install run ci

GRADLE   := ./gradlew
APP_ID   := com.swiperight
ACTIVITY := $(APP_ID)/.ui.MainActivity
ASSETS   := app/src/main/assets
MODEL    := $(ASSETS)/distilgpt2.tflite
MODEL_VERSION = $(shell python3 -c "import json; print(json.load(open('$(ASSETS)/model_manifest.json'))['model_version'])")

help: ## Show this list
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

run: install ## Install on the connected phone and open the setup screen
	@"$$(tools/preflight.sh device)" shell am start -n $(ACTIVITY) >/dev/null
	@echo "Opened swiperight on the phone: tap Enable keyboard, then Switch keyboard."

install: model ## Build and install the debug APK on the connected phone
	@tools/preflight.sh device >/dev/null
	$(GRADLE) installDebug

build: model ## Build the debug APK
	@tools/preflight.sh build
	$(GRADLE) assembleDebug

build-release: model ## Build the minified release APK (debug-signed)
	@tools/preflight.sh build
	$(GRADLE) assembleRelease

test: ## Run host unit tests
	@tools/preflight.sh build
	$(GRADLE) testDebugUnitTest

device-test: model ## Run instrumented tests on the connected phone
	@tools/preflight.sh device >/dev/null
	$(GRADLE) connectedDebugAndroidTest

lint: model ## Run Android lint
	@tools/preflight.sh build
	$(GRADLE) lintDebug

ci: test lint build ## Everything that must pass before a commit

model: $(MODEL) ## Download the prebuilt model (61 MB) if it's missing

$(MODEL):
	tools/fetch_model.sh

model-build: ## Rebuild the model from distilgpt2 (needs uv, ~2 GB of wheels)
	tools/build_model.sh

model-publish: ## Upload the local model to its GitHub release (maintainers)
	@if gh release view model-$(MODEL_VERSION) >/dev/null 2>&1; then \
	  echo "Release model-$(MODEL_VERSION) already exists."; exit 0; fi; \
	gh release create model-$(MODEL_VERSION) $(MODEL) licenses/APACHE-2.0.txt licenses/MODEL_NOTICE.md \
	  --title "Model $(MODEL_VERSION)" \
	  --notes "Prebuilt model for swiperight, fetched by 'make model' and checked against assets/model_manifest.json. Derived from distilbert/distilgpt2 (Apache-2.0); see MODEL_NOTICE.md for the changes." \
	  --latest=false

clean: ## Remove build outputs (keeps the model)
	$(GRADLE) clean
