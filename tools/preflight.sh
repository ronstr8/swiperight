#!/usr/bin/env bash
# Check that this machine can build the app and reach a phone, and say how
# to fix whatever is missing. Prints the adb path on success.
#
#   tools/preflight.sh build     # JDK + Android SDK
#   tools/preflight.sh device    # ...plus exactly one adb device
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-build}"
fail() { echo "preflight: $*" >&2; exit 1; }

command -v java >/dev/null || fail "no JDK on PATH. Install JDK 17 or newer (e.g. 'sudo apt install openjdk-17-jdk')."

# Gradle finds the SDK through local.properties or ANDROID_HOME; a fresh
# clone has neither, so look in the usual install places and write one.
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
if [ -z "$SDK" ] && [ -f "$ROOT/local.properties" ]; then
  SDK="$(sed -n 's/^sdk\.dir=//p' "$ROOT/local.properties")"
fi
for guess in "$HOME/Android/Sdk" "$HOME/Library/Android/sdk"; do
  [ -n "$SDK" ] || { [ -d "$guess" ] && SDK="$guess"; } || true
done
[ -n "$SDK" ] && [ -d "$SDK" ] || fail "Android SDK not found. Install Android Studio (or the command-line tools) and set ANDROID_HOME."
[ -f "$ROOT/local.properties" ] || echo "sdk.dir=$SDK" > "$ROOT/local.properties"

[ "$MODE" = device ] || exit 0

ADB="$(command -v adb || true)"
[ -n "$ADB" ] || ADB="$SDK/platform-tools/adb"
[ -x "$ADB" ] || fail "adb not found. Install 'platform-tools' with the SDK manager."

DEVICES="$("$ADB" devices | awk 'NR>1 && $2=="device"' | wc -l)"
if [ "$DEVICES" -eq 0 ]; then
  fail "no phone connected. Plug in with USB debugging on, or pair over Wi-Fi:
    $ADB pair <phone-ip>:<pairing-port>
    $ADB connect <phone-ip>:<debug-port>
  (Settings > Developer options > Wireless debugging). See README."
fi
[ "$DEVICES" -eq 1 ] || [ -n "${ANDROID_SERIAL:-}" ] || fail "$DEVICES devices connected. Pick one: ANDROID_SERIAL=<serial> make run"
echo "$ADB"
