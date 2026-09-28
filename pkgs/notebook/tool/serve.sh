#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NOTEBOOK_DIR/../.." && pwd)"

# Select Dart SDK: env DART_SDK, or ~/.dvm/darts/3.13.1, or ~/.dvm/darts/3.12.0, or dart in PATH
if [ -n "$DART_SDK" ] && [ -x "$DART_SDK/bin/dart" ]; then
  DART="$DART_SDK/bin/dart"
elif [ -x "$HOME/.dvm/darts/3.13.1/bin/dart" ]; then
  DART="$HOME/.dvm/darts/3.13.1/bin/dart"
elif [ -x "$HOME/.dvm/darts/3.12.0/bin/dart" ]; then
  DART="$HOME/.dvm/darts/3.12.0/bin/dart"
elif command -v dart >/dev/null 2>&1; then
  DART="dart"
else
  echo "Error: No Dart SDK found in ~/.dvm/darts/ or PATH." >&2
  exit 1
fi

# Store active SDK version in hooks_runner marker to only recompile when SDK version changes
CURRENT_SDK_VERSION="$("$DART" --version 2>&1)"
MARKER_FILE="$REPO_ROOT/.dart_tool/hooks_runner/.sdk_version"

if [ ! -f "$MARKER_FILE" ] || [ "$(cat "$MARKER_FILE" 2>/dev/null)" != "$CURRENT_SDK_VERSION" ]; then
  rm -rf "$REPO_ROOT/.dart_tool/hooks_runner" "$NOTEBOOK_DIR/.dart_tool/hooks_runner"
  mkdir -p "$REPO_ROOT/.dart_tool/hooks_runner"
  echo "$CURRENT_SDK_VERSION" > "$MARKER_FILE"
fi

cd "$NOTEBOOK_DIR"
exec "$DART" run bin/notebook_server.dart "$@"
