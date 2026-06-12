#!/usr/bin/env bash
# Optional local preview helper: build a static WASM bundle of the app.
# The canonical build happens in GitHub Actions (.github/workflows/deploy.yml).
#
# Output: ./build/index.html (gitignored) and any sibling assets marimo
# produces. Preview with `cd build && python -m http.server`.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export UV_PROJECT_ENVIRONMENT="$HOME/.uv_envs/schelling_segregation"
export UV_LINK_MODE=copy

OUT_DIR="${1:-build}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Exporting WASM bundle to $OUT_DIR/index.html ..."
uv run marimo export html-wasm app.py -o "$OUT_DIR/index.html" --mode run

echo
echo "Done. To preview locally:"
echo "  cd $OUT_DIR && python -m http.server 8000"
echo "  then open http://localhost:8000/"
