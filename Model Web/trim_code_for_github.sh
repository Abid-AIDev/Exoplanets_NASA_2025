#!/usr/bin/env bash
# Safely trim Python source files for GitHub by moving non-essential .py files into archive_pruned/code_archive/
# Usage:
#   ./trim_code_for_github.sh --dry-run   # show files that WOULD be moved
#   ./trim_code_for_github.sh            # actually move files

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR="$ROOT/archive_pruned/code_archive"
DRY_RUN=false
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=true
fi

# Whitelist of files and directories to KEEP in the repository
read -r -d '' WHITELIST <<'WL'
app/Comprehensive_NASA_Space_Apps_App.py
save_merged_dataset.py
simple_train.py
train_exoplanet_model.py
train_enhanced_exoplanet_model.py
prune_for_github.sh
trim_code_for_github.sh
requirements.txt
README.md
NASA_SPACE_APPS_README.md
FINAL_NASA_SPACE_APPS_SOLUTION.md
models/README_MODEL.md
data/processed
dataset
archive_pruned
WL

echo "Root: $ROOT"
echo "Archive target: $ARCHIVE_DIR"
echo

mkdir -p "$ARCHIVE_DIR"

mapfile -t keep <<<"$(echo "$WHITELIST" | sed '/^$/d')"

# Find .py files excluding virtualenv and the archive dir
IFS=$'\n' read -r -d '' -a pyfiles < <(find "$ROOT" -type f -name "*.py" \! -path "*/.venv/*" \! -path "*/.venv312/*" \! -path "*/archive_pruned/*" -print0 | xargs -0 -n1 printf "%s\n" && printf '\0')

declare -a to_move
for f in "${pyfiles[@]}"; do
  rel=${f#$ROOT/}
  # check whitelist (prefix match)
  keep_flag=false
  for k in "${keep[@]}"; do
    if [[ "$rel" == "$k" ]] || [[ "$rel" == $k/* ]]; then
      keep_flag=true
      break
    fi
  done
  if [[ "$keep_flag" == false ]]; then
    to_move+=("$rel")
  fi
done

if [[ ${#to_move[@]} -eq 0 ]]; then
  echo "No non-whitelisted .py files found to move. Nothing to do."
  exit 0
fi

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN: The following .py files would be moved to $ARCHIVE_DIR:" 
  for p in "${to_move[@]}"; do
    echo "  - $p"
  done
  echo
  echo "Run './trim_code_for_github.sh' to perform the move. Files will be preserved in archive_pruned/code_archive/"
  exit 0
fi

for rel in "${to_move[@]}"; do
  src="$ROOT/$rel"
  dest="$ARCHIVE_DIR/$(basename "$rel")"
  echo "Moving $rel -> $dest"
  mv "$src" "$dest"
done

echo "Done. Archived ${#to_move[@]} files to $ARCHIVE_DIR"
