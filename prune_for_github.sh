#!/usr/bin/env bash
# Prune large files for GitHub push by moving them into archive_pruned/
# Usage:
#   ./prune_for_github.sh --dry-run   # shows what will be moved
#   ./prune_for_github.sh            # actually moves files

set -euo pipefail

DRY_RUN=false
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=true
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR="$ROOT/archive_pruned"

declare -a MOVE_PATHS=(
  "models/comprehensive_best_model.joblib"
  "models/improved_best_model.joblib"
  "models/enhanced_best_model.joblib"
  "models/final_binary_model.joblib"
  "models/tabular_xgb.joblib"
  "models/nasa_space_apps_model.joblib"
  "models/best_model.joblib"
  "models/binary_best_model.joblib"
  "dataset/Kepler Objects of Interest (KOI)_2025.09.29_04.48.01.csv"
  "dataset/K2 Planets and Candidates_2025.09.29_04.49.17.csv"
  "dataset/TESS Objects of Interest (TOI)_2025.09.29_04.48.50.csv"
  "data/processed/comprehensive_exoplanet_dataset.csv"
  "data/processed/enhanced_exoplanet_dataset.csv"
  "data/processed/merged_exoplanet_dataset.csv"
  "data/processed/final_balanced_dataset.csv"
)

echo "Prune script root: $ROOT"
echo "Archive dir: $ARCHIVE_DIR"
echo

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN: The following files would be moved to $ARCHIVE_DIR:" 
  for p in "${MOVE_PATHS[@]}"; do
    if [[ -e "$ROOT/$p" ]]; then
      echo "  - $p"
    fi
  done
  echo
  echo "Run './prune_for_github.sh' to perform the move. Archived files will be placed in archive_pruned/ and NOT deleted."
  exit 0
fi

mkdir -p "$ARCHIVE_DIR"

for p in "${MOVE_PATHS[@]}"; do
  src="$ROOT/$p"
  if [[ -e "$src" ]]; then
    dest="$ARCHIVE_DIR/$(basename "$p")"
    echo "Moving $p -> $dest"
    mv "$src" "$dest"
  fi
done

echo "Done. Large files moved to $ARCHIVE_DIR. Update git and push."
