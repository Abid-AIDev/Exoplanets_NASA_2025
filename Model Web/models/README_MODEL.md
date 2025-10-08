This directory contained trained model artifacts used during development.

To keep the repository lightweight for GitHub, very large model files were moved to the top-level `archive_pruned/` folder by the `prune_for_github.sh` script. The following guidance explains what was kept and what was archived:

- Keep small metadata files (training summaries, jsons) in `models/` so the app can display training info.
- Archive very large binary model files (for example `comprehensive_best_model.joblib` which can be multiple GBs).

If you need to restore archived models locally, copy them back from `archive_pruned/` into `models/` before running the Streamlit app. Example:

  cp archive_pruned/comprehensive_best_model.joblib models/

Note on reproducibility: The repository keeps training code and smaller artifacts so the models can be re-created if required. The archive is provided separately to avoid pushing multi-gigabyte files to GitHub.
