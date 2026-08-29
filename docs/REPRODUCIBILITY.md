# Reproducibility Notes

- Later stages use strict provenance checks and SHA256 hashes of predecessor scripts.
- Stages are result-dependent only through precommitted gates written before each run.
- Many later stages are read-only audits and do not create new optimizer conditions.
- The later mechanism stages use float64.
- Use the original stage result archives when replay scripts request predecessor folders.
- The shell runners assume a project virtual environment at `../.venv/bin/python`.
  Adjust only the environment path if your local repository layout differs.
- Do not silently alter seeds, thresholds, horizons or gate definitions when reproducing the reported results.
