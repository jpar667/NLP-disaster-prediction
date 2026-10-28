# Executed notebooks

- [01_eda.ipynb](01_eda.ipynb): data quality, missing values, label conflicts, text features, and vocabulary.
- [02_baseline.ipynb](02_baseline.ipynb): three feature configurations evaluated on the same five grouped folds.

Install with `python -m pip install -e .` and select that environment as the notebook kernel. Rebuild from the percent-format sources:

```sh
python scripts/build_notebooks.py eda
python scripts/build_notebooks.py baseline
```

These commands also generate standalone HTML copies under `reports/html/`.
