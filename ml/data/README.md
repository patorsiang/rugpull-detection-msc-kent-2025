# Data

This folder contains all datasets used in the machine learning pipeline, split into raw and processed formats.

---

## Structure

```txt
data/
├── external/                               # Third-party data (e.g., CRPWarner, RPToken, RPHunter, GoPlus)
├── interim/                                # Intermediate data (e.g., partially cleaned, merged)
├── processed/                              # Final feature-rich datasets used for training/testing
├── raw/                                    # Original, unmodified datasets
└── README.md # Documentation (this file)   # Documentation
```

---

### `raw/`

- Contains raw, unmodified datasets from source.
- Examples:
  - `transactions_raw.csv`
  - `contracts_dump.json`

### `interim/`

- Contains data after basic cleaning or merging, but before final feature engineering.
- Examples:
  - `merged_with_labels.csv`
  - `cleaned_transactions.csv`

### `processed/`

- Final, model-ready datasets.
- Examples:
  - `features.csv`
  - `X_train.csv`, `y_train.csv`

### `external/`

- Manually downloaded datasets from public research or third-party APIs.
- Data is excluded from GitHub via .`gitignore` and folders are preserved with `.gitkeep`.
- Follow instructions below to populate each subdirectory.

#### Data Sources

| Dataset    | Link                                                         | destination directory |
|------------|--------------------------------------------------------------|-----------------------|
| CRPWarner  | <https://github.com/CRPWarner/RugPull/tree/main/dataset>     | /data/external/crpwarner |
