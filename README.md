# İzmir urban mobility & transport demand ML

A three-person CE 477 course project exploring İzmirim Kart ridership data with clustering, classification, regression and association-rule mining. It studies demand patterns by operator and fare type and compares forecasting approaches.

[Portfolio context](https://onurerguden.dev/en/projects#urban-mobility) · [Research context](https://onurerguden.dev/en/research) · [Onur Ergüden](https://onurerguden.dev)

## What the project explores

- Ridership patterns and seasonal features in public-transport data.
- DBSCAN, hierarchical clustering and K-means comparisons.
- Demand regression with XGBoost, Random Forest and Ridge.
- Classification comparisons and confusion matrices.
- Association rules for fare-type and usage patterns.
- Exploratory analysis of the Narlıdere metro expansion.

## Repository guide

| Path | Purpose |
| --- | --- |
| `data/current-data/` | Current CSV snapshots |
| `data/old-data/` | Earlier snapshot |
| `data_loader.py`, `preprocessing.py`, `main.py` | Data loading and exploratory workflow |
| `clustering/` | Clustering scripts and comparisons |
| `classification/` | Classifier comparisons and saved figures |
| `regression/` | Regression experiments |
| `association_mining/` | Association rules |
| `narlidere_efficiency.py` | Metro-expansion analysis |
| `plots/` | Saved visualizations and model-comparison figures |

Start with the [saved regression figures](plots/regression_final/) and [clustering comparisons](clustering/clustering_results/) for a quick review, then inspect the scripts that generated each result.

## Working with the code

The repository consists of academic analysis scripts rather than a packaged application. There is currently no consolidated requirements file. Dependencies vary by experiment; the regression code uses pandas, NumPy, matplotlib, seaborn, scikit-learn and XGBoost.

Run the root exploratory workflow from the repository root after installing its dependencies:

```sh
python main.py
```

`data_loader.py` reads the semicolon-delimited snapshot at `data/current-data/izmirim-kart-ulasim-istatistikleri-guncel-extended.csv`. Individual experiments may use separate copies and relative paths; inspect the chosen script's input path before running it.

## Interpreting the analysis

This is an independent course study and has not been submitted for peer review. Saved scores describe particular experiments and data splits; they are not guarantees of future forecasting performance. The metro-expansion analysis is exploratory and should not be treated as causal evidence without checking its assumptions and controls.
