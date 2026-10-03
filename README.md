# Freight Rate Prediction

Predicts `posted_rate` for truckload freight. See `Freight_Rate_ML_Assessment.pdf` for the brief.

## Run

```bash
python -m pip install -r requirements.txt
python src/train.py            # validates, trains, writes validation_predictions.csv + data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs: `validation_predictions.csv`, completed `data/december_chart_inputs.csv`, `scorer_results/candidate_december.png`, `scorer_results/validation_metrics.json`.

## Approach

- **Cleaning** (`src/features.py`): negative weights -> absolute value (292 rows); missing weight (300) -> equipment median + flag; missing `market_index` (374) -> same-date median + flag. The `distance` column is kept as is: it is consistent within each lane (~2%) and short lanes have a 70-mile minimum, so rebuilding it from coordinates (tested) made short-lane predictions worse.
- **Features**: log distance, great-circle distance, equipment, weight, market index, quote signal (and its deviation from 2.0), same-date market medians, day of week, lat/lon of both ends and their deltas. City names are *not* used as features, because 8 cities in `validation.csv` never appear in training; coordinates generalize to them.
- **Model**: LightGBM, L1 objective on `log(rate / distance)` (robust to the ~0.7% extreme rate outliers), 5-seed average, trained on all labeled data for the final predictions.
- **Validation**: time-based holdout (train Jan-Aug, test Sep-Oct) because the final set is Nov-Dec; plus an unseen-city holdout. A random split is deliberately not used for model selection (it showed MAE of about $78 versus about $111 for the honest time split, which is leakage).
- **December chart**: only the date varies. Date-level `market_index` / `quote_signal` come from the median of `validation.csv` loads on that date.
