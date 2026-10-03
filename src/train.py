"""Train, validate, and predict. Run from the repo root:  python src/train.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, lightgbm as lgb
from features import clean, build_features, daily_reference

PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
SEEDS = [0, 1, 2, 3, 4]


def metrics(yt, yp):
    return {"MAE": float(np.mean(abs(yt - yp))), "MedAE": float(np.median(abs(yt - yp))),
            "MAPE_%": float(np.mean(abs(yt - yp) / yt) * 100), "medAPE_%": float(np.median(abs(yt - yp) / yt) * 100),
            "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2)))}


def fit(X, y, dist):
    """Target = log(rate / distance): the model learns rate-per-mile, distance scales it back."""
    models = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(random_state=s, **PARAMS)
        m.fit(X, np.log(y) - np.log(dist)); models.append(m)
    return models


def predict(models, X, dist):
    return np.exp(np.mean([m.predict(X) for m in models], axis=0) + np.log(dist))


def main():
    t = pd.read_csv("data/train_test.csv", parse_dates=["date"])
    v = pd.read_csv("data/validation.csv", parse_dates=["date"])
    dec = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])
    # per-date market signals (date-level medians; unlabeled features only, no target used)
    ref = daily_reference([t, v])
    T, V = clean(t, ref), clean(v, ref)
    XT, XV = build_features(T, ref), build_features(V, ref)
    y, dT = T["posted_rate"].values, T["distance"].values

    # ---- time-based validation: train Jan-Aug, test Sep-Oct (mimics forecasting Nov-Dec)
    tr = (T.date < "2025-09-01").values
    p = predict(fit(XT[tr], y[tr], dT[tr]), XT[~tr], dT[~tr])
    report = {"time_holdout_Sep_Oct": metrics(y[~tr], p)}
    # ---- unseen-city holdout (validation has 8 cities never seen in training)
    hold = set(np.random.RandomState(1).choice(sorted(set(T.pickup)), 8, replace=False))
    touch = (T.pickup.isin(hold) | T.delivery.isin(hold)).values
    a, b = tr & ~touch, ~tr & touch
    report["unseen_city_holdout"] = metrics(y[b], predict(fit(XT[a], y[a], dT[a]), XT[b], dT[b]))
    print(json.dumps(report, indent=2))

    # ---- final model on all labeled data
    models = fit(XT, y, dT)
    pred = predict(models, XV, V["distance"].values)
    out = pd.DataFrame({"load_id": v["load_id"], "predicted_rate": np.round(pred, 2)})
    tmpl = pd.read_csv("data/validation_predictions_template.csv")
    assert list(tmpl.load_id) == list(out.load_id), "ID order mismatch with template"
    out.to_csv("validation_predictions.csv", index=False)

    # ---- fixed December chart: only the date varies. Date-level market_index/quote_signal come
    # from the validation file's December loads (the same market data used for the rest of validation).
    coords = pd.concat([t[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "lat", "lon"], axis=1),
                        t[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["c", "lat", "lon"], axis=1)]).drop_duplicates("c").set_index("c")
    D = dec.copy()
    for side in ["pickup", "delivery"]:
        D[f"{side}_lat"] = D[side].map(coords["lat"]); D[f"{side}_lon"] = D[side].map(coords["lon"])
    D["market_index"] = D["date"].map(ref["market_index"]); D["quote_signal"] = D["date"].map(ref["quote_signal"])
    D = clean(D, ref)
    XD = build_features(D, ref)
    dec["predicted_rate"] = np.round(predict(models, XD, D["distance"].values), 2)
    dec.to_csv("data/december_chart_inputs.csv", index=False, date_format="%Y-%m-%d")
    Path("scorer_results").mkdir(exist_ok=True)
    Path("scorer_results/validation_metrics.json").write_text(json.dumps(report, indent=2))
    print("December range:", dec.predicted_rate.min(), dec.predicted_rate.max())


if __name__ == "__main__":
    main()
