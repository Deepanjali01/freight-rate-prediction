"""Cleaning and feature engineering shared by training and prediction."""
import numpy as np
import pandas as pd

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}


def haversine(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


def clean(df: pd.DataFrame, daily_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fix data-quality problems. daily_ref: per-date market_index/quote_signal medians."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["weight_missing"] = d["weight"].isna().astype(int)
    d["weight_was_negative"] = (d["weight"] < 0).astype(int)
    d["weight"] = d["weight"].abs()                      # sign-flip errors
    d["weight"] = d["weight"].fillna(d.groupby("equipment")["weight"].transform("median"))
    d["weight"] = d["weight"].fillna(31_000)

    d["hav"] = haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    # NOTE: the `distance` column is clean. Within a lane it varies by only ~2%, and short lanes have a
    # 70-mile minimum, so ratios to great-circle distance look odd (coordinates are rough, e.g. Allentown
    # sits ~1 mile from New York). Rebuilding "odd" distances was tested and made short-lane errors worse.

    d["market_missing"] = d["market_index"].isna().astype(int)
    ref = d.groupby("date")["market_index"].median() if daily_ref is None else daily_ref["market_index"]
    d["market_index"] = d["market_index"].fillna(d["date"].map(ref))
    d["market_index"] = d["market_index"].fillna(d["market_index"].median())
    return d


def daily_reference(frames) -> pd.DataFrame:
    allf = pd.concat(frames)
    allf["date"] = pd.to_datetime(allf["date"])
    return allf.groupby("date")[["market_index", "quote_signal"]].median()


def build_features(d: pd.DataFrame, daily_ref: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["log_dist"] = np.log(d["distance"])
    X["distance"] = d["distance"]
    X["hav"] = d["hav"]
    X["equip"] = d["equipment"].map(EQUIP)
    X["weight"] = d["weight"]
    X["weight_missing"] = d["weight_missing"]
    X["weight_was_negative"] = d["weight_was_negative"]
    X["market_index"] = d["market_index"]
    X["quote_signal"] = d["quote_signal"]
    X["qs_dev"] = d["quote_signal"] - 2.0
    X["qs_absdev"] = (d["quote_signal"] - 2.0).abs()
    X["day_mkt"] = d["date"].map(daily_ref["market_index"])
    X["day_qs"] = d["date"].map(daily_ref["quote_signal"])
    X["dow"] = d["date"].dt.dayofweek
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = d[c]
    X["dlat"] = d["delivery_lat"] - d["pickup_lat"]
    X["dlon"] = d["delivery_lon"] - d["pickup_lon"]
    return X
