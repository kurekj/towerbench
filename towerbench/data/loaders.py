"""Raw public-dataset loaders.

Every loader returns ``(inter, items)``:
  inter : DataFrame[user(str), item(str), ts(int64 seconds)]  implicit feedback events
  items : DataFrame[item(str), text(str), lat(float|nan), lon(float|nan), price(float|nan),
                    cat(str)]  side information (missing columns are allowed and filled later)
Nothing here is proprietary: all sources are public downloads listed in download_raw.sh.
"""
from __future__ import annotations

import ast
import gzip
import json
import re

import numpy as np
import pandas as pd

from ..paths import RAW

_HTML = re.compile(r"<[^>]+>")


def _epoch_seconds(x) -> pd.Series:
    """Integer UTC seconds independent of the datetime resolution pandas infers (s/ms/ns)."""
    dt = pd.to_datetime(x, utc=True, errors="coerce", format="ISO8601")
    return ((dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("Int64")


def _clean(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return _HTML.sub(" ", str(s)).replace("\n", " ").strip()


# ----------------------------------------------------------------------------- MovieLens-1M
def load_ml1m(min_rating: float = 0.0):
    d = RAW / "ml-1m" / "ml-1m"
    r = pd.read_csv(d / "ratings.dat", sep="::", engine="python",
                    names=["user", "item", "rating", "ts"])
    r = r[r.rating >= min_rating]
    inter = pd.DataFrame({"user": r.user.astype(str), "item": r.item.astype(str),
                          "ts": r.ts.astype("int64")})
    m = pd.read_csv(d / "movies.dat", sep="::", engine="python", encoding="latin-1",
                    names=["item", "title", "genres"])
    items = pd.DataFrame({"item": m.item.astype(str),
                          "text": m.title + ". Genres: " + m.genres.str.replace("|", ", "),
                          "cat": m.genres.str.split("|").str[0]})
    return inter, items


# ----------------------------------------------------------------------------- Gowalla
def load_gowalla():
    p = RAW / "gowalla" / "loc-gowalla_totalCheckins.txt.gz"
    df = pd.read_csv(p, sep="\t", names=["user", "time", "lat", "lon", "item"])
    df["ts"] = _epoch_seconds(df.time)
    inter = pd.DataFrame({"user": df.user.astype(str), "item": df.item.astype(str), "ts": df.ts})
    items = df.groupby("item").agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    items["item"] = items.item.astype(str)
    items["text"] = ""
    return inter, items


# ----------------------------------------------------------------------------- Amazon 2023
def load_amazon(category: str):
    d = RAW / "amazon"
    r = pd.read_csv(d / f"{category}.5core.csv")
    # columns: user_id,parent_asin,rating,timestamp (ms)
    inter = pd.DataFrame({"user": r.user_id.astype(str), "item": r.parent_asin.astype(str),
                          "ts": (r.timestamp // 1000).astype("int64")})
    keep = set(inter.item.unique())
    rows = []
    with open(d / f"meta_{category}.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            a = o.get("parent_asin")
            if a not in keep:
                continue
            feats = " ".join(o.get("features") or [])
            desc = " ".join(o.get("description") or [])
            cats = o.get("categories") or []
            price = o.get("price")
            try:
                price = float(price) if price not in (None, "", "None") else np.nan
            except (TypeError, ValueError):
                price = np.nan
            title = o.get("title", "")
            rows.append({"item": a, "text": _clean(f"{title}. {feats} {desc}")[:2000],
                         "price": price,
                         "cat": cats[-1] if cats else (o.get("main_category") or "")})
    items = pd.DataFrame(rows).drop_duplicates("item")
    return inter, items


# ----------------------------------------------------------------------------- Inside Airbnb
AIRBNB_CITIES = ["new-york-city", "london", "paris", "amsterdam", "barcelona", "berlin", "rome",
                 "los-angeles", "lisbon", "madrid"]


def load_airbnb(cities: list[str] | None = None):
    d = RAW / "airbnb"
    cities = cities or AIRBNB_CITIES
    inters, items = [], []
    for c in cities:
        rp, lp = d / f"{c}_reviews.csv.gz", d / f"{c}_listings.csv.gz"
        if not (rp.exists() and lp.exists()):
            continue
        rv = pd.read_csv(rp, usecols=["listing_id", "reviewer_id", "date"])
        rv["ts"] = _epoch_seconds(rv.date)
        inters.append(pd.DataFrame({"user": rv.reviewer_id.astype(str),
                                    "item": c + ":" + rv.listing_id.astype(str), "ts": rv.ts}))
        ls = pd.read_csv(lp, low_memory=False)
        price = ls["price"].astype(str).str.replace(r"[$,]", "", regex=True)
        price = pd.to_numeric(price, errors="coerce")

        def col(name, ls=ls):
            return ls[name] if name in ls else pd.Series([""] * len(ls))

        txt = (col("name").map(_clean) + ". " + col("description").map(_clean) + " "
               + col("neighborhood_overview").map(_clean) + " Amenities: "
               + col("amenities").map(_clean))
        items.append(pd.DataFrame({
            "item": c + ":" + ls.id.astype(str), "text": txt.str[:2000],
            "lat": ls.latitude, "lon": ls.longitude, "price": price,
            "cat": col("room_type").fillna(""),
            "city": c}))
    inter = pd.concat(inters, ignore_index=True)
    items = pd.concat(items, ignore_index=True).drop_duplicates("item")
    return inter, items


# ----------------------------------------------------------------------------- Steam
def load_steam():
    d = RAW / "steam"
    rows = []
    with gzip.open(d / "steam_reviews.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            o = ast.literal_eval(line)  # upstream file is python-literal, not JSON
            rows.append((o["username"], str(o["product_id"]), o["date"]))
    r = pd.DataFrame(rows, columns=["user", "item", "date"])
    r["ts"] = _epoch_seconds(r.date)
    inter = r.dropna(subset=["ts"])[["user", "item", "ts"]]
    meta = []
    with gzip.open(d / "steam_games.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            o = ast.literal_eval(line)
            if "id" not in o:
                continue
            price = o.get("price")
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = np.nan
            genres = " ".join(o.get("genres") or [])
            tags = " ".join(o.get("tags") or [])
            title = o.get("title", "")
            meta.append({"item": str(o["id"]), "price": price,
                         "text": _clean(f"{title}. {genres}. {tags}")[:1000],
                         "cat": (o.get("genres") or [""])[0]})
    items = pd.DataFrame(meta).drop_duplicates("item")
    return inter, items


LOADERS = {
    "ml-1m": lambda: load_ml1m(),
    "gowalla": load_gowalla,
    "amazon-instruments": lambda: load_amazon("Musical_Instruments"),
    "amazon-videogames": lambda: load_amazon("Video_Games"),
    "amazon-software": lambda: load_amazon("Software"),
    "airbnb": lambda: load_airbnb(),
    "airbnb-nyc": lambda: load_airbnb(["new-york-city"]),
    "steam": load_steam,
}
