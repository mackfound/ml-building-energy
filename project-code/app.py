import re
import warnings
import threading
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_STATE = "IN"
BUCKET = "oedi-data-lake"
ROOT = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
RELEASE_CANDIDATES = [
    "2025/resstock_amy2018_release_1",
    "2024/resstock_amy2018_release_2",
    "2024/resstock_tmy3_release_2",
    "2022/resstock_amy2018_release_1",
    "2022/resstock_tmy3_release_1",
    "2021/resstock_2018_release_1",
]
TARGET_H = "out.electricity.heating.energy_consumption.kwh"
TARGET_C = "out.electricity.cooling.energy_consumption.kwh"

FEATURE_PATTERNS = [
    r"^in\.geometry.*floor",
    r"^in\.geometry.*building",
    r"^in\.geometry.*stories",
    r"^in\.geometry.*attic",
    r"^in\.geometry.*foundation",
    r"^in\.vintage",
    r"^in\.heating",
    r"^in\.cooling",
    r"^in\.hvac",
    r"^in\.insulation",
    r"^in\.window",
    r"^in\.duct",
    r"^in\.income",
    r"^in\.occupants",
    r"^in\.bedrooms",
    r"^in\.climate",
]

# ── Shared mutable state ───────────────────────────────────────────────────────

_state = {
    "ready": False,
    "loading": False,
    "error": None,
    "current_state": DEFAULT_STATE,
    "heating_model": None,
    "cooling_model": None,
    "X_heat_template": None,
    "X_cool_template": None,
    "df": None,
    "form_cols": {},
    "state_avg_cache": {},
}
_lock = threading.Lock()

# ── Helper functions ───────────────────────────────────────────────────────────

def find_columns(patterns, columns, flags=re.IGNORECASE):
    return [
        c for c in columns
        if any(re.search(p, c, flags=flags) for p in patterns)
    ]


def first_match(patterns, columns, exclude=None):
    hits = find_columns(patterns, columns)
    if exclude:
        hits = [h for h in hits if not re.search(exclude, h, re.IGNORECASE)]
    return hits[0] if hits else None


def find_baseline_uri(state, fs):
    for release in RELEASE_CANDIDATES:
        base = (
            f"{BUCKET}/{ROOT}/{release}"
            f"/metadata_and_annual_results/by_state/state={state}/parquet"
        )
        matches = []
        for pattern in [f"{base}/**/*.parquet", f"{base}/*.parquet"]:
            try:
                matches.extend(fs.glob(pattern))
            except Exception:
                pass
        baseline = sorted(
            m for m in set(matches)
            if "baseline" in m.lower() and "upgrade" not in m.lower()
        )
        if baseline:
            return "s3://" + baseline[0]
    raise FileNotFoundError(f"No baseline parquet found for state {state!r}")


def parse_floor_area(val):
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", "")
    if "-" in s:
        parts = s.split("-")
        try:
            return (int(parts[0]) + int(parts[1])) / 2
        except ValueError:
            return np.nan
    if ">" in s:
        try:
            return float(s.replace(">", "").strip())
        except ValueError:
            return np.nan
    if "<" in s:
        try:
            return float(s.replace("<", "").strip()) / 2
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _extract_number(val):
    m = re.search(r"-?\d+\.?\d*", str(val))
    return float(m.group()) if m else None


def nearest_category(series, target_value, parser=_extract_number):
    options = series.dropna().unique()
    parsed = {
        opt: v
        for opt in options
        if (v := parser(opt)) is not None and not (isinstance(v, float) and np.isnan(v))
    }
    if not parsed:
        mode = series.mode(dropna=True)
        return mode.iloc[0] if not mode.empty else (options[0] if len(options) else None)
    return min(parsed, key=lambda opt: abs(parsed[opt] - target_value))


def default_row(template_df):
    row = {}
    for c in template_df.columns:
        if pd.api.types.is_numeric_dtype(template_df[c]):
            row[c] = template_df[c].median()
        else:
            mode = template_df[c].mode(dropna=True)
            row[c] = mode.iloc[0] if not mode.empty else np.nan
    return pd.DataFrame([row])


def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
        ]), cat_cols),
    ])


def train_model(target_col, df, feature_cols, force_keep, n_estimators=100):
    data = df[[target_col] + feature_cols].copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col])

    keep = [
        c for c in feature_cols
        if c in data.columns and (
            c in force_keep
            or (data[c].isna().mean() < 0.80 and data[c].nunique(dropna=True) > 1)
        )
    ]

    X = data[keep]
    y = data[target_col].astype(float)

    # Use 75/25 split for preprocessor fitting, then refit pipeline on all data
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=34351)
    preprocess = build_preprocessor(X_tr)

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators, max_depth=14, min_samples_leaf=5,
            random_state=34351, n_jobs=-1,
        )),
    ])
    pipe.fit(X, y)   # fit on full dataset for production serving
    return pipe, X


# ── Model initialisation (runs in background thread) ──────────────────────────

def _load(state):
    try:
        import s3fs
        import pyarrow.parquet as pq

        fs = s3fs.S3FileSystem(anon=True)
        uri = find_baseline_uri(state, fs)

        with fs.open(uri.replace("s3://", ""), "rb") as fh:
            all_columns = pq.ParquetFile(fh).schema.names

        # Feature columns
        candidates = list(dict.fromkeys(
            c for pat in FEATURE_PATTERNS
            for c in find_columns([pat], all_columns)
        ))
        feature_cols = candidates[:25]

        # Form-control columns
        form_cols = {
            k: v for k, v in {
                "floor_area": first_match([r"geometry.*floor.*area", r"floor.*area"], all_columns),
                "foundation": first_match([r"foundation"], all_columns),
                "fuel":       first_match([r"heating.*fuel"], all_columns),
                "setpoint":   first_match([r"heating.*setpoint"], all_columns, exclude=r"offset"),
                "stories":    first_match([r"geometry.*stories", r"\bstories\b"], all_columns),
            }.items() if v is not None
        }

        # Load parquet
        needed = list(dict.fromkeys(
            [TARGET_H, TARGET_C] + feature_cols + list(form_cols.values())
        ))
        needed = [c for c in needed if c in all_columns]
        df = pd.read_parquet(uri, columns=needed, storage_options={"anon": True})

        # Train
        model_features = list(dict.fromkeys(feature_cols + list(form_cols.values())))
        model_features = [c for c in model_features if c in df.columns]
        force = list(form_cols.values())

        heating_model, X_h = train_model(TARGET_H, df, model_features, force)
        cooling_model, X_c = train_model(TARGET_C, df, model_features, force)

        avg_h = float(pd.to_numeric(df[TARGET_H], errors="coerce").mean())
        avg_c = float(pd.to_numeric(df[TARGET_C], errors="coerce").mean())

        with _lock:
            _state.update({
                "ready": True,
                "loading": False,
                "error": None,
                "current_state": state,
                "heating_model": heating_model,
                "cooling_model": cooling_model,
                "X_heat_template": X_h,
                "X_cool_template": X_c,
                "df": df,
                "form_cols": form_cols,
                "state_avg_cache": {state: {"avg_h": avg_h, "avg_c": avg_c}},
            })

    except Exception as e:
        with _lock:
            _state["error"] = f"{e}\n\n{traceback.format_exc()}"
            _state["loading"] = False
            _state["ready"] = False


def start_loading(state=DEFAULT_STATE):
    with _lock:
        if _state["loading"]:
            return
        _state["loading"] = True
        _state["ready"] = False
        _state["error"] = None
    threading.Thread(target=_load, args=(state,), daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "ready":   _state["ready"],
            "loading": _state["loading"],
            "error":   _state["error"],
            "state":   _state["current_state"],
        })


@app.route("/api/load", methods=["POST"])
def api_load():
    state = (request.json or {}).get("state", DEFAULT_STATE).upper()
    with _lock:
        already = _state["current_state"] == state and (_state["ready"] or _state["loading"])
    if not already:
        start_loading(state)
    return jsonify({"ok": True})


@app.route("/api/choices")
def api_choices():
    with _lock:
        ready = _state["ready"]
        df = _state.get("df")
        form_cols = _state.get("form_cols", {})
    if not ready or df is None:
        return jsonify({"error": "not ready"}), 503

    choices = {}
    for field, col in form_cols.items():
        if col and col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist(), key=str)
            choices[field] = vals
        else:
            choices[field] = []
    return jsonify(choices)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    with _lock:
        ready          = _state["ready"]
        heating_model  = _state["heating_model"]
        cooling_model  = _state["cooling_model"]
        X_h            = _state["X_heat_template"]
        X_c            = _state["X_cool_template"]
        form_cols      = _state["form_cols"]
        avg_cache      = _state["state_avg_cache"]
        cur_state      = _state["current_state"]

    if not ready:
        return jsonify({"error": "Models not ready"}), 503

    body       = request.json or {}
    fuel       = body.get("fuel")
    foundation = body.get("foundation")
    floor_area = float(body.get("floor_area", 1500))
    setpoint   = float(body.get("setpoint", 70))
    stories    = int(body.get("stories", 1))

    heat_row = default_row(X_h)
    cool_row = default_row(X_c)

    for row, tmpl in [(heat_row, X_h), (cool_row, X_c)]:
        col = form_cols.get("floor_area")
        if col and col in row.columns:
            row[col] = nearest_category(tmpl[col], floor_area, parser=parse_floor_area)

        col = form_cols.get("fuel")
        if col and col in row.columns and fuel:
            row[col] = fuel

        col = form_cols.get("foundation")
        if col and col in row.columns and foundation:
            row[col] = foundation

        col = form_cols.get("setpoint")
        if col and col in row.columns:
            row[col] = nearest_category(tmpl[col], setpoint)

        col = form_cols.get("stories")
        if col and col in row.columns:
            row[col] = nearest_category(tmpl[col], stories)

    pred_h = float(heating_model.predict(heat_row)[0])
    pred_c = float(cooling_model.predict(cool_row)[0])

    avgs = avg_cache.get(cur_state, {"avg_h": 0, "avg_c": 0})

    return jsonify({
        "heating":     round(pred_h),
        "cooling":     round(pred_c),
        "avg_heating": round(avgs["avg_h"]),
        "avg_cooling": round(avgs["avg_c"]),
        "total":       round(pred_h + pred_c),
        "cost":        round((pred_h + pred_c) * 0.15, 2),
        "state":       cur_state,
    })


if __name__ == "__main__":
    import os
    start_loading(DEFAULT_STATE)
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
