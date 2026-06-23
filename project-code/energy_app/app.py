import os
import re
import warnings
import threading
import traceback

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

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
    # RF models
    "heating_model": None,
    "cooling_model": None,
    # KNN models
    "knn_heating_model": None,
    "knn_cooling_model": None,
    # Input templates (feature column names + mode/median defaults)
    "X_heat_template": None,
    "X_cool_template": None,
    # Raw data + metadata
    "df": None,
    "form_cols": {},
    "state_avg_cache": {},
    # Test-set metrics for all model types
    "rf_metrics":    {"heating": {}, "cooling": {}},
    "knn_metrics":   {"heating": {}, "cooling": {}},
    "ridge_metrics": {"heating": {}, "cooling": {}},
    # Sampled test-set points for diagnostic scatter plots
    "rf_plot_data":  {"heating": {}, "cooling": {}},
    "knn_plot_data": {"heating": {}, "cooling": {}},
}
_lock = threading.Lock()

# ── Utility functions ──────────────────────────────────────────────────────────

def find_columns(patterns, columns, flags=re.IGNORECASE):
    return [c for c in columns if any(re.search(p, c, flags=flags) for p in patterns)]


def first_match(patterns, columns, exclude=None):
    hits = find_columns(patterns, columns)
    if exclude:
        hits = [h for h in hits if not re.search(exclude, h, re.IGNORECASE)]
    return hits[0] if hits else None


def find_baseline_uri(state, fs):
    attempted = []
    for release in RELEASE_CANDIDATES:
        base = (
            f"{BUCKET}/{ROOT}/{release}"
            f"/metadata_and_annual_results/by_state/state={state}/parquet"
        )
        matches = []
        for pattern in [f"{base}/**/*.parquet", f"{base}/*.parquet"]:
            try:
                matches.extend(fs.glob(pattern))
            except Exception as exc:
                attempted.append((pattern, repr(exc)))
        baseline = sorted(
            m for m in set(matches)
            if "baseline" in m.lower() and "upgrade" not in m.lower()
        )
        if baseline:
            return "s3://" + baseline[0]
        if not matches:
            attempted.append((f"{base}/**/*.parquet", "no matches"))

    detail = "\n".join(f"  {p}: {e}" for p, e in attempted[-12:])
    raise FileNotFoundError(
        f"No baseline parquet found for state {state!r}.\n"
        f"Patterns tried (last 12):\n{detail}"
    )


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
    for prefix in (">", "<"):
        if prefix in s:
            try:
                v = float(s.replace(prefix, "").strip())
                return v if prefix == ">" else v / 2
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


def build_preprocessor(X_ref):
    """Create (unfitted) ColumnTransformer based on dtypes of X_ref."""
    num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_ref.columns if c not in num_cols]
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


def _compute_metrics(y_true, y_pred, n_train, n_test):
    return {
        "r2":      round(float(r2_score(y_true, y_pred)), 4),
        "rmse":    round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 1),
        "mae":     round(float(mean_absolute_error(y_true, y_pred)), 1),
        "n_train": n_train,
        "n_test":  n_test,
    }


def train_and_evaluate(target_col, df, feature_cols, force_keep, estimator):
    """
    Filter features, evaluate on a held-out split, then refit on all data.
    Returns (production_pipe, X_template, metrics_dict).
    """
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
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=34351)

    # Evaluate on held-out test set
    eval_pipe = Pipeline([
        ("preprocess", build_preprocessor(X_tr)),
        ("model", clone(estimator)),
    ])
    eval_pipe.fit(X_tr, y_tr)
    y_pred_eval = eval_pipe.predict(X_te)
    metrics = _compute_metrics(y_te, y_pred_eval, len(X_tr), len(X_te))

    # Sample up to 1 000 points for diagnostic plots
    rng      = np.random.default_rng(42)
    n_sample = min(1000, len(y_te))
    idx      = rng.choice(len(y_te), n_sample, replace=False)
    plot_pts = {
        "actual":    y_te.values[idx].tolist(),
        "predicted": y_pred_eval[idx].tolist(),
    }

    # Refit on all data for production serving
    prod_pipe = Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", clone(estimator)),
    ])
    prod_pipe.fit(X, y)

    return prod_pipe, X, metrics, plot_pts


# ── Input row builder (shared by RF and KNN prediction endpoints) ──────────────

def _build_input_row(body, template_df, form_cols):
    row = default_row(template_df)
    floor_area = float(body.get("floor_area", 1500))
    setpoint   = float(body.get("setpoint", 70))
    stories    = int(body.get("stories", 1))
    fuel       = body.get("fuel")
    foundation = body.get("foundation")

    col = form_cols.get("floor_area")
    if col and col in row.columns:
        row[col] = nearest_category(template_df[col], floor_area, parser=parse_floor_area)

    col = form_cols.get("fuel")
    if col and col in row.columns and fuel:
        row[col] = fuel

    col = form_cols.get("foundation")
    if col and col in row.columns and foundation:
        row[col] = foundation

    col = form_cols.get("setpoint")
    if col and col in row.columns:
        row[col] = nearest_category(template_df[col], setpoint)

    col = form_cols.get("stories")
    if col and col in row.columns:
        row[col] = nearest_category(template_df[col], stories)

    return row


# ── Model initialisation (background thread) ───────────────────────────────────

def _load(state):
    try:
        import s3fs
        import pyarrow.parquet as pq

        fs  = s3fs.S3FileSystem(anon=True)
        uri = find_baseline_uri(state, fs)

        with fs.open(uri.replace("s3://", ""), "rb") as fh:
            all_columns = pq.ParquetFile(fh).schema.names

        candidates = list(dict.fromkeys(
            c for pat in FEATURE_PATTERNS for c in find_columns([pat], all_columns)
        ))
        feature_cols = candidates[:25]

        form_cols = {
            k: v for k, v in {
                "floor_area": first_match([r"geometry.*floor.*area", r"floor.*area"], all_columns),
                "foundation": first_match([r"foundation"], all_columns),
                "fuel":       first_match([r"heating.*fuel"], all_columns),
                "setpoint":   first_match([r"heating.*setpoint"], all_columns, exclude=r"offset"),
                "stories":    first_match([r"geometry.*stories", r"\bstories\b"], all_columns),
            }.items() if v is not None
        }

        needed = list(dict.fromkeys([TARGET_H, TARGET_C] + feature_cols + list(form_cols.values())))
        needed = [c for c in needed if c in all_columns]
        df = pd.read_parquet(uri, columns=needed, storage_options={"anon": True})

        model_features = list(dict.fromkeys(feature_cols + list(form_cols.values())))
        model_features = [c for c in model_features if c in df.columns]
        force = list(form_cols.values())

        rf    = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                      random_state=34351, n_jobs=-1)
        knn   = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
        ridge = Ridge(alpha=1.0)

        rf_h,    X_h, rf_metrics_h,    rf_pts_h   = train_and_evaluate(TARGET_H, df, model_features, force, rf)
        rf_c,    X_c, rf_metrics_c,    rf_pts_c   = train_and_evaluate(TARGET_C, df, model_features, force, rf)
        knn_h,   _,   knn_metrics_h,   knn_pts_h  = train_and_evaluate(TARGET_H, df, model_features, force, knn)
        knn_c,   _,   knn_metrics_c,   knn_pts_c  = train_and_evaluate(TARGET_C, df, model_features, force, knn)
        ridge_h, _,   ridge_metrics_h, _          = train_and_evaluate(TARGET_H, df, model_features, force, ridge)
        ridge_c, _,   ridge_metrics_c, _          = train_and_evaluate(TARGET_C, df, model_features, force, ridge)

        avg_h = float(pd.to_numeric(df[TARGET_H], errors="coerce").mean())
        avg_c = float(pd.to_numeric(df[TARGET_C], errors="coerce").mean())

        with _lock:
            _state.update({
                "ready": True, "loading": False, "error": None,
                "current_state": state,
                "heating_model":     rf_h,
                "cooling_model":     rf_c,
                "knn_heating_model": knn_h,
                "knn_cooling_model": knn_c,
                "X_heat_template":   X_h,
                "X_cool_template":   X_c,
                "df": df,
                "form_cols": form_cols,
                "state_avg_cache": {state: {"avg_h": avg_h, "avg_c": avg_c}},
                "rf_metrics":    {"heating": rf_metrics_h,    "cooling": rf_metrics_c},
                "knn_metrics":   {"heating": knn_metrics_h,   "cooling": knn_metrics_c},
                "ridge_metrics": {"heating": ridge_metrics_h, "cooling": ridge_metrics_c},
                "rf_plot_data": {
                    "heating": {**rf_pts_h, "r2": rf_metrics_h["r2"]},
                    "cooling": {**rf_pts_c, "r2": rf_metrics_c["r2"]},
                },
                "knn_plot_data": {
                    "heating": {**knn_pts_h, "r2": knn_metrics_h["r2"]},
                    "cooling": {**knn_pts_c, "r2": knn_metrics_c["r2"]},
                },
            })

    except Exception as e:
        with _lock:
            _state["error"]   = f"{e}\n\n{traceback.format_exc()}"
            _state["loading"] = False
            _state["ready"]   = False


def start_loading(state=DEFAULT_STATE):
    with _lock:
        if _state["loading"]:
            return
        _state["loading"] = True
        _state["ready"]   = False
        _state["error"]   = None
    threading.Thread(target=_load, args=(state,), daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/knn")
def knn_page():
    return render_template("knn.html")


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
        ready     = _state["ready"]
        df        = _state.get("df")
        form_cols = _state.get("form_cols", {})
    if not ready or df is None:
        return jsonify({"error": "not ready"}), 503

    choices = {}
    for field, col in form_cols.items():
        if col and col in df.columns:
            choices[field] = sorted(df[col].dropna().unique().tolist(), key=str)
        else:
            choices[field] = []
    return jsonify(choices)


@app.route("/api/metrics")
def api_metrics():
    with _lock:
        ready         = _state["ready"]
        rf_metrics    = _state["rf_metrics"]
        knn_metrics   = _state["knn_metrics"]
        ridge_metrics = _state["ridge_metrics"]
    if not ready:
        return jsonify({"error": "not ready"}), 503
    return jsonify({"rf": rf_metrics, "knn": knn_metrics, "ridge": ridge_metrics})


@app.route("/api/rf_predictions")
def api_rf_predictions():
    with _lock:
        ready = _state["ready"]
        data  = _state["rf_plot_data"]
    if not ready:
        return jsonify({"error": "not ready"}), 503
    return jsonify(data)


@app.route("/api/knn_predictions")
def api_knn_predictions():
    with _lock:
        ready = _state["ready"]
        data  = _state["knn_plot_data"]
    if not ready:
        return jsonify({"error": "not ready"}), 503
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    with _lock:
        ready         = _state["ready"]
        heating_model = _state["heating_model"]
        cooling_model = _state["cooling_model"]
        X_h           = _state["X_heat_template"]
        X_c           = _state["X_cool_template"]
        form_cols     = _state["form_cols"]
        avg_cache     = _state["state_avg_cache"]
        cur_state     = _state["current_state"]
    if not ready:
        return jsonify({"error": "Models not ready"}), 503

    body = request.json or {}
    heat_row = _build_input_row(body, X_h, form_cols)
    cool_row = _build_input_row(body, X_c, form_cols)

    pred_h = float(heating_model.predict(heat_row)[0])
    pred_c = float(cooling_model.predict(cool_row)[0])
    avgs   = avg_cache.get(cur_state, {"avg_h": 0, "avg_c": 0})

    return jsonify({
        "heating":     round(pred_h),
        "cooling":     round(pred_c),
        "avg_heating": round(avgs["avg_h"]),
        "avg_cooling": round(avgs["avg_c"]),
        "total":       round(pred_h + pred_c),
        "cost":        round((pred_h + pred_c) * 0.15, 2),
        "state":       cur_state,
    })


@app.route("/api/predict_knn", methods=["POST"])
def api_predict_knn():
    with _lock:
        ready             = _state["ready"]
        knn_heat          = _state["knn_heating_model"]
        knn_cool          = _state["knn_cooling_model"]
        rf_heat           = _state["heating_model"]
        rf_cool           = _state["cooling_model"]
        X_h               = _state["X_heat_template"]
        X_c               = _state["X_cool_template"]
        form_cols         = _state["form_cols"]
        avg_cache         = _state["state_avg_cache"]
        cur_state         = _state["current_state"]
    if not ready:
        return jsonify({"error": "Models not ready"}), 503

    body = request.json or {}
    heat_row = _build_input_row(body, X_h, form_cols)
    cool_row = _build_input_row(body, X_c, form_cols)

    knn_h = float(knn_heat.predict(heat_row)[0])
    knn_c = float(knn_cool.predict(cool_row)[0])
    rf_h  = float(rf_heat.predict(heat_row)[0])
    rf_c  = float(rf_cool.predict(cool_row)[0])
    avgs  = avg_cache.get(cur_state, {"avg_h": 0, "avg_c": 0})

    return jsonify({
        "knn": {
            "heating": round(knn_h),
            "cooling": round(knn_c),
            "total":   round(knn_h + knn_c),
            "cost":    round((knn_h + knn_c) * 0.15, 2),
        },
        "rf": {
            "heating": round(rf_h),
            "cooling": round(rf_c),
            "total":   round(rf_h + rf_c),
            "cost":    round((rf_h + rf_c) * 0.15, 2),
        },
        "avg_heating": round(avgs["avg_h"]),
        "avg_cooling": round(avgs["avg_c"]),
        "state":       cur_state,
    })


if __name__ == "__main__":
    start_loading(DEFAULT_STATE)
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
