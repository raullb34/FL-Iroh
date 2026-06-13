"""
preprocess_e7.py — Build the E7 FL dataset from CyL air-quality + AEMET data.

Inputs  (relative to repo root):
  data/air-quailty/datasets/calidad-del-aire-datos-historicos-diarios-cyl.csv
  data/air-quailty/datasets/aemet_data_<provincia>.csv  (11 files)

Outputs (same datasets/ folder):
  air_quality_fl_classification.csv  — 10 clients (provincias), 7-day-ahead ICA labels
  air_quality_fl_timeseries.csv      — same rows, raw daily values for ProphetWrapper

Schema of classification CSV:
  provincia, fecha, NO2, O3, PM_particle, CO, velmedia, prec, label_ica, split

Task framing (t+7 forecasting):
  Features  = pollutant readings + meteorology on day t  (outdoor stations, JCyL network)
  Label ICA = ICA class of NO2 on day t+7
  This enables agricultural decision-making: a farmer observes today's air quality and
  receives a 7-day-ahead forecast of outdoor air pollution risk.
  Labels use training-set NO2 terciles (Q1/Q2) to produce balanced 3-class targets.
  The 7-day horizon eliminates the target-leakage present in same-day classification.

Train split: 2011–2018  (8 years, avoids COVID artefact documented in IJIMAI paper)
Test  split: 2019

Note on applicability: features include outdoor wind speed (velmedia) and precipitation
(prec) from AEMET stations. This dataset is suitable for outdoor monitoring deployments
only; greenhouse or indoor scenarios require a separate feature set.

Usage:
  python data/air-quailty/notebooks/preprocess_e7.py
  # or from repo root:
  python -m data.air-quailty.notebooks.preprocess_e7
"""
from __future__ import annotations

import csv
import logging
import statistics
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("preprocess_e7")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[3]
DATASETS    = REPO_ROOT / "data" / "air-quailty" / "datasets"
AIR_CSV     = DATASETS / "calidad-del-aire-datos-historicos-diarios-cyl.csv"
OUT_CLASSIF = DATASETS / "air_quality_fl_classification.csv"
OUT_TS      = DATASETS / "air_quality_fl_timeseries.csv"

# AEMET province mapping: canonical province name → AEMET filename stem
# NOTE: The CyL air-quality CSV uses 'Avila' (without accent) for this province.
AEMET_MAP: dict[str, str] = {
    "Avila"     : "aemet_data_avila",
    "Burgos"    : "aemet_data_burgos",
    "León"      : "aemet_data_leon",
    "Palencia"  : "aemet_data_palencia",
    "Ponferrada": "aemet_data_ponferrada",  # industrial León sub-region (Compostilla/Ponferrada stations)
    "Salamanca" : "aemet_data_salamanca",
    "Segovia"   : "aemet_data_segovia",
    "Soria"     : "aemet_data_soria",
    "Valladolid": "aemet_data_valladolid",
    "Zamora"    : "aemet_data_zamora",
}

# The 10 FL clients — Ponferrada is a separate station in León province
# but treated as an independent client (matches the IJIMAI paper)
FL_CLIENTS = list(AEMET_MAP.keys())

TRAIN_YEARS = set(range(2011, 2019))   # 2011–2018
TEST_YEARS  = {2019}
FORECAST_HORIZON = 7   # days ahead: features at t, label = ICA(NO2 at t+7)

# ICA labels derived from data-driven terciles of NO2 in the training set.
# The absolute Spanish ICA thresholds (40 / 100 μg/m³) are designed for
# hourly peak values. CyL *daily medians* have a mean of ~12 μg/m³, so
# absolute thresholds produce a 99%+ class-0 imbalance unsuitable for FL.
# We use province-wide training-set terciles instead, which:
#   • preserve the NO2 ranking  (higher = worse air quality)
#   • produce balanced classes  (~33% each)
#   • retain geographic non-IID (industrial Ponferrada/Valladolid have
#     proportionally more "high" days than rural Ávila/Soria)
# Tercile boundaries are computed in main() from the training data.
_ICA_Q1: float = 0.0   # 33rd percentile — set in main()
_ICA_Q2: float = 0.0   # 66th percentile — set in main()

def ica_label(no2: float) -> int:
    """Assign 0/1/2 based on training-set tercile boundaries."""
    if no2 < _ICA_Q1:
        return 0   # Bajo
    if no2 < _ICA_Q2:
        return 1   # Medio
    return 2       # Alto

ICA_NAMES = {0: "Bajo", 1: "Medio", 2: "Alto"}

# ---------------------------------------------------------------------------
# Step 1: Load and aggregate air-quality data by (provincia, fecha)
# ---------------------------------------------------------------------------

def load_air_quality() -> dict[tuple[str, str], dict]:
    """
    Return dict keyed by (provincia_canonical, fecha_str) with daily medians of:
      NO2, O3, PM_particle (PM2.5 if available, else PM10), CO
    across all stations in that province.
    """
    # Accumulate per-(provincia, fecha): list of values per pollutant
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"NO2": [], "O3": [], "PM25": [], "PM10": [], "CO": []}
    )

    with open(AIR_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            prov_raw = row.get("Provincia", "").strip()
            fecha    = row.get("Fecha", "").strip()
            if not fecha or not prov_raw:
                continue

            # Normalise province name (Ponferrada stations are in León province
            # in the CSV, but we keep them separate by station name)
            estacion = row.get("Estación", "").strip()
            if "Ponferrada" in estacion or "Compostilla" in estacion or "Cortiguera" in estacion:
                prov = "Ponferrada"
            else:
                prov = prov_raw

            if prov not in FL_CLIENTS:
                continue

            key = (prov, fecha)
            b   = buckets[key]

            def _f(col: str) -> float | None:
                v = row.get(col, "").strip()
                if v in ("", "None", "-"):
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            for col, field in [
                ("NO2", "NO2 (ug/m3)"),
                ("O3",  "O3 (ug/m3)"),
                ("PM25","PM25 (ug/m3)"),
                ("PM10","PM10 (ug/m3)"),
                ("CO",  "CO (mg/m3)"),
            ]:
                v = _f(field)
                if v is not None and v >= 0:
                    b[col].append(v)

    # Compute daily medians
    result: dict[tuple[str, str], dict] = {}
    for (prov, fecha), b in buckets.items():
        no2  = statistics.median(b["NO2"])  if b["NO2"]  else None
        o3   = statistics.median(b["O3"])   if b["O3"]   else None
        pm   = (statistics.median(b["PM25"]) if b["PM25"]
                else statistics.median(b["PM10"]) if b["PM10"]
                else None)
        co   = statistics.median(b["CO"])   if b["CO"]   else None
        result[(prov, fecha)] = {
            "NO2": no2, "O3": o3, "PM_particle": pm, "CO": co
        }

    log.info("Air-quality records (provincia, fecha): %d", len(result))
    return result

# ---------------------------------------------------------------------------
# Step 2: Load AEMET meteorological data per province
# ---------------------------------------------------------------------------

def load_aemet() -> dict[tuple[str, str], dict]:
    """Return dict keyed by (provincia, fecha_str) → {velmedia, prec}."""
    result: dict[tuple[str, str], dict] = {}

    for prov, stem in AEMET_MAP.items():
        fpath = DATASETS / f"{stem}.csv"
        if not fpath.exists():
            log.warning("AEMET file not found: %s — using NaN for %s", fpath, prov)
            continue
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fecha = row.get("fecha", "").strip()
                if not fecha:
                    continue
                def _f(k: str) -> float | None:
                    v = row.get(k, "").strip()
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None

                result[(prov, fecha)] = {
                    "velmedia": _f("velmedia"),
                    "prec"    : _f("prec"),
                }
        # Ponferrada uses León AEMET data (closest available station)
        if prov == "León":
            leo_data = {k: v for k, v in result.items() if k[0] == "León"}
            for (_, fecha), vals in leo_data.items():
                if ("Ponferrada", fecha) not in result:
                    result[("Ponferrada", fecha)] = vals.copy()

    log.info("AEMET records: %d", len(result))
    return result

# ---------------------------------------------------------------------------
# Step 3: Impute missing values with provincial medians
# ---------------------------------------------------------------------------

def impute(
    rows: list[dict],
    num_cols: list[str],
) -> list[dict]:
    """Replace None values with the median of the same (provincia, column)."""
    # Compute medians per (provincia, col)
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        for col in num_cols:
            v = r.get(col)
            if v is not None:
                acc[(r["provincia"], col)].append(v)

    medians: dict[tuple[str, str], float] = {
        k: statistics.median(vs) for k, vs in acc.items() if vs
    }
    global_medians: dict[str, float] = {}
    for col in num_cols:
        all_vals = [v for k, vs in acc.items() if k[1] == col for v in vs]
        if all_vals:
            global_medians[col] = statistics.median(all_vals)

    imputed = 0
    for r in rows:
        for col in num_cols:
            if r.get(col) is None:
                key = (r["provincia"], col)
                r[col] = medians.get(key, global_medians.get(col, 0.0))
                imputed += 1

    log.info("Imputed %d missing values", imputed)
    return rows

# ---------------------------------------------------------------------------
# Step 4: Z-score normalisation (global statistics over train set)
# ---------------------------------------------------------------------------

def zscore_normalize(
    rows: list[dict],
    num_cols: list[str],
) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    """Normalise num_cols in-place using train-set global statistics."""
    train_rows = [r for r in rows if r["split"] == "train"]
    stats: dict[str, tuple[float, float]] = {}
    for col in num_cols:
        vals = [r[col] for r in train_rows if r.get(col) is not None]
        if not vals:
            stats[col] = (0.0, 1.0)
            continue
        mu  = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 1.0
        std = std if std > 1e-8 else 1.0
        stats[col] = (mu, std)

    for r in rows:
        for col in num_cols:
            mu, std = stats[col]
            r[col] = (r[col] - mu) / std

    log.info("Z-score stats: %s", {k: f"μ={v[0]:.2f} σ={v[1]:.2f}" for k, v in stats.items()})
    return rows, stats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _ICA_Q1, _ICA_Q2

    air = load_air_quality()
    met = load_aemet()

    num_cols = ["NO2", "O3", "PM_particle", "CO", "velmedia", "prec"]

    # --- Compute tercile boundaries from TRAINING rows BEFORE labelling -------
    # We need raw NO2 values from training years to set the thresholds.
    train_no2_raw: list[float] = []
    for (prov, fecha), aq in air.items():
        if aq["NO2"] is None:
            continue
        try:
            year = int(fecha[:4])
        except ValueError:
            continue
        if year in TRAIN_YEARS and prov in FL_CLIENTS:
            train_no2_raw.append(aq["NO2"])

    train_no2_raw.sort()
    n_raw = len(train_no2_raw)
    _ICA_Q1 = train_no2_raw[int(n_raw * 0.333)] if n_raw >= 3 else 8.0
    _ICA_Q2 = train_no2_raw[int(n_raw * 0.666)] if n_raw >= 3 else 15.0
    log.info(
        "NO2 tercile thresholds (training set, n=%d): Q1=%.2f  Q2=%.2f  μg/m³",
        n_raw, _ICA_Q1, _ICA_Q2,
    )
    # --------------------------------------------------------------------------

    # Build combined rows
    all_rows: list[dict] = []
    for (prov, fecha), aq in sorted(air.items()):
        try:
            year = int(fecha[:4])
        except ValueError:
            continue

        if year not in TRAIN_YEARS and year not in TEST_YEARS:
            continue

        if aq["NO2"] is None:
            continue   # NO2 is required for the ICA label

        met_vals = met.get((prov, fecha), {})
        row = {
            "provincia"  : prov,
            "fecha"      : fecha,
            "NO2"        : aq["NO2"],
            "O3"         : aq["O3"],
            "PM_particle": aq["PM_particle"],
            "CO"         : aq["CO"],
            "velmedia"   : met_vals.get("velmedia"),
            "prec"       : met_vals.get("prec"),
            "_no2_raw"   : aq["NO2"],   # kept for t+7 label assignment below
            "label_ica"  : -1,           # placeholder; filled after t+7 shift
            "split"      : "train" if year in TRAIN_YEARS else "test",
        }
        all_rows.append(row)

    log.info("Rows before imputation: %d", len(all_rows))

    # --- Apply t+7 label shift -----------------------------------------------
    # For each row at date t, the ICA label = class of NO2 at (t + 7 days).
    # Rows without a t+7 observation are dropped.
    import datetime
    # Build raw NO2 lookup: (provincia, fecha_str) → raw NO2
    no2_lookup: dict[tuple[str, str], float] = {
        (r["provincia"], r["fecha"]): r["_no2_raw"]
        for r in all_rows
        if r["_no2_raw"] is not None
    }
    labeled_rows: list[dict] = []
    for r in all_rows:
        try:
            d = datetime.date.fromisoformat(r["fecha"])
        except ValueError:
            continue
        future_date = (d + datetime.timedelta(days=FORECAST_HORIZON)).isoformat()
        future_no2 = no2_lookup.get((r["provincia"], future_date))
        if future_no2 is None:
            continue  # no observation 7 days ahead — drop this row
        r["label_ica"] = ica_label(future_no2)
        labeled_rows.append(r)
    all_rows = labeled_rows
    log.info(
        "After t+%d label shift: %d rows retained (%.1f%% of pre-shift)",
        FORECAST_HORIZON, len(all_rows),
        100 * len(all_rows) / max(len(labeled_rows) + (len(labeled_rows) - len(all_rows)), 1),
    )

    all_rows = impute(all_rows, num_cols)

    # Remove internal helper column before writing
    for r in all_rows:
        r.pop("_no2_raw", None)

    # Save raw timeseries (for ProphetWrapper — NO normalisation)
    ts_fields = ["provincia", "fecha", "NO2", "O3", "PM_particle", "CO",
                 "velmedia", "prec", "label_ica", "split"]
    with open(OUT_TS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ts_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Saved timeseries CSV: %s (%d rows)", OUT_TS, len(all_rows))

    # Save ICA thresholds so that the experiment can reproduce labels
    import json
    thresholds_path = DATASETS / "air_quality_ica_thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({"q1": _ICA_Q1, "q2": _ICA_Q2,
                   "description": "NO2 training-set tercile thresholds (ug/m3)"}, f, indent=2)
    log.info("Saved ICA thresholds: %s  (Q1=%.2f Q2=%.2f)", thresholds_path, _ICA_Q1, _ICA_Q2)

    # Normalise and save classification CSV
    all_rows, _ = zscore_normalize(all_rows, num_cols)
    cl_fields = ts_fields  # same schema
    with open(OUT_CLASSIF, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cl_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Saved classification CSV: %s (%d rows)", OUT_CLASSIF, len(all_rows))

    # ---- Diagnostics -------------------------------------------------------
    log.info("--- Class distribution per provincia (train set) ---")
    for prov in FL_CLIENTS:
        prov_train = [r for r in all_rows if r["provincia"] == prov and r["split"] == "train"]
        if not prov_train:
            log.warning("  %s: NO DATA", prov)
            continue
        from collections import Counter
        dist = Counter(r["label_ica"] for r in prov_train)
        total = len(prov_train)
        dist_str = "  ".join(
            f"{ICA_NAMES[k]}={dist.get(k,0)} ({100*dist.get(k,0)/total:.0f}%)"
            for k in [0, 1, 2]
        )
        log.info("  %-12s  n=%4d  %s", prov, total, dist_str)

    log.info("--- Train/test split ---")
    n_train = sum(1 for r in all_rows if r["split"] == "train")
    n_test  = sum(1 for r in all_rows if r["split"] == "test")
    log.info("  train=%d  test=%d  total=%d", n_train, n_test, len(all_rows))


if __name__ == "__main__":
    main()
