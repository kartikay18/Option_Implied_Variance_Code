#!/usr/bin/env python3
"""Additional V2.1 robustness and audit analyses.

This script is intentionally separate from the frozen V2 core pipeline. It reads
its row-level forecasts, recomputes alternative-loss and inference-sensitivity
results, adds calendar-day scaling robustness, and verifies the scale invariance
of idealized fund-drag R2 under the theoretical leverage loading.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

HORIZONS = (5, 22, 66)
MODELS = ("mean", "q_raw", "q_mz", "har4", "harqts")
PAIRS = (
    ("q_mz", "q_raw", "MZ option vs raw option"),
    ("q_mz", "har4", "MZ option vs HAR(4)"),
    ("harqts", "har4", "HAR + option + term vs HAR(4)"),
)
N_BOOT = 4999
SEED = 20260817


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hac_test(x: Iterable[float], lags: int) -> tuple[float, float, float, int]:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return math.nan, math.nan, math.nan, len(a)
    fit = sm.OLS(a, np.ones(len(a))).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(lags)}, use_t=False
    )
    return float(a.mean()), float(fit.tvalues[0]), float(fit.pvalues[0]), len(a)


def block_pvalue(x: Iterable[float], block: int, seed: int, reps: int = N_BOOT) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 20:
        return math.nan
    obs = float(a.mean())
    centered = a - obs
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < reps:
        batch = min(200, reps - done)
        starts = rng.integers(0, n, size=(batch, n_blocks))
        idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(batch, -1)[:, :n]
        means = centered[idx].mean(axis=1)
        exceed += int((np.abs(means) >= abs(obs)).sum())
        done += batch
    return float((exceed + 1) / (reps + 1))


def qlike_loss(y: np.ndarray, f: np.ndarray, floor: float) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=float), np.finfo(float).tiny)
    f = np.maximum(np.asarray(f, dtype=float), float(floor))
    ratio = y / f
    return ratio - np.log(ratio) - 1.0


def mse_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (y - f) ** 2


def mae_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.abs(y - f)


def oos_r2(y: np.ndarray, f: np.ndarray, benchmark: np.ndarray) -> float:
    denom = float(np.sum((y - benchmark) ** 2))
    return 1.0 - float(np.sum((y - f) ** 2)) / denom


def load_forecasts(results: Path, h: int) -> pd.DataFrame:
    df = pd.read_csv(results / f"variance_forecasts_{h}d.csv", parse_dates=["Date", "target_end", "training_end"])
    return df.set_index("Date").sort_index()


def alternative_loss_tests(forecasts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    floor_fracs = (1e-4, 1e-3, 1e-2)
    for h, df in forecasts.items():
        y = df.realized_variance.to_numpy(float)
        mean_y = float(np.mean(y))
        for pair_no, (a, b, label) in enumerate(PAIRS):
            fa = df[f"forecast_{a}"].to_numpy(float)
            fb = df[f"forecast_{b}"].to_numpy(float)
            delta_r2 = oos_r2(y, fa, df.forecast_mean.to_numpy(float)) - oos_r2(y, fb, df.forecast_mean.to_numpy(float))
            for loss_name, loss_fn in (("MSE", mse_loss), ("MAE", mae_loss)):
                diff = loss_fn(y, fb) - loss_fn(y, fa)
                mean, t, p, n = hac_test(diff, h + 5)
                rows.append({
                    "horizon": h, "comparison": label, "model_a": a, "model_b": b,
                    "loss": loss_name, "floor_fraction_of_mean_target": math.nan,
                    "floor_value": math.nan, "n": n, "delta_oos_r2": delta_r2,
                    "mean_loss_gain_a_over_b": mean, "hac_lags": h + 5,
                    "hac_t": t, "hac_p": p,
                    "block_length": h + 5,
                    "block_p": block_pvalue(diff, h + 5, SEED + h * 100 + pair_no * 10),
                })
            for j, frac in enumerate(floor_fracs):
                floor = mean_y * frac
                diff = qlike_loss(y, fb, floor) - qlike_loss(y, fa, floor)
                mean, t, p, n = hac_test(diff, h + 5)
                rows.append({
                    "horizon": h, "comparison": label, "model_a": a, "model_b": b,
                    "loss": "QLIKE", "floor_fraction_of_mean_target": frac,
                    "floor_value": floor, "n": n, "delta_oos_r2": delta_r2,
                    "mean_loss_gain_a_over_b": mean, "hac_lags": h + 5,
                    "hac_t": t, "hac_p": p,
                    "block_length": h + 5,
                    "block_p": block_pvalue(diff, h + 5, SEED + 5000 + h * 100 + pair_no * 10 + j),
                })
    return pd.DataFrame(rows)


def inference_sensitivity(forecasts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for h, df in forecasts.items():
        y = df.realized_variance.to_numpy(float)
        for pair_no, (a, b, label) in enumerate(PAIRS):
            fa = df[f"forecast_{a}"].to_numpy(float)
            fb = df[f"forecast_{b}"].to_numpy(float)
            diff = mse_loss(y, fb) - mse_loss(y, fa)
            for lag in sorted({max(1, h - 1), h + 5, 2 * h}):
                mean, t, p, n = hac_test(diff, lag)
                rows.append({
                    "horizon": h, "comparison": label, "model_a": a, "model_b": b,
                    "method": "HAC", "parameter": lag, "n": n,
                    "mean_loss_gain_a_over_b": mean, "statistic": t, "p_value": p,
                })
            for block in sorted({max(1, h - 1), h + 5, 2 * h}):
                rows.append({
                    "horizon": h, "comparison": label, "model_a": a, "model_b": b,
                    "method": "circular_block_bootstrap", "parameter": block, "n": len(diff),
                    "mean_loss_gain_a_over_b": float(np.mean(diff)), "statistic": math.nan,
                    "p_value": block_pvalue(diff, block, SEED + 10000 + h * 100 + pair_no * 10 + block),
                })
    return pd.DataFrame(rows)


def calibration_diagnostics(forecasts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for h, df in forecasts.items():
        y = df.realized_variance.to_numpy(float)
        benchmark = df.forecast_mean.to_numpy(float)
        for model in MODELS:
            f = df[f"forecast_{model}"].to_numpy(float)
            err = y - f
            mse = float(np.mean(err ** 2))
            mean_error = float(np.mean(err))
            X = sm.add_constant(f)
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 5}, use_t=False)
            rows.append({
                "horizon": h, "model": model, "n": len(y),
                "mean_realized": float(np.mean(y)), "mean_forecast": float(np.mean(f)),
                "mean_forecast_to_realized_ratio": float(np.mean(f) / np.mean(y)),
                "mean_error_realized_minus_forecast": mean_error,
                "mean_absolute_error": float(np.mean(np.abs(err))),
                "mse": mse, "squared_bias_share_of_mse": float(mean_error ** 2 / mse),
                "correlation": float(np.corrcoef(y, f)[0, 1]),
                "oos_r2": 0.0 if model == "mean" else oos_r2(y, f, benchmark),
                "evaluation_intercept": float(fit.params[0]),
                "evaluation_intercept_p": float(fit.pvalues[0]),
                "evaluation_slope": float(fit.params[1]),
                "evaluation_slope_p": float(fit.pvalues[1]),
                "zero_forecast_count": int(np.sum(f <= 0)),
            })
    return pd.DataFrame(rows)


def idealized_scale_invariance(forecasts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    gammas = {"SSO": 1.0, "UPRO": 3.0, "SDS": 3.0, "SPXU": 6.0}
    for h, df in forecasts.items():
        y = df.realized_variance.to_numpy(float)
        b = df.forecast_mean.to_numpy(float)
        underlying = {m: oos_r2(y, df[f"forecast_{m}"].to_numpy(float), b) for m in ("q_raw", "q_mz", "har4", "harqts")}
        for fund, gamma in gammas.items():
            yd = -gamma * y
            bd = -gamma * b
            for model, r2_u in underlying.items():
                fd = -gamma * df[f"forecast_{model}"].to_numpy(float)
                r2_d = oos_r2(yd, fd, bd)
                rows.append({
                    "horizon": h, "fund": fund, "gamma": gamma, "model": model,
                    "underlying_variance_oos_r2": r2_u,
                    "idealized_drag_oos_r2": r2_d,
                    "absolute_difference": abs(r2_d - r2_u),
                })
    return pd.DataFrame(rows)


def import_v2_module(code_path: Path):
    spec = importlib.util.spec_from_file_location("run_analysis_v2_module", code_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import V2 analysis module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def calendar_scaling_results(root: Path, forecasts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    v2 = import_v2_module(root / "code" / "run_analysis_v2.py")
    px, _ = v2.load_data(root / "data" / "market_data.csv", root / "data" / "rates.csv")
    ret = np.log(px.SPY).diff()
    vday = ret.pow(2)
    rows: list[dict] = []
    for h in HORIZONS:
        y_s = v2.forward_sum(vday, h)
        calendar_days = pd.Series(np.nan, index=px.index, dtype=float)
        for pos in range(len(px) - h):
            calendar_days.iloc[pos] = float((px.index[pos + h] - px.index[pos]).days)
        iv = px.VIX if h != 66 else px.VIX3M
        q = (iv / 100.0).pow(2) * calendar_days / 365.0
        term = ((px.VIX3M / 100.0).pow(2) - (px.VIX / 100.0).pow(2)) * calendar_days / 365.0
        X3 = pd.DataFrame({"d": vday, "w": vday.rolling(5).mean(), "m": vday.rolling(22).mean()}, index=px.index)
        X4 = X3.assign(qtr=vday.rolling(66).mean())
        Xs = {
            "q_mz": q.to_numpy(float)[:, None],
            "har4": X4.to_numpy(float),
            "harqts": np.column_stack([X4.to_numpy(float), q.to_numpy(float), term.to_numpy(float)]),
        }
        y = y_s.to_numpy(float)
        models = {name: v2.RecursiveOLS(X, y) for name, X in Xs.items()}
        mean_model = v2.RecursiveMean(y)
        recs: list[dict] = []
        for pos in range(len(px) - h):
            end = pos - h
            if end < 0:
                continue
            mean, nmean = mean_model.value(0, end)
            if nmean < v2.MIN_TRAIN:
                continue
            preds = {"mean": mean, "q_raw": float(q.iloc[pos])}
            ns = [nmean]
            for name, X in Xs.items():
                pred, _, n = models[name].predict(X[pos], 0, end, True)
                preds[name] = pred
                ns.append(n)
            vals = [y[pos], *preds.values()]
            if not np.isfinite(vals).all():
                continue
            recs.append({"Date": px.index[pos], "realized_variance": float(y[pos]), **{f"forecast_{k}": float(v) for k, v in preds.items()}})
        cal = pd.DataFrame(recs).set_index("Date")
        baseline = forecasts[h].loc[cal.index]
        yv = cal.realized_variance.to_numpy(float)
        for model in ("q_raw", "q_mz", "har4", "harqts"):
            r2_cal = oos_r2(yv, cal[f"forecast_{model}"].to_numpy(float), cal.forecast_mean.to_numpy(float))
            r2_base = oos_r2(
                baseline.realized_variance.to_numpy(float),
                baseline[f"forecast_{model}"].to_numpy(float),
                baseline.forecast_mean.to_numpy(float),
            )
            rows.append({
                "horizon": h, "model": model, "n": len(cal),
                "trading_day_scaling_oos_r2": r2_base,
                "calendar_day_scaling_oos_r2": r2_cal,
                "difference_calendar_minus_trading": r2_cal - r2_base,
                "mean_calendar_days": float(calendar_days.loc[cal.index].mean()),
            })
    return pd.DataFrame(rows)


def write_tex_tables(results: Path, alt: pd.DataFrame, sensitivity: pd.DataFrame, calibration: pd.DataFrame, scale: pd.DataFrame, calendar: pd.DataFrame) -> None:
    tables = results.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # Alternative loss: show one common QLIKE floor of 1e-2 of the mean target.
    rows = [r"\begin{tabular}{rllrrr}", r"\toprule", r"$h$ & Comparison & Loss & Gain & HAC $t$ & HAC $p$ \\", r"\midrule"]
    for h in HORIZONS:
        for a, b, label in PAIRS:
            for loss in ("MSE", "MAE", "QLIKE"):
                x = alt[(alt.horizon == h) & (alt.model_a == a) & (alt.model_b == b) & (alt.loss == loss)]
                if loss == "QLIKE":
                    x = x[np.isclose(x.floor_fraction_of_mean_target, 1e-2)]
                r = x.iloc[0]
                rows.append(f"{h} & {label} & {loss} & {r.mean_loss_gain_a_over_b:.6g} & {r.hac_t:.2f} & {r.hac_p:.3f} " + r"\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table_s5_alternative_loss_v21.tex").write_text("\n".join(rows) + "\n")

    rows = [r"\begin{tabular}{rllrrr}", r"\toprule", r"$h$ & Comparison & Method & Parameter & Statistic & $p$ \\", r"\midrule"]
    for h in HORIZONS:
        for a, b, label in PAIRS:
            x = sensitivity[(sensitivity.horizon == h) & (sensitivity.model_a == a) & (sensitivity.model_b == b)]
            for _, r in x.iterrows():
                stat = "" if pd.isna(r.statistic) else f"{r.statistic:.2f}"
                rows.append(f"{h} & {label} & {r.method} & {int(r.parameter)} & {stat} & {r.p_value:.3f} " + r"\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table_s6_inference_sensitivity_v21.tex").write_text("\n".join(rows) + "\n")

    x = calibration[(calibration.horizon == 66) & calibration.model.isin(["q_raw", "q_mz", "har4", "harqts"])]
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Model & Mean forecast/mean RV & Bias & Bias share & Corr. & Eval. slope & OOS $R^2$ \\", r"\midrule"]
    labels = {"q_raw": "Raw option", "q_mz": "MZ option", "har4": "HAR(4)", "harqts": "HAR+option+term"}
    for model in ("q_raw", "q_mz", "har4", "harqts"):
        r = x[x.model == model].iloc[0]
        rows.append(f"{labels[model]} & {r.mean_forecast_to_realized_ratio:.3f} & {r.mean_error_realized_minus_forecast:.6f} & {r.squared_bias_share_of_mse:.3f} & {r.correlation:.3f} & {r.evaluation_slope:.3f} & {r.oos_r2:.3f} " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table_s7_calibration_diagnostics_v21.tex").write_text("\n".join(rows) + "\n")

    maxdiff = float(scale.absolute_difference.max())
    (tables / "table_s8_scale_invariance_v21.tex").write_text(
        "Maximum absolute difference between underlying-variance and idealized-drag out-of-sample R2 across all horizons, models, and leverage loadings: " + f"{maxdiff:.3e}.\n"
    )

    rows = [r"\begin{tabular}{rlrrr}", r"\toprule", r"$h$ & Model & Trading-day scaling & Calendar-day scaling & Difference \\", r"\midrule"]
    labels = {"q_raw": "Raw option", "q_mz": "MZ option", "har4": "HAR(4)", "harqts": "HAR+option+term"}
    for h in HORIZONS:
        for model in ("q_raw", "q_mz", "har4", "harqts"):
            r = calendar[(calendar.horizon == h) & (calendar.model == model)].iloc[0]
            rows.append(f"{h} & {labels[model]} & {r.trading_day_scaling_oos_r2:.3f} & {r.calendar_day_scaling_oos_r2:.3f} & {r.difference_calendar_minus_trading:.3f} " + r"\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "table_s9_calendar_scaling_v21.tex").write_text("\n".join(rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=root)
    args = ap.parse_args()
    root = args.root.resolve()
    results = root / "results"
    forecasts = {h: load_forecasts(results, h) for h in HORIZONS}

    alt = alternative_loss_tests(forecasts)
    sensitivity = inference_sensitivity(forecasts)
    calibration = calibration_diagnostics(forecasts)
    scale = idealized_scale_invariance(forecasts)
    calendar = calendar_scaling_results(root, forecasts)

    alt.to_csv(results / "alternative_loss_tests_v21.csv", index=False)
    sensitivity.to_csv(results / "inference_sensitivity_v21.csv", index=False)
    calibration.to_csv(results / "calibration_diagnostics_v21.csv", index=False)
    scale.to_csv(results / "idealized_scale_invariance_v21.csv", index=False)
    calendar.to_csv(results / "calendar_scaling_results_v21.csv", index=False)
    write_tex_tables(results, alt, sensitivity, calibration, scale, calendar)

    manifest = {
        "version": "2.1",
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "source_forecast_hashes": {str(h): sha256(results / f"variance_forecasts_{h}d.csv") for h in HORIZONS},
        "bootstrap_replications": N_BOOT,
        "qlike_floor_fractions_of_mean_target": [1e-4, 1e-3, 1e-2],
        "inference_parameters": {str(h): sorted({max(1, h - 1), h + 5, 2 * h}) for h in HORIZONS},
        "max_scale_invariance_difference": float(scale.absolute_difference.max()),
    }
    (results / "hardening_manifest_v21.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("V2.1 hardening analyses completed")
    print("\n66-day alternative-loss comparisons:")
    print(alt[(alt.horizon == 66) & ((alt.loss != "QLIKE") | np.isclose(alt.floor_fraction_of_mean_target, 1e-2))][["comparison", "loss", "mean_loss_gain_a_over_b", "hac_t", "hac_p", "block_p"]].to_string(index=False))
    print("\n66-day calibration diagnostics:")
    print(calibration[(calibration.horizon == 66) & calibration.model.isin(["q_raw", "q_mz", "har4", "harqts"])][["model", "mean_forecast_to_realized_ratio", "squared_bias_share_of_mse", "correlation", "oos_r2"]].to_string(index=False))
    print("\nMaximum scale-invariance difference:", scale.absolute_difference.max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
