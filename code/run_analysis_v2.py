#!/usr/bin/env python3
"""V2 replication analysis for option-implied variance and SPX LETF drag.

All forecasts are genuinely recursive. At forecast origin t, a k-day outcome
is eligible for estimation only if its endpoint is observed by t. The script
separates the common underlying-variance forecasting problem from the
fund-specific mapping into observed market-price drag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

FUNDS: Dict[str, float] = {"SSO": 2.0, "UPRO": 3.0, "SDS": -2.0, "SPXU": -3.0}
HORIZONS: Tuple[int, ...] = (5, 22, 66)
MIN_TRAIN = 750
HAC_EXTRA = 5
SEED = 20260816
N_BOOT = 4999
MODELS = ("mean", "q_raw", "q_mz", "har3", "har4", "harq", "harqts")


@dataclass(frozen=True)
class Spec:
    name: str = "baseline"
    signal_lag: int = 0
    rolling_window: Optional[int] = None
    use_vix_at_66: bool = False
    floor_zero: bool = True
    target: str = "rv"  # rv or bipower


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def forward_sum(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k).sum().shift(-k)


def prefix(a: np.ndarray) -> np.ndarray:
    shape = (1,) + a.shape[1:]
    return np.concatenate([np.zeros(shape, dtype=float), np.cumsum(a, axis=0)], axis=0)


def range_sum(p: np.ndarray, start: int, end: int):
    if end < start:
        return np.zeros(p.shape[1:], dtype=float)
    return p[end + 1] - p[start]


def start_for(end: int, window: Optional[int]) -> int:
    return 0 if window is None else max(0, end - window + 1)


class RecursiveOLS:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X[:, None]
        Z = np.column_stack([np.ones(len(X)), X])
        valid = np.isfinite(Z).all(axis=1) & np.isfinite(y)
        Z0 = np.where(valid[:, None], Z, 0.0)
        y0 = np.where(valid, y, 0.0)
        self.n = prefix(valid.astype(float))
        self.xx = prefix(np.einsum("ni,nj->nij", Z0, Z0))
        self.xy = prefix(Z0 * y0[:, None])
        self.p = Z.shape[1]

    def fit(self, start: int, end: int) -> tuple[np.ndarray, int]:
        n = int(round(float(range_sum(self.n, start, end))))
        if n < max(MIN_TRAIN, self.p + 5):
            return np.full(self.p, np.nan), n
        xx = range_sum(self.xx, start, end)
        xy = range_sum(self.xy, start, end)
        try:
            b = np.linalg.solve(xx, xy)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(xx, xy, rcond=None)[0]
        return np.asarray(b, float), n

    def predict(self, x: np.ndarray, start: int, end: int, floor: bool) -> tuple[float, np.ndarray, int]:
        if not np.isfinite(x).all():
            return float("nan"), np.full(self.p, np.nan), 0
        b, n = self.fit(start, end)
        if not np.isfinite(b).all():
            return float("nan"), b, n
        v = float(np.r_[1.0, x] @ b)
        return (max(0.0, v) if floor else v), b, n


class RecursiveMean:
    def __init__(self, y: np.ndarray):
        valid = np.isfinite(y)
        self.s = prefix(np.where(valid, y, 0.0))
        self.n = prefix(valid.astype(float))

    def value(self, start: int, end: int) -> tuple[float, int]:
        n = int(round(float(range_sum(self.n, start, end))))
        if n < MIN_TRAIN:
            return float("nan"), n
        return float(range_sum(self.s, start, end) / n), n


def hac_test(x: Iterable[float], lags: int) -> tuple[float, float, float, int]:
    a = np.asarray(list(x), float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return float("nan"), float("nan"), float("nan"), len(a)
    fit = sm.OLS(a, np.ones(len(a))).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)}, use_t=False)
    return float(a.mean()), float(fit.tvalues[0]), float(fit.pvalues[0]), len(a)


def block_pvalue(x: Iterable[float], block: int, seed: int, reps: int = N_BOOT) -> float:
    a = np.asarray(list(x), float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 20:
        return float("nan")
    obs = float(a.mean())
    z = a - obs
    block = max(1, min(int(block), n))
    nb = int(math.ceil(n / block))
    offsets = np.arange(block)
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < reps:
        batch = min(200, reps - done)
        starts = rng.integers(0, n, size=(batch, nb))
        idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(batch, -1)[:, :n]
        sims = z[idx].mean(axis=1)
        exceed += int((np.abs(sims) >= abs(obs)).sum())
        done += batch
    return float((exceed + 1) / (reps + 1))


def load_data(market: Path, rates: Path) -> tuple[pd.DataFrame, pd.Series]:
    px = pd.read_csv(market, parse_dates=["Date"]).set_index("Date").sort_index()
    px = px[["SPY", *FUNDS, "VIX", "VIX3M"]].astype(float).dropna()
    rf = pd.read_csv(rates, parse_dates=["observation_date"]).set_index("observation_date")["DTB3"].astype(float)
    rf = rf.reindex(px.index).ffill().bfill() / 100.0
    return px, rf


def check_data(px: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    lr = np.log(px.SPY).diff()
    rows = [
        ("rows", len(px), len(px) == 4023),
        ("start", str(px.index.min().date()), str(px.index.min().date()) == "2010-01-04"),
        ("end", str(px.index.max().date()), str(px.index.max().date()) == "2025-12-30"),
        ("monotonic", bool(px.index.is_monotonic_increasing), bool(px.index.is_monotonic_increasing)),
        ("duplicates", int(px.index.duplicated().sum()), not px.index.has_duplicates),
        ("missing_market", int(px.isna().sum().sum()), int(px.isna().sum().sum()) == 0),
        ("missing_rates", int(rf.isna().sum()), int(rf.isna().sum()) == 0),
        ("positive", bool((px > 0).all().all()), bool((px > 0).all().all())),
        ("vix_peak_date", str(px.VIX.idxmax().date()), str(px.VIX.idxmax().date()) == "2020-03-16"),
        ("spy_worst_date", str(lr.idxmin().date()), str(lr.idxmin().date()) == "2020-03-16"),
    ]
    logs = np.log(px)
    for f, beta in FUNDS.items():
        b = float(logs[f].diff().cov(lr) / lr.var())
        rows.append((f"daily_beta_{f}", b, abs(b - beta) < 0.08))
    return pd.DataFrame(rows, columns=["check", "value", "pass"])


def daily_target(spy_ret: pd.Series, kind: str) -> pd.Series:
    if kind == "rv":
        return spy_ret.pow(2)
    if kind == "bipower":
        return (math.pi / 2.0) * spy_ret.abs() * spy_ret.shift(1).abs()
    raise ValueError(kind)


def generate_variance(px: pd.DataFrame, h: int, spec: Spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret = np.log(px.SPY).diff()
    vday = daily_target(ret, spec.target)
    y_s = forward_sum(vday, h)
    vix = px.VIX.shift(spec.signal_lag)
    vix3m = px.VIX3M.shift(spec.signal_lag)
    q_vix = (vix / 100.0).pow(2) * h / 252.0
    q_vix3m = (vix3m / 100.0).pow(2) * h / 252.0
    q_s = q_vix if (h != 66 or spec.use_vix_at_66) else q_vix3m
    q_name = "VIX" if (h != 66 or spec.use_vix_at_66) else "VIX3M"
    term = ((vix3m / 100.0).pow(2) - (vix / 100.0).pow(2)) * h / 252.0
    X3 = pd.DataFrame({
        "d": vday,
        "w": vday.rolling(5).mean(),
        "m": vday.rolling(22).mean(),
    }, index=px.index)
    X4 = X3.assign(qtr=vday.rolling(66).mean())
    Xs = {
        "q_mz": q_s.to_numpy(float)[:, None],
        "har3": X3.to_numpy(float),
        "har4": X4.to_numpy(float),
        "harq": np.column_stack([X4.to_numpy(float), q_s.to_numpy(float)]),
        "harqts": np.column_stack([X4.to_numpy(float), q_s.to_numpy(float), term.to_numpy(float)]),
    }
    y = y_s.to_numpy(float)
    models = {name: RecursiveOLS(X, y) for name, X in Xs.items()}
    mean_model = RecursiveMean(y)
    rows: list[dict] = []
    coeffs: list[dict] = []
    for pos in range(len(px) - h):
        end = pos - h
        if end < 0:
            continue
        start = start_for(end, spec.rolling_window)
        mean, nmean = mean_model.value(start, end)
        if nmean < MIN_TRAIN:
            continue
        preds = {"mean": mean, "q_raw": float(q_s.iloc[pos])}
        ns = [nmean]
        for name, X in Xs.items():
            pred, coef, n = models[name].predict(X[pos], start, end, spec.floor_zero)
            preds[name] = pred
            ns.append(n)
            if name in {"q_mz", "harqts"} and np.isfinite(pred):
                rec = {"Date": px.index[pos], "horizon": h, "spec": spec.name, "model": name, "n_train": n}
                for j, val in enumerate(coef):
                    rec[f"coef_{j}"] = float(val)
                coeffs.append(rec)
        vals = [y[pos], *preds.values()]
        if not np.isfinite(vals).all():
            continue
        rows.append({
            "Date": px.index[pos],
            "position": pos,
            "target_end": px.index[pos + h],
            "training_end": px.index[end],
            "horizon": h,
            "spec": spec.name,
            "target_type": spec.target,
            "option_signal": q_name,
            "n_train": int(min(ns)),
            "realized_variance": float(y[pos]),
            "vix": float(px.VIX.iloc[pos]),
            "vix3m": float(px.VIX3M.iloc[pos]),
            "term_variance": float(term.iloc[pos]),
            **{f"forecast_{k}": float(v) for k, v in preds.items()},
        })
    return pd.DataFrame(rows).set_index("Date"), pd.DataFrame(coeffs)


def eval_models(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna()
    y = d.realized_variance
    b = (y - d.forecast_mean).pow(2)
    denom = float(b.sum())
    rows = []
    for model in MODELS:
        p = d[f"forecast_{model}"]
        loss = (y - p).pow(2)
        rows.append({
            "horizon": int(d.horizon.iloc[0]),
            "spec": d.spec.iloc[0],
            "target_type": d.target_type.iloc[0],
            "model": model,
            "n": len(d),
            "start": str(d.index.min().date()),
            "end": str(d.index.max().date()),
            "oos_r2": 0.0 if model == "mean" else 1.0 - float(loss.sum()) / denom,
            "rmse": math.sqrt(float(loss.mean())),
            "mae": float((y - p).abs().mean()),
            "mean_realized": float(y.mean()),
            "mean_forecast": float(p.mean()),
        })
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, a: str, b: str, nested: bool, do_boot: bool, seed: int) -> dict:
    d = df[["realized_variance", f"forecast_{a}", f"forecast_{b}"]].dropna()
    y = d.realized_variance
    pa = d[f"forecast_{a}"]
    pb = d[f"forecast_{b}"]
    diff = (y - pb).pow(2) - (y - pa).pow(2)  # positive => A lower loss
    h = int(df.horizon.iloc[0])
    m, t, p, n = hac_test(diff, h + HAC_EXTRA)
    out = {
        "horizon": h, "spec": df.spec.iloc[0], "model_a": a, "model_b": b,
        "n": n, "mean_loss_gain_a_over_b": m, "hac_t": t, "hac_p": p,
        "block_p": block_pvalue(diff, h + HAC_EXTRA, seed) if do_boot else float("nan"),
        "cw_mean": float("nan"), "cw_t": float("nan"), "cw_p_one_sided": float("nan"),
    }
    if nested:
        cw = diff + (pb - pa).pow(2)
        cm, ct, _, _ = hac_test(cw, h + HAC_EXTRA)
        out.update({"cw_mean": cm, "cw_t": ct, "cw_p_one_sided": float(norm.sf(ct))})
    return out


def encompassing(df: pd.DataFrame, option_model: str) -> dict:
    d = df[["realized_variance", "forecast_har4", f"forecast_{option_model}"]].dropna()
    X = sm.add_constant(d[["forecast_har4", f"forecast_{option_model}"]])
    h = int(df.horizon.iloc[0])
    fit = sm.OLS(d.realized_variance, X).fit(cov_type="HAC", cov_kwds={"maxlags": h + HAC_EXTRA}, use_t=False)
    return {
        "horizon": h, "option_model": option_model, "n": int(fit.nobs),
        "alpha": float(fit.params.const), "p_alpha": float(fit.pvalues.const),
        "beta_physical": float(fit.params.forecast_har4), "p_physical": float(fit.pvalues.forecast_har4),
        "beta_option": float(fit.params[f"forecast_{option_model}"]), "p_option": float(fit.pvalues[f"forecast_{option_model}"]),
        "r2": float(fit.rsquared),
    }


def variance_suite(forecasts: Dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf = pd.concat([eval_models(forecasts[h]) for h in HORIZONS], ignore_index=True)
    pairs = [
        ("q_mz", "q_raw", True),
        ("q_mz", "har4", False),
        ("harq", "har4", True),
        ("harqts", "har4", True),
        ("harqts", "q_mz", False),
        ("q_raw", "har4", False),
    ]
    tests = []
    for h in HORIZONS:
        for j, (a, b, nested) in enumerate(pairs):
            tests.append(compare(forecasts[h], a, b, nested, True, SEED + 1000 * j + h))
    enc = [encompassing(forecasts[h], m) for h in HORIZONS for m in ("q_raw", "q_mz")]
    return perf, pd.DataFrame(tests), pd.DataFrame(enc)


def subset_result(df: pd.DataFrame, name: str, mask: np.ndarray) -> Optional[dict]:
    sub = df.loc[mask].copy()
    if len(sub) < 50:
        return None
    p = eval_models(sub).set_index("model").oos_r2.to_dict()
    out = {"horizon": int(df.horizon.iloc[0]), "subset": name, "n": len(sub), "start": str(sub.index.min().date()), "end": str(sub.index.max().date())}
    out.update({f"r2_{m}": float(p[m]) for m in MODELS})
    for a, b, nested in [("q_mz", "q_raw", True), ("q_mz", "har4", False), ("harqts", "har4", True)]:
        z = compare(sub, a, b, nested, False, SEED)
        out[f"{a}_vs_{b}_t"] = z["hac_t"]
        out[f"{a}_vs_{b}_p"] = z["hac_p"]
    return out


def regimes_66(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    ends = pd.to_datetime(df.target_end)
    q75 = float(df.vix.quantile(.75)); q90 = float(df.vix.quantile(.90)); rv99 = float(df.realized_variance.quantile(.99))
    event_years = {2018, 2020, 2022}
    no_event = np.array([not any(y in event_years for y in range(a.year, b.year + 1)) for a, b in zip(idx, ends)])
    masks = {
        "full_sample": np.ones(len(df), dtype=bool),
        "pre_2020": idx < pd.Timestamp("2020-01-01"),
        "2020_and_after": idx >= pd.Timestamp("2020-01-01"),
        "exclude_calendar_2020_origins": idx.year != 2020,
        "exclude_windows_intersecting_2018_2020_2022": no_event,
        "vix_below_75th_percentile": df.vix.to_numpy() < q75,
        "vix_at_or_above_75th_percentile": df.vix.to_numpy() >= q75,
        "vix_at_or_above_90th_percentile": df.vix.to_numpy() >= q90,
        "vix_term_contango": df.vix3m.to_numpy() > df.vix.to_numpy(),
        "vix_term_backwardation": df.vix3m.to_numpy() <= df.vix.to_numpy(),
        "exclude_top_1pct_realized_variance": df.realized_variance.to_numpy() <= rv99,
    }
    rows = [subset_result(df, name, mask) for name, mask in masks.items()]
    return pd.DataFrame([x for x in rows if x is not None])


def nonoverlap(forecasts: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, df in forecasts.items():
        for off in range(h):
            sub = df.iloc[off::h]
            if len(sub) < 25:
                continue
            p = eval_models(sub).set_index("model").oos_r2.to_dict()
            rows.append({
                "horizon": h, "offset": off, "n": len(sub),
                **{f"r2_{m}": float(p[m]) for m in MODELS},
                "delta_qmz_qraw": float(p["q_mz"] - p["q_raw"]),
                "delta_harqts_har4": float(p["harqts"] - p["har4"]),
            })
    return pd.DataFrame(rows)


def common_origins(forecasts: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    common = forecasts[5].index
    for h in (22, 66):
        common = common.intersection(forecasts[h].index)
    parts = []
    for h in HORIZONS:
        p = eval_models(forecasts[h].loc[common])
        p["sample"] = "common_origins"
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def rolling_loss(df: pd.DataFrame, window: int = 504) -> pd.DataFrame:
    y = df.realized_variance
    scale = float(((y - df.forecast_mean) ** 2).mean())
    out = pd.DataFrame(index=df.index)
    out["qmz_vs_qraw"] = (((y - df.forecast_q_raw) ** 2 - (y - df.forecast_q_mz) ** 2) / scale).rolling(window, min_periods=252).mean()
    out["qmz_vs_har4"] = (((y - df.forecast_har4) ** 2 - (y - df.forecast_q_mz) ** 2) / scale).rolling(window, min_periods=252).mean()
    out["harqts_vs_har4"] = (((y - df.forecast_har4) ** 2 - (y - df.forecast_harqts) ** 2) / scale).rolling(window, min_periods=252).mean()
    return out


def fund_mapping(px: pd.DataFrame, rf: pd.Series, vf: pd.DataFrame, fund: str, h: int) -> tuple[pd.DataFrame, dict]:
    beta = FUNDS[fund]
    gamma = .5 * beta * (beta - 1.0)
    logs = np.log(px)
    drag_s = logs[fund].shift(-h) - logs[fund] - beta * (logs.SPY.shift(-h) - logs.SPY)
    rv_full = forward_sum(logs.SPY.diff().pow(2), h)
    residual_s = drag_s + gamma * rv_full
    rate_x = (rf * h / 252.0).to_numpy(float)[:, None]
    drag = drag_s.to_numpy(float); residual = residual_s.to_numpy(float)
    m_drag = RecursiveMean(drag); m_res = RecursiveMean(residual); rate_model = RecursiveOLS(rate_x, residual)
    rows = []
    for date, vr in vf.iterrows():
        pos = int(vr.position); end = pos - h
        if end < 0: continue
        mean_drag, n1 = m_drag.value(0, end)
        c, n2 = m_res.value(0, end)
        c_rate, b_rate, n3 = rate_model.predict(rate_x[pos], 0, end, False)
        vals = [drag[pos], mean_drag, c, c_rate]
        if min(n1, n2, n3) < MIN_TRAIN or not np.isfinite(vals).all():
            continue
        row = {
            "Date": date, "fund": fund, "horizon": h, "position": pos,
            "target_end": vr.target_end, "beta": beta, "gamma": gamma,
            "realized_drag": float(drag[pos]), "realized_variance": float(vr.realized_variance),
            "historical_mean": mean_drag, "c_intercept": c, "c_rate": c_rate,
            "rate_slope": float(b_rate[1]),
        }
        for model in ("q_raw", "q_mz", "har4", "harq", "harqts"):
            vhat = float(vr[f"forecast_{model}"])
            row[f"forecast_{model}"] = c - gamma * vhat
            row[f"forecast_{model}_rate"] = c_rate - gamma * vhat
        rows.append(row)
    dff = pd.DataFrame(rows).set_index("Date")

    diag = pd.concat([drag_s.rename("drag"), rv_full.rename("rv")], axis=1).dropna()
    fit = sm.OLS(diag.drag, sm.add_constant(diag.rv)).fit(cov_type="HAC", cov_kwds={"maxlags": h + HAC_EXTRA}, use_t=False)
    theory = -gamma * diag.rv
    resid_fixed = diag.drag - theory
    diagnostics = {
        "fund": fund, "horizon": h, "n": len(diag), "beta": beta, "gamma": gamma,
        "estimated_rv_slope": float(fit.params.rv), "estimated_rv_slope_se": float(fit.bse.rv),
        "slope_minus_theory": float(fit.params.rv + gamma), "slope_theory_wald_t": float((fit.params.rv + gamma) / fit.bse.rv),
        "mapping_r2": float(fit.rsquared), "corr_drag_theory": float(diag.drag.corr(theory)),
        "sd_theory": float(theory.std()), "sd_residual": float(resid_fixed.std()),
        "signal_noise_ratio": float(theory.std() / resid_fixed.std()),
        "variance_share_ratio": float(theory.var() / diag.drag.var()),
    }
    return dff, diagnostics


def eval_fund(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(); y = d.realized_drag
    denom = float(((y - d.historical_mean) ** 2).sum())
    rows = []
    for model in ("q_raw", "q_mz", "har4", "harq", "harqts"):
        for mapping in ("intercept", "rate"):
            col = f"forecast_{model}" if mapping == "intercept" else f"forecast_{model}_rate"
            loss = (y - d[col]) ** 2
            rows.append({
                "fund": d.fund.iloc[0], "horizon": int(d.horizon.iloc[0]), "model": model, "mapping": mapping,
                "n": len(d), "oos_r2": 1.0 - float(loss.sum()) / denom,
                "rmse": math.sqrt(float(loss.mean())), "mae": float((y - d[col]).abs().mean()),
            })
    return pd.DataFrame(rows)


def fund_test(df: pd.DataFrame, a: str, b: str, mapping: str) -> dict:
    ca = f"forecast_{a}" if mapping == "intercept" else f"forecast_{a}_rate"
    cb = f"forecast_{b}" if mapping == "intercept" else f"forecast_{b}_rate"
    d = df[["realized_drag", ca, cb]].dropna()
    diff = (d.realized_drag - d[cb]) ** 2 - (d.realized_drag - d[ca]) ** 2
    h = int(df.horizon.iloc[0]); m, t, p, n = hac_test(diff, h + HAC_EXTRA)
    return {"fund": df.fund.iloc[0], "horizon": h, "mapping": mapping, "model_a": a, "model_b": b, "n": n, "mean_loss_gain": m, "hac_t": t, "hac_p": p}


def fund_panel(series: Dict[tuple[str, int], pd.DataFrame], h: int, a: str, b: str) -> dict:
    cols = []
    for fund in FUNDS:
        d = series[(fund, h)].dropna()
        diff = (d.realized_drag - d[f"forecast_{b}"]) ** 2 - (d.realized_drag - d[f"forecast_{a}"]) ** 2
        scale = float(((d.realized_drag - d.historical_mean) ** 2).mean())
        cols.append((diff / scale).rename(fund))
    D = pd.concat(cols, axis=1).dropna(); z = D.mean(axis=1)
    m, t, p, n = hac_test(z, h + HAC_EXTRA)
    corr = float(D.corr().to_numpy()[np.triu_indices(len(FUNDS), 1)].mean())
    return {"horizon": h, "model_a": a, "model_b": b, "n_dates": n, "mean_normalized_loss_gain": m, "hac_t": t, "hac_p": p, "block_p": block_pvalue(z, h + HAC_EXTRA, SEED + 5000 + h), "mean_pairwise_correlation": corr}


def drag_materiality(px: pd.DataFrame) -> pd.DataFrame:
    logs = np.log(px); rows = []
    for h in HORIZONS:
        for fund, beta in FUNDS.items():
            drag = (logs[fund].shift(-h) - logs[fund] - beta * (logs.SPY.shift(-h) - logs.SPY)).dropna()
            rows.append({
                "fund": fund, "horizon": h, "n": len(drag),
                "mean_drag_pct": 100 * float(drag.mean()), "median_drag_pct": 100 * float(drag.median()),
                "mean_abs_drag_pct": 100 * float(drag.abs().mean()),
                "p90_abs_drag_pct": 100 * float(drag.abs().quantile(.90)),
                "p95_abs_drag_pct": 100 * float(drag.abs().quantile(.95)),
            })
    return pd.DataFrame(rows)


def leakage_test(px: pd.DataFrame, baseline: Dict[int, pd.DataFrame], cutoff: str = "2019-12-31") -> pd.DataFrame:
    alt = px.copy(); mask = alt.index > pd.Timestamp(cutoff); rng = np.random.default_rng(SEED)
    for c in alt.columns:
        alt.loc[mask, c] = alt.loc[mask, c].to_numpy() * np.exp(rng.normal(0, .15, mask.sum()))
    rows = []
    for h in HORIZONS:
        af, _ = generate_variance(alt, h, Spec(name="perturbed"))
        cols = [f"forecast_{m}" for m in MODELS]
        a = baseline[h].loc[:cutoff, cols]; b = af.loc[:cutoff, cols]; ix = a.index.intersection(b.index)
        md = float(np.nanmax(np.abs(a.loc[ix].to_numpy() - b.loc[ix].to_numpy())))
        rows.append({"horizon": h, "n_compared": len(ix), "max_abs_difference": md, "pass": md < 1e-12})
    return pd.DataFrame(rows)


def make_figures(perf: pd.DataFrame, forecasts: Dict[int, pd.DataFrame], diagnostics: pd.DataFrame, fund_perf: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    P = perf.pivot(index="horizon", columns="model", values="oos_r2")
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for m, label, marker, ls in [
        ("q_raw", "Unadjusted option signal", "o", "-"),
        ("q_mz", "MZ-calibrated option signal", "s", "--"),
        ("har4", "HAR physical forecast", "^", ":"),
        ("harqts", "HAR + option + term structure", "D", "-."),
    ]:
        ax.plot(P.index, P[m], marker=marker, linestyle=ls, linewidth=1.8, label=label)
    ax.axhline(0, linewidth=.8); ax.set_xticks(HORIZONS); ax.set_xlabel("Forecast horizon (trading days)"); ax.set_ylabel("Out-of-sample $R^2$")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(out / "Figure_1_Variance_OOS_R2.pdf", bbox_inches="tight"); fig.savefig(out / "Figure_1_Variance_OOS_R2.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    delta = pd.DataFrame(index=HORIZONS)
    delta["MZ minus raw option"] = P.q_mz - P.q_raw
    delta["MZ option minus HAR"] = P.q_mz - P.har4
    delta["HAR+option+term minus HAR"] = P.harqts - P.har4
    fig, ax = plt.subplots(figsize=(7.4, 4.6)); x = np.arange(3); w = .24
    for j, c in enumerate(delta.columns): ax.bar(x + (j - 1) * w, delta[c].values, width=w, label=c)
    ax.axhline(0, linewidth=.8); ax.set_xticks(x, [str(h) for h in HORIZONS]); ax.set_xlabel("Forecast horizon (trading days)"); ax.set_ylabel("Incremental out-of-sample $R^2$")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(out / "Figure_2_Incremental_Option_R2.pdf", bbox_inches="tight"); fig.savefig(out / "Figure_2_Incremental_Option_R2.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    roll = rolling_loss(forecasts[66])
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    for c, label, ls in [("qmz_vs_qraw", "MZ versus raw option", "-"), ("qmz_vs_har4", "MZ option versus HAR", ":"), ("harqts_vs_har4", "HAR+option+term versus HAR", "--")]:
        ax.plot(roll.index, roll[c], label=label, linestyle=ls, linewidth=1.4)
    ax.axhline(0, linewidth=.8); ax.set_xlabel("Forecast origin"); ax.set_ylabel("Two-year rolling standardized loss improvement"); ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(out / "Figure_3_Rolling_Incremental_Loss.pdf", bbox_inches="tight"); fig.savefig(out / "Figure_3_Rolling_Incremental_Loss.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    D = diagnostics[diagnostics.horizon == 66].set_index("fund")
    G = fund_perf[(fund_perf.horizon == 66) & (fund_perf.mapping == "intercept")].pivot(index="fund", columns="model", values="oos_r2")
    D = D.join((G.q_mz - G.q_raw).rename("gain"))
    fig, ax = plt.subplots(figsize=(6.8, 4.6)); ax.scatter(D.signal_noise_ratio, D.gain, s=58)
    for f, r in D.iterrows(): ax.annotate(f, (r.signal_noise_ratio, r.gain), xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, linewidth=.8); ax.set_xlabel("Theoretical variance-component SD / residual SD"); ax.set_ylabel("66-day drag $R^2$ gain: MZ minus raw option"); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "Figure_4_Fund_Heterogeneity.pdf", bbox_inches="tight"); fig.savefig(out / "Figure_4_Fund_Heterogeneity.png", dpi=600, bbox_inches="tight"); plt.close(fig)


def write_tables(perf: pd.DataFrame, tests: pd.DataFrame, enc: pd.DataFrame, regimes: pd.DataFrame, diag: pd.DataFrame, fund_perf: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    P = perf.pivot(index="horizon", columns="model", values="oos_r2")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Horizon & Raw option & MZ option & HAR(3) & HAR(4) & HAR+option & HAR+option+term \\", r"\midrule"]
    for h in HORIZONS:
        r = P.loc[h]
        lines.append(f"{h} & {r.q_raw:.3f} & {r.q_mz:.3f} & {r.har3:.3f} & {r.har4:.3f} & {r.harq:.3f} & {r.harqts:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_variance_performance.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Horizon & Comparison & $\Delta R^2_{OOS}$ & HAC $t$ & HAC $p$ & Block $p$ & CW $p$ \\", r"\midrule"]
    for h in HORIZONS:
        for a, b, label in [("q_mz", "q_raw", "MZ vs raw option"), ("q_mz", "har4", "MZ option vs HAR(4)"), ("harqts", "har4", "HAR+option+term vs HAR(4)")]:
            r = tests[(tests.horizon == h) & (tests.model_a == a) & (tests.model_b == b)].iloc[0]
            delta = float(P.loc[h, a] - P.loc[h, b])
            cw = "" if pd.isna(r.cw_p_one_sided) else f"{r.cw_p_one_sided:.3f}"
            lines.append(f"{h} & {label} & {delta:.3f} & {r.hac_t:.2f} & {r.hac_p:.3f} & {r.block_p:.3f} & {cw} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table2_pairwise_tests.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Horizon & Option forecast & $\beta_{HAR}$ & $p$ & $\beta_{option}$ & $p$ & $R^2$ \\", r"\midrule"]
    for h in HORIZONS:
        for om, label in [("q_raw", "Raw"), ("q_mz", "MZ-calibrated")]:
            r = enc[(enc.horizon == h) & (enc.option_model == om)].iloc[0]
            lines.append(f"{h} & {label} & {r.beta_physical:.3f} & {r.p_physical:.3f} & {r.beta_option:.3f} & {r.p_option:.3f} & {r.r2:.3f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table3_encompassing.tex").write_text("\n".join(lines) + "\n")

    wanted = ["full_sample", "pre_2020", "2020_and_after", "exclude_calendar_2020_origins", "exclude_windows_intersecting_2018_2020_2022", "vix_below_75th_percentile", "vix_at_or_above_75th_percentile", "exclude_top_1pct_realized_variance"]
    label = {"full_sample": "Full sample", "pre_2020": "Pre-2020", "2020_and_after": "2020 onward", "exclude_calendar_2020_origins": "Exclude 2020 origins", "exclude_windows_intersecting_2018_2020_2022": "Exclude windows touching 2018/2020/2022", "vix_below_75th_percentile": "VIX below 75th percentile", "vix_at_or_above_75th_percentile": "VIX at/above 75th percentile", "exclude_top_1pct_realized_variance": "Exclude top 1 percent RV"}
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Sample & $N$ & Raw option & MZ option & HAR(4) & HAR+option+term \\", r"\midrule"]
    for w in wanted:
        r = regimes[regimes.subset == w].iloc[0]
        lines.append(f"{label[w]} & {int(r.n)} & {r.r2_q_raw:.3f} & {r.r2_q_mz:.3f} & {r.r2_har4:.3f} & {r.r2_harqts:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table4_regime_robustness.tex").write_text("\n".join(lines) + "\n")

    D = diag[diag.horizon == 66].set_index("fund")
    G = fund_perf[(fund_perf.horizon == 66) & (fund_perf.mapping == "intercept")].pivot(index="fund", columns="model", values="oos_r2")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Fund & $\beta$ & $\gamma$ & Mapping $R^2$ & Signal/noise & Raw drag $R^2$ & MZ drag $R^2$ \\", r"\midrule"]
    for f, beta in FUNDS.items():
        r = D.loc[f]
        lines.append(f"{f} & {beta:.0f} & {r.gamma:.0f} & {r.mapping_r2:.3f} & {r.signal_noise_ratio:.3f} & {G.loc[f, 'q_raw']:.3f} & {G.loc[f, 'q_mz']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table5_fund_mapping.tex").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(); root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", type=Path, default=root / "data" / "market_data.csv")
    ap.add_argument("--rates", type=Path, default=root / "data" / "rates.csv")
    ap.add_argument("--output", type=Path, default=root)
    ap.add_argument("--skip-leakage", action="store_true")
    args = ap.parse_args(); out = args.output.resolve()
    for d in (out / "results", out / "figures", out / "tables", out / "logs"): d.mkdir(parents=True, exist_ok=True)

    px, rf = load_data(args.data, args.rates)
    checks = check_data(px, rf); checks.to_csv(out / "results" / "data_checks.csv", index=False)
    if not checks["pass"].all(): raise RuntimeError("data checks failed")

    baseline: Dict[int, pd.DataFrame] = {}; coef = []
    for h in HORIZONS:
        f, c = generate_variance(px, h, Spec())
        baseline[h] = f; coef.append(c); f.to_csv(out / "results" / f"variance_forecasts_{h}d.csv")
    pd.concat(coef, ignore_index=True).to_csv(out / "results" / "recursive_coefficients.csv", index=False)
    perf, tests, enc = variance_suite(baseline)
    perf.to_csv(out / "results" / "variance_model_results.csv", index=False)
    tests.to_csv(out / "results" / "variance_forecast_tests.csv", index=False)
    enc.to_csv(out / "results" / "forecast_encompassing.csv", index=False)

    reg = regimes_66(baseline[66]); reg.to_csv(out / "results" / "regime_results_66d.csv", index=False)
    nonoverlap(baseline).to_csv(out / "results" / "nonoverlap_results.csv", index=False)
    common_origins(baseline).to_csv(out / "results" / "common_origin_results.csv", index=False)
    rolling_loss(baseline[66]).to_csv(out / "results" / "rolling_loss_66d.csv")

    robust = []
    specs = [Spec(name="lagged_one_day", signal_lag=1), Spec(name="rolling_five_year", rolling_window=1260), Spec(name="wrong_maturity_vix_66", use_vix_at_66=True), Spec(name="no_floor", floor_zero=False), Spec(name="daily_bipower", target="bipower")]
    for s in specs:
        hs = (66,) if s.name == "wrong_maturity_vix_66" else HORIZONS
        for h in hs:
            f, _ = generate_variance(px, h, s); p = eval_models(f); p["robustness"] = s.name; robust.append(p)
            if h == 66 or s.name == "daily_bipower": f.to_csv(out / "results" / f"variance_forecasts_{s.name}_{h}d.csv")
    pd.concat(robust, ignore_index=True).to_csv(out / "results" / "robustness_model_results.csv", index=False)

    fund_series: Dict[tuple[str, int], pd.DataFrame] = {}; fund_perf = []; diag = []; ftests = []
    for h in HORIZONS:
        for fund in FUNDS:
            f, d = fund_mapping(px, rf, baseline[h], fund, h); fund_series[(fund, h)] = f; diag.append(d); fund_perf.append(eval_fund(f)); f.to_csv(out / "results" / f"fund_forecasts_{fund}_{h}d.csv")
            for a, b in [("q_mz", "q_raw"), ("q_mz", "har4"), ("harqts", "har4")]:
                for mapping in ("intercept", "rate"): ftests.append(fund_test(f, a, b, mapping))
    fund_perf_df = pd.concat(fund_perf, ignore_index=True); diag_df = pd.DataFrame(diag)
    fund_perf_df.to_csv(out / "results" / "fund_model_results.csv", index=False); diag_df.to_csv(out / "results" / "fund_decomposition.csv", index=False); pd.DataFrame(ftests).to_csv(out / "results" / "fund_forecast_tests.csv", index=False)
    pd.DataFrame([fund_panel(fund_series, h, a, b) for h in HORIZONS for a, b in [("q_mz", "q_raw"), ("q_mz", "har4"), ("harqts", "har4")]]).to_csv(out / "results" / "fund_panel_tests.csv", index=False)
    drag_materiality(px).to_csv(out / "results" / "drag_materiality.csv", index=False)

    if not args.skip_leakage:
        leak = leakage_test(px, baseline); leak.to_csv(out / "results" / "future_perturbation_test.csv", index=False)
        if not leak["pass"].all(): raise RuntimeError("leakage test failed")

    make_figures(perf, baseline, diag_df, fund_perf_df, out / "figures")
    write_tables(perf, tests, enc, reg, diag_df, fund_perf_df, out / "tables")
    manifest = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "python": sys.version, "platform": platform.platform(), "market_data_sha256": sha256(args.data), "rates_sha256": sha256(args.rates), "rows": len(px), "start": str(px.index.min().date()), "end": str(px.index.max().date()), "horizons": list(HORIZONS), "funds": FUNDS, "min_train": MIN_TRAIN, "seed": SEED, "bootstrap_replications": N_BOOT}
    (out / "results" / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("V2 analysis completed successfully")
    print(perf.pivot(index="horizon", columns="model", values="oos_r2").round(4).to_string())
    print("\nSelected comparisons")
    print(tests[((tests.model_a == "q_mz") & (tests.model_b.isin(["q_raw", "har4"]))) | ((tests.model_a == "harqts") & (tests.model_b == "har4"))][["horizon", "model_a", "model_b", "hac_t", "hac_p", "block_p", "cw_p_one_sided"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
