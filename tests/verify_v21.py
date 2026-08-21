#!/usr/bin/env python3
"""Reproducibility and consistency checks for Version 2.1.

The program verifies internal reproducibility from the frozen inputs. It does not
claim external validity, data-licensing clearance, or that unavailable intraday,
NAV, VIX9D, or cross-market extensions were completed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "replication"
RES = REP / "results"
PAPER = ROOT / "paper"
VER = ROOT / "verification_v21"
HORIZONS = (5, 22, 66)
MODELS = ("mean", "q_raw", "q_mz", "har3", "har4", "harq", "harqts")

checks: list[dict] = []


def record(name: str, passed: bool, detail: object = "") -> None:
    checks.append({"check": name, "pass": bool(passed), "detail": str(detail)})
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def oos_r2(y: np.ndarray, f: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.sum((y - f) ** 2)) / float(np.sum((y - b) ** 2))


def qlike(y: np.ndarray, f: np.ndarray, floor: float) -> np.ndarray:
    yy = np.maximum(np.asarray(y, float), np.finfo(float).tiny)
    ff = np.maximum(np.asarray(f, float), floor)
    z = yy / ff
    return z - np.log(z) - 1.0


def hac(diff: np.ndarray, lags: int) -> tuple[float, float, float]:
    fit = sm.OLS(diff, np.ones(len(diff))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}, use_t=False
    )
    return float(diff.mean()), float(fit.tvalues[0]), float(fit.pvalues[0])


VER.mkdir(parents=True, exist_ok=True)

# 1. Frozen inputs and core forecast rows.
manifest = json.loads((RES / "run_manifest.json").read_text())
hard_manifest = json.loads((RES / "hardening_manifest_v21.json").read_text())
record("hardening version is 2.1", hard_manifest.get("version") == "2.1", hard_manifest.get("version"))
record("market data hash", sha256(REP / "data" / "market_data.csv") == manifest["market_data_sha256"], manifest["market_data_sha256"])
record("rates hash", sha256(REP / "data" / "rates.csv") == manifest["rates_sha256"], manifest["rates_sha256"])
record("all core data checks pass", pd.read_csv(RES / "data_checks.csv")["pass"].all())

reported = pd.read_csv(RES / "variance_model_results.csv")
row_counts = {5: 3198, 22: 3164, 66: 3076}
forecast_frames: dict[int, pd.DataFrame] = {}
for h in HORIZONS:
    d = pd.read_csv(
        RES / f"variance_forecasts_{h}d.csv",
        parse_dates=["Date", "target_end", "training_end"],
    ).set_index("Date")
    forecast_frames[h] = d
    record(f"{h}d row count", len(d) == row_counts[h], len(d))
    record(f"{h}d target follows origin", bool((d.target_end > d.index).all()))
    record(f"{h}d training resolves by origin", bool((d.training_end <= d.index).all()))
    record(f"{h}d minimum training sample", int(d.n_train.min()) >= 750, int(d.n_train.min()))
    cols = ["realized_variance"] + [f"forecast_{m}" for m in MODELS]
    record(f"{h}d finite row-level values", np.isfinite(d[cols].to_numpy(float)).all())
    y = d.realized_variance.to_numpy(float)
    b = d.forecast_mean.to_numpy(float)
    for model in MODELS:
        calc = 0.0 if model == "mean" else oos_r2(y, d[f"forecast_{model}"].to_numpy(float), b)
        target = float(reported[(reported.horizon == h) & (reported.model == model)].iloc[0].oos_r2)
        record(f"{h}d OOS R2 {model}", abs(calc - target) < 5e-13, f"{calc:.15g} vs {target:.15g}")

# 2. Selected core inference and encompassing calculations.
d66 = forecast_frames[66]
y66 = d66.realized_variance.to_numpy(float)
diff_mse = (y66 - d66.forecast_q_raw.to_numpy(float)) ** 2 - (y66 - d66.forecast_q_mz.to_numpy(float)) ** 2
mean66, t66, p66 = hac(diff_mse, 71)
tests = pd.read_csv(RES / "variance_forecast_tests.csv")
r = tests[(tests.horizon == 66) & (tests.model_a == "q_mz") & (tests.model_b == "q_raw")].iloc[0]
record("66d MZ-vs-raw mean loss gain", abs(mean66 - float(r.mean_loss_gain_a_over_b)) < 5e-14)
record("66d MZ-vs-raw HAC t", abs(t66 - float(r.hac_t)) < 5e-12, f"{t66} vs {r.hac_t}")
record("66d MZ-vs-raw HAC p", abs(p66 - float(r.hac_p)) < 5e-12, f"{p66} vs {r.hac_p}")

enc = pd.read_csv(RES / "forecast_encompassing.csv")
fit = sm.OLS(y66, sm.add_constant(d66[["forecast_har4", "forecast_q_mz"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 71}, use_t=False
)
er = enc[(enc.horizon == 66) & (enc.option_model == "q_mz")].iloc[0]
record("66d encompassing option coefficient", abs(float(fit.params.forecast_q_mz) - float(er.beta_option)) < 5e-12)
record("66d encompassing option p", abs(float(fit.pvalues.forecast_q_mz) - float(er.p_option)) < 5e-12)

# 3. V2.1 alternative-loss calculations.
alt = pd.read_csv(RES / "alternative_loss_tests_v21.csv")
mean_y = float(y66.mean())
comparisons = [
    ("q_mz", "q_raw", "MZ option vs raw option"),
    ("q_mz", "har4", "MZ option vs HAR(4)"),
    ("harqts", "har4", "HAR + option + term vs HAR(4)"),
]
for a, b, label in comparisons:
    fa = d66[f"forecast_{a}"].to_numpy(float)
    fb = d66[f"forecast_{b}"].to_numpy(float)
    losses = {
        "MSE": (y66 - fb) ** 2 - (y66 - fa) ** 2,
        "MAE": np.abs(y66 - fb) - np.abs(y66 - fa),
        "QLIKE": qlike(y66, fb, mean_y * 0.01) - qlike(y66, fa, mean_y * 0.01),
    }
    for loss, diff in losses.items():
        rr = alt[(alt.horizon == 66) & (alt.model_a == a) & (alt.model_b == b) & (alt.loss == loss)]
        if loss == "QLIKE":
            rr = rr[np.isclose(rr.floor_fraction_of_mean_target, 0.01)]
        rr = rr.iloc[0]
        m, tt, pp = hac(diff, 71)
        record(f"66d {label} {loss} mean", abs(m - float(rr.mean_loss_gain_a_over_b)) < 5e-12)
        record(f"66d {label} {loss} HAC t", abs(tt - float(rr.hac_t)) < 5e-10)
        record(f"66d {label} {loss} HAC p", abs(pp - float(rr.hac_p)) < 5e-10)
record("MZ beats raw under 66d MSE", float(alt[(alt.horizon == 66) & (alt.model_a == "q_mz") & (alt.model_b == "q_raw") & (alt.loss == "MSE")].iloc[0].mean_loss_gain_a_over_b) > 0)
record("MZ beats raw under 66d MAE", float(alt[(alt.horizon == 66) & (alt.model_a == "q_mz") & (alt.model_b == "q_raw") & (alt.loss == "MAE")].iloc[0].mean_loss_gain_a_over_b) > 0)
qrow = alt[(alt.horizon == 66) & (alt.model_a == "q_mz") & (alt.model_b == "q_raw") & (alt.loss == "QLIKE") & np.isclose(alt.floor_fraction_of_mean_target, 0.01)].iloc[0]
record("no false QLIKE dominance claim", float(qrow.mean_loss_gain_a_over_b) < 0 and float(qrow.hac_p) > 0.9, qrow.to_dict())

# 4. V2.1 inference sensitivity, calibration, scale invariance, and calendar scaling.
sens = pd.read_csv(RES / "inference_sensitivity_v21.csv")
s66 = sens[(sens.horizon == 66) & (sens.model_a == "q_mz") & (sens.model_b == "q_raw")]
hac_ps = s66[s66.method == "HAC"].p_value.to_numpy(float)
block_ps = s66[s66.method == "circular_block_bootstrap"].p_value.to_numpy(float)
record("66d HAC sensitivity remains below 0.03", np.all(hac_ps < 0.03), hac_ps)
record("66d block sensitivity remains below 0.03", np.all(block_ps < 0.03), block_ps)

cal = pd.read_csv(RES / "calibration_diagnostics_v21.csv")
raw66 = cal[(cal.horizon == 66) & (cal.model == "q_raw")].iloc[0]
mz66 = cal[(cal.horizon == 66) & (cal.model == "q_mz")].iloc[0]
record("raw 66d forecast mean ratio", abs(float(raw66.mean_forecast_to_realized_ratio) - 1.4314788225) < 5e-9)
record("MZ reduces squared-bias share", float(mz66.squared_bias_share_of_mse) < float(raw66.squared_bias_share_of_mse))
record("MZ does not increase correlation", float(mz66.correlation) < float(raw66.correlation))

scale = pd.read_csv(RES / "idealized_scale_invariance_v21.csv")
record("idealized leverage scale invariance", float(scale.absolute_difference.max()) < 1e-12, scale.absolute_difference.max())
calendar = pd.read_csv(RES / "calendar_scaling_results_v21.csv")
for model, target in [("q_raw", -0.2054109), ("q_mz", 0.0632852)]:
    value = float(calendar[(calendar.horizon == 66) & (calendar.model == model)].iloc[0].calendar_day_scaling_oos_r2)
    record(f"66d calendar scaling {model}", abs(value - target) < 5e-6, value)

# 5. Fund, regime, overlap, and leakage consistency.
reg = pd.read_csv(RES / "regime_results_66d.csv")
record("full regime matches baseline MZ", abs(float(reg[reg.subset == "full_sample"].iloc[0].r2_q_mz) - float(reported[(reported.horizon == 66) & (reported.model == "q_mz")].iloc[0].oos_r2)) < 1e-13)
record("event-exclusion MZ remains positive", float(reg[reg.subset == "exclude_windows_intersecting_2018_2020_2022"].iloc[0].r2_q_mz) > 0)
fund = pd.read_csv(RES / "fund_model_results.csv")
for f in ("SSO", "UPRO", "SDS", "SPXU"):
    d = pd.read_csv(RES / f"fund_forecasts_{f}_66d.csv")
    y = d.realized_drag.to_numpy(float)
    b = d.historical_mean.to_numpy(float)
    for model in ("q_raw", "q_mz", "har4", "harqts"):
        calc = oos_r2(y, d[f"forecast_{model}"].to_numpy(float), b)
        target = float(fund[(fund.fund == f) & (fund.horizon == 66) & (fund.model == model) & (fund.mapping == "intercept")].iloc[0].oos_r2)
        record(f"fund 66d R2 {f} {model}", abs(calc - target) < 5e-13)
non = pd.read_csv(RES / "nonoverlap_results.csv")
record("all nonoverlap offsets present", all(len(non[non.horizon == h]) == h for h in HORIZONS), non.groupby("horizon").size().to_dict())
leak = pd.read_csv(RES / "future_perturbation_test.csv")
record("future-data perturbation passes", leak["pass"].all() and (leak.max_abs_difference == 0).all(), leak.to_dict("records"))

# 6. Source-language, numerical-claim, anonymity, and availability checks.
anon = (PAPER / "manuscript_anonymous.md").read_text(encoding="utf-8")
named = (PAPER / "manuscript_named.md").read_text(encoding="utf-8")
supp = (PAPER / "supplement.md").read_text(encoding="utf-8")
combined = anon + "\n" + supp
for bad in ["only the probability measure changes", "cleanly identifies", "HAR-RV", "[REPOSITORY", "TODO", "Revstov", "independent verification script"]:
    record(f"obsolete phrase absent: {bad}", bad.lower() not in combined.lower())
for identity in ["Kartikay", "Goyle", "Yevgen", "Revtsov", "goylekar", "yrevtsov"]:
    record(f"anonymous source lacks {identity}", identity.lower() not in anon.lower())
record("named source contains both authors", "Kartikay Goyle" in named and "Yevgen Revtsov" in named)
required_phrases = [
    "not under QLIKE",
    "no loss-uniform dominance",
    "scaling alone cannot change R2",
    "SPY-SPX basis",
    "ex post diagnostics",
    "not a substitute for intraday realized variance",
    "not significantly outperform the HAR(4)",
    "VIX9D",
    "outside the implemented analysis",
]
for phrase in required_phrases:
    record(f"required qualification present: {phrase}", phrase.lower() in combined.lower())
for token in ["0.352", "0.184", "-0.204", "0.063", "0.020", "0.021", "0.181", "0.925", "1.431", "0.883", "4.44e-16", "11.48"]:
    record(f"headline token in source: {token}", token in combined)
record("no Unicode em dash", "\u2014" not in combined)
prose = re.sub(r"^\s*\|.*\|\s*$", "", combined, flags=re.M)
prose = re.sub(r"^---\s*$", "", prose, flags=re.M)
record("no prose triple-hyphen", "---" not in prose)
record("data availability matches public archive", "An anonymized replication archive" in anon and "omits raw third-party" in anon)

# 7. Table fragments are populated and LaTeX row terminators are valid.
for table in sorted((REP / "tables").glob("*.tex")):
    record(f"table fragment nonempty {table.name}", table.stat().st_size > 50, table.stat().st_size)
    if "tabular" in table.read_text():
        for line_no, line in enumerate(table.read_text().splitlines(), 1):
            stripped = line.rstrip()
            if " & " in line and not stripped.startswith(("%", r"\begin", r"\toprule", r"\midrule", r"\bottomrule")):
                record(f"LaTeX row terminator {table.name}:{line_no}", stripped.endswith(r"\\"), stripped)
record("Table 2 reports delta R2", r"\Delta R^2_{OOS}" in (REP / "tables" / "table2_pairwise_tests.tex").read_text())

# 8. Artifact and PDF anonymity checks (run after document build).
expected = ["manuscript_anonymous", "manuscript_named", "supplement", "title_page", "cover_letter", "response_to_reviewers"]
for stem in expected:
    for ext in ("pdf", "docx"):
        p = PAPER / f"{stem}.{ext}"
        record(f"artifact exists {p.name}", p.exists() and p.stat().st_size > 1000, p.stat().st_size if p.exists() else "missing")
subprocess.run(["pdftotext", str(PAPER / "manuscript_anonymous.pdf"), str(VER / "anonymous_pdf_text.txt")], check=True)
pdftext = (VER / "anonymous_pdf_text.txt").read_text(errors="ignore")
for identity in ["Kartikay", "Goyle", "Yevgen", "Revtsov", "goylekar", "yrevtsov"]:
    record(f"anonymous PDF lacks {identity}", identity.lower() not in pdftext.lower())
record("anonymous PDF contains QLIKE qualification", "qlike" in pdftext.lower() and "loss-uniform" in pdftext.lower())

# 9. Clean-room regeneration of core and V2.1 outputs.
clean = VER / "clean_run_v21"
log = VER / "clean_run_v21.log"
clean_complete = (clean / "results" / "hardening_manifest_v21.json").exists() and (clean / "results" / "variance_model_results.csv").exists()
if not clean_complete:
    if clean.exists():
        shutil.rmtree(clean)
    (clean / "code").mkdir(parents=True)
    (clean / "data").mkdir(parents=True)
    shutil.copy2(REP / "code" / "run_analysis_v2.py", clean / "code" / "run_analysis_v2.py")
    shutil.copy2(REP / "code" / "run_hardening_v21.py", clean / "code" / "run_hardening_v21.py")
    shutil.copy2(REP / "data" / "market_data.csv", clean / "data" / "market_data.csv")
    shutil.copy2(REP / "data" / "rates.csv", clean / "data" / "rates.csv")
    with log.open("w") as fh:
        subprocess.run([
            "python", str(clean / "code" / "run_analysis_v2.py"),
            "--data", str(clean / "data" / "market_data.csv"),
            "--rates", str(clean / "data" / "rates.csv"),
            "--output", str(clean), "--skip-leakage",
        ], check=True, stdout=fh, stderr=subprocess.STDOUT)
        subprocess.run([
            "python", str(clean / "code" / "run_hardening_v21.py"), "--root", str(clean)
        ], check=True, stdout=fh, stderr=subprocess.STDOUT)
else:
    record("existing clean-room run detected", True, str(clean))

skip = {"future_perturbation_test.csv"}
for p in sorted(RES.glob("*.csv")):
    if p.name in skip:
        continue
    q = clean / "results" / p.name
    record(f"clean run produced {p.name}", q.exists())
    a = pd.read_csv(p)
    b = pd.read_csv(q)
    record(f"clean shape {p.name}", a.shape == b.shape and list(a.columns) == list(b.columns), f"{a.shape} vs {b.shape}")
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            record(f"clean numeric {p.name}:{col}", np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=5e-12, equal_nan=True))
        else:
            record(f"clean text {p.name}:{col}", a[col].fillna("").astype(str).equals(b[col].fillna("").astype(str)))
for name in [
    "Figure_1_Variance_OOS_R2.png",
    "Figure_2_Incremental_Option_R2.png",
    "Figure_3_Rolling_Incremental_Loss.png",
    "Figure_4_Fund_Heterogeneity.png",
]:
    record(f"clean figure {name}", sha256(REP / "figures" / name) == sha256(clean / "figures" / name))

# 10. Write machine-readable audit summary.
out = pd.DataFrame(checks)
out.to_csv(VER / "verification_checks_v21.csv", index=False)
summary = {
    "status": "PASS",
    "version": "2.1",
    "checks_passed": int(out["pass"].sum()),
    "checks_total": int(len(out)),
    "market_data_sha256": manifest["market_data_sha256"],
    "rates_sha256": manifest["rates_sha256"],
    "max_scale_invariance_difference": float(scale.absolute_difference.max()),
    "clean_run_log": str(log.relative_to(ROOT)),
    "limitations_not_removed": [
        "no intraday realized variance",
        "no sponsor NAV history",
        "SPY realized-variance proxy versus SPX-option indices",
        "no VIX9D implementation",
        "no Nasdaq-100/VXN or other-market replication",
    ],
}
(VER / "verification_summary_v21.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
