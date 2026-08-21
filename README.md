# Replication package: option-implied and physical variance forecasts

This repository reproduces the analysis for:

**When Does Option-Implied Variance Add to Physical Forecasts? Evidence from S&P 500 Leveraged ETF Drag**

## Quick start

1. Obtain the inputs from providers whose terms permit your use, then place the files at:
   - `data/market_data.csv`
   - `data/rates.csv`
2. Install the Python dependencies in `requirements.txt`.
3. Run:

```bash
python code/run_analysis_v2.py --output .
python code/run_hardening_v21.py --root .
```

The first program creates the core row-level forecasts, result tables, figures, robustness checks, and run manifest. The V2.1 hardening program adds alternative-loss, inference-bandwidth, calibration, scale-invariance, and calendar-scaling checks. Generated row-level outputs remain local because `.gitignore` excludes them from the public repository.

## Core design

- Sample: January 4, 2010 through December 30, 2025
- Underlying: SPY
- Funds: SSO, UPRO, SDS, SPXU
- Option signals: VIX at 5 and 22 trading days; VIX3M at 66 trading days
- Horizons: 5, 22, 66 trading days
- Minimum resolved estimation sample: 750 targets
- Forecasts: historical mean, raw option signal, recursive Mincer-Zarnowitz calibration, HAR(3), HAR(4), HAR plus option, HAR plus option and term structure
- Inference: Newey-West HAC, circular block bootstrap, Clark-West, and encompassing regressions

## Directory structure

- `code/`: analysis program
- `tests/`: full-package verification programs retained for transparency
- `data/`: input files in the private package; hashes only in the public repository
- `results/`: aggregate public outputs; ignored row-level forecasts are generated locally
- `figures/`: publication-quality figures
- `tables/`: LaTeX table fragments
- `logs/`: run logs

## Data licensing

The input series are third-party adjusted market prices, Cboe volatility-index values, and Treasury-bill rates. The public repository does not redistribute the frozen raw series or row-level outputs that contain or reversibly encode those series. It includes acquisition instructions, the expected schema, SHA-256 hashes, aggregate results, and the analysis code. A separate private verification archive retained by the authors contains the exact frozen inputs and complete row-level outputs and should not be posted publicly unless redistribution rights are confirmed. See `DATA_POLICY.md` for the precise public/private boundary.

The precomputed audit summary under `verification_v21/` records the completed full-package verification. The scripts in `tests/` require the licensed inputs and the authors' complete manuscript directory tree; they are not a zero-data test suite for the public checkout.

## Scope and limitations

The analysis is scoped to the S&P 500/VIX complex. It uses SPY as the realized-variance proxy while VIX and VIX3M are SPX-option indices, adjusted market prices rather than sponsor NAVs, and daily return-based variance measures rather than intraday realized variance. VIX9D, Nasdaq-100/VXN, sponsor NAV, and intraday extensions are not implemented. The manuscript states these limitations explicitly.


## Working-paper record

- SSRN abstract ID: 7119799
- DOI: 10.2139/ssrn.7119799
- Working-paper title: *When Does Option-Implied Variance Add to Physical Forecasts? Evidence from S&P 500 Leveraged ETF Drag*
- Version: substantially revised August 2026
