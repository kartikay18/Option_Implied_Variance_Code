# Public-release data policy

This repository publishes the analysis code, documentation, aggregate statistical results, tables, figures, input hashes, and verification summaries for the associated working paper. It does not publish the frozen third-party price, volatility-index, or interest-rate observations.

## Deliberately excluded files

The public Git history excludes:

- `data/market_data.csv` and `data/rates.csv`;
- row-level `results/variance_forecasts*.csv` files, which include VIX/VIX3M observations and reversibly encoded option signals;
- row-level `results/fund_forecasts_*.csv` files and related recursive or rolling series; and
- clean-room copies of licensed inputs and private manuscript-build artifacts.

These paths are protected by `.gitignore` so that running the analysis locally does not make them eligible for an accidental broad `git add`.

## Included outputs

The committed `results/` files are aggregate model evaluations, robustness statistics, diagnostics, and manifests. The committed `tables/` and `figures/` files are presentation outputs derived from those analyses. They do not contain the frozen source rows.

## Reproduction

Obtain the inputs from providers whose terms permit your use, construct the schemas in `DATA_ACQUISITION.md`, and compare locally held files with `data/INPUT_HASHES.txt` when you have access to the exact frozen inputs. Running the analysis will regenerate both public aggregate outputs and ignored row-level working files.

This policy documents the technical contents of this repository; it is not a legal opinion about any particular data-provider agreement.
