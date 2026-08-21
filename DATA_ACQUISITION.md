# Data acquisition and frozen-input schema

## Market data

Create `data/market_data.csv` with one row per trading date and the columns:

```text
Date,SPY,SSO,UPRO,SDS,SPXU,VIX,VIX3M
```

- `Date`: ISO date
- `SPY`, `SSO`, `UPRO`, `SDS`, `SPXU`: adjusted daily market prices
- `VIX`: Cboe VIX close
- `VIX3M`: Cboe three-month volatility-index close

The frozen file used in the paper contains 4,023 complete rows from 2010-01-04 through 2025-12-30.

## Rates

Create `data/rates.csv` with:

```text
observation_date,DTB3
```

`DTB3` is the annualized three-month Treasury-bill rate in percent. Missing nontrading-date observations are forward-filled after alignment to the market calendar.

## Frozen hashes

See `data/INPUT_HASHES.txt`. The analysis checks the sample range, completeness, positivity, date uniqueness, selected shock dates, and approximate daily fund betas before running.

## Measurement and redistribution notes

SPY is used as the investable S&P 500 proxy, while VIX and VIX3M are calculated from SPX options. The analysis therefore does not eliminate SPY-SPX basis and dividend differences. VIX9D is not included in the frozen input file, so the 5-day use of VIX is an explicitly approximate flat-forward extrapolation. Sponsor NAV histories, intraday returns, VXN, and Nasdaq-100 leveraged funds are not part of the implemented analysis.

The public replication repository omits the raw third-party market, volatility-index, and rate series. It also omits row-level outputs that contain or reversibly encode those inputs. Users should obtain data from their authorized providers, construct files matching the schemas above, and compare their files with `data/INPUT_HASHES.txt` where the exact frozen inputs are available to them. See `DATA_POLICY.md` for the complete public-release boundary.
