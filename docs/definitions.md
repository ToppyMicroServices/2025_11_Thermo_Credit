# Thermo Credit v2 Definitions

This file fixes the operational meaning of the core v2 variables before a full
estimation pipeline is added.

## Core Variables

| Symbol | Name | Type | Units | Operational definition | Current repo status |
| --- | --- | --- | --- | --- | --- |
| `C_t` | Total new credit flow | observed / derived | local currency per period | New lending flow when available; otherwise first difference of nominal private credit stock | partial proxy available |
| `q_t` | GDP-linked credit share | proxy / latent | share in `[0, 1]` | Share of `C_t` flowing to GDP-linked transactions rather than existing-asset transactions | not yet implemented |
| `C_t^R` | GDP-linked credit flow | derived | local currency per period | `q_t C_t` | not yet implemented |
| `C_t^A` | Existing-asset credit flow | derived | local currency per period | `(1 - q_t) C_t` | not yet implemented |
| `C_t^G` | GDP-transaction credit | observed / latent | local currency per period | Credit tied to current production, income, and business spending | not yet implemented |
| `C_t^B` | Construction credit | observed / latent | local currency per period | Credit for new building and development; intermediate between GDP-linked and asset-linked uses | not yet implemented |
| `C_t^E` | Existing-asset credit | observed / latent | local currency per period | Credit used to acquire existing land, housing, CRE, or financial assets | not yet implemented |
| `lambda_B` | Construction weight | calibrated / design | share in `[0, 1]` | Share of `C_t^B` treated as GDP-linked when forming `C_t^R` and `C_t^A` | not yet implemented |
| `Y_t^N` | Nominal output proxy | observed / proxy | local currency | Nominal GDP or value-added proxy aligned to credit frequency | partial proxy available |
| `Y_t^R` | Real activity proxy | observed / proxy | index or real currency | Real GDP, industrial production, or equivalent real-activity target | not yet implemented |
| `P_t` | Price level / inflation | observed | index or percent | CPI, GDP deflator, or equivalent broad price measure | not yet implemented |
| `A_t` | Asset-price proxy | observed / proxy | index or currency | Housing, equity, or CRE price proxy; use balance-sheet stock when available | not yet implemented |
| `S_t` | Stress / fragility state | proxy / latent | index or z-score | Summary of funding, spread, volatility, and macro-financial stress | partial proxy available |

## Derived Thermo v2 Metrics

| Symbol | Name | Type | Units | Operational definition | Current repo status |
| --- | --- | --- | --- | --- | --- |
| `eta_t` | Credit efficiency | derived | ratio | `Delta Y_t^N / C_t` or standardized equivalent | not yet implemented |
| `d_t` | Asset bias | derived | ratio or score | `Delta A_t / C_t`; if only an index exists, use normalized `Delta log(A_t)` | not yet implemented |
| `sigma_t` | Dissipation proxy | derived | ratio or score | Baseline proxy `max(0, 1 - eta_t)`; may later absorb rollover and turnover measures | not yet implemented |

## Existing Repo Thermo Diagnostics

| Symbol | Name | Type | Units | Operational definition | Current repo status |
| --- | --- | --- | --- | --- | --- |
| `S_M` | Monetary dispersion entropy | derived | scaled entropy | Entropy-like measure built from `data/allocation_q*.csv` | implemented |
| `T_L` | Liquidity temperature | derived | index | Liquidity-intensity proxy from depth, turnover, and spreads | implemented |
| `p_C` | Credit pressure | derived | index | Credit-capacity pressure gauge | implemented |
| `U` | Internal-energy-like gauge | derived | currency-like gauge | Bookkeeping potential built from output / credit proxies | implemented |
| `F_C` | Free-energy-like gauge | derived | currency-like gauge | `U - T_0 S_M` style monitor | implemented |
| `X_C` | Exergy-like ceiling | derived | currency-like gauge | Usable credit-capacity proxy | implemented |
| `loop_area` | Loop dissipation | derived | area / score | Hysteresis-like loop area in state space | implemented |

## Identification Notes

- `q_t` is the central missing variable. It should not be conflated with the
  entropy allocation buckets already used for `S_M`.
- In the Werner-style reading used here, `C_t^R` is credit tied to GDP-linked
  transactions, while `C_t^A` is credit mainly used to purchase existing
  assets. Construction credit may need a partial weight instead of a hard
  bucket.
- A practical implementation can therefore start from the three-way split
  `C_t = C_t^G + C_t^B + C_t^E` and only then collapse it into `C_t^R` and
  `C_t^A` with `lambda_B`.
- `S_t` can start as a weighted stress proxy built from existing spread and
  volatility series, then migrate to a latent state-space estimate later.
- `eta_t`, `d_t`, and `sigma_t` are intentionally simple at first. Their role is
  to create a measurable bridge from theory to forecasting.

## Falsifiability Notes

- If `q_t` does not improve forecast performance, the partition is too weak.
- If `C_t^R` and `C_t^A` behave similarly in data, the model should collapse
  back toward a simpler credit-volume design.
- If `S_t` adds no downside signal beyond current repo indicators, the thermo
  extension should be simplified.
