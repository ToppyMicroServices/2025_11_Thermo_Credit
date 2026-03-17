# Thermo Credit v2 Specification

Status: working draft

This document narrows Thermo Credit from a broad conceptual framework into a
measurement-first macro-credit model. It complements
`tex/theory.tex` rather than replacing it.

## 1. Problem Statement

Thermo Credit v2 is designed to explain how the destination of new credit
changes the transformation of credit expansion into:

- output and inflation pass-through,
- asset-price acceleration,
- fragility and downside sensitivity.

The primary research question is:

How does the partition of new credit between real activity and asset markets
change growth, prices, asset prices, and instability over time?

The model should stay focused on three use cases:

1. early-warning signals for asset inflation and fragility,
2. description of the real-versus-asset credit mix,
3. forecast improvement relative to credit-volume-only models.

## 2. Minimal State Vector

The baseline state vector is intentionally small.

- `C_t`: new credit flow over period `t`
- `q_t in [0, 1]`: share of new credit flowing to real activity
- `1 - q_t`: share of new credit flowing to asset markets
- `Y_t^N`: nominal output proxy used for accounting ratios
- `Y_t^R`: real activity proxy used for forecast targets
- `P_t`: general price level or inflation proxy
- `A_t`: asset-price index or asset-balance-sheet proxy
- `S_t`: system stress / fragility state

Credit partition:

```text
C_t^R = q_t C_t
C_t^A = (1 - q_t) C_t
```

Implementation note:

- `Y_t^R` should remain the main real-economy outcome.
- `Y_t^N` is tracked separately because efficiency ratios such as
  `Delta Y_t / C_t` need unit consistency when credit is measured in currency.
- If only an index is available for `A_t`, the asset-bias metric becomes a
  normalized proxy rather than a literal stock-flow ratio.

## 3. Thermodynamic Interpretation

The thermodynamic layer should be operational, not metaphorical.

Baseline mapping:

- energy inflow -> `C_t`
- useful work -> `Delta Y_t^N`
- storage / balance-sheet accumulation -> `Delta A_t`
- dissipation -> rollover, speculative turnover, price distortion, weak pass-through
- entropy / irreversibility -> declining allocation quality and path dependence

Initial measurable proxies:

```text
eta_t = Delta Y_t^N / C_t
d_t   = Delta A_t / C_t
sigma_t = max(0, 1 - eta_t)
```

Fallback when only an asset-price index is available:

```text
d_t^proxy = Delta log(A_t) / (C_t / Y_t^N)
```

These are provisional quantities. They are useful because they can be
estimated now and refined later.

## 4. State Dynamics And Irreversibility

Thermo Credit should differ from static credit accounting by making
irreversibility explicit.

Baseline reduced-form dynamics:

```text
eta_t = eta_0 - alpha (1 - q_t) - beta S_{t-1} + eps_eta,t

S_t = rho S_{t-1} + gamma (C_t^A / Y_t^N) - delta (C_t^R / Y_t^N) + eps_S,t
```

Interpretation:

- more asset-directed credit raises the future stress state,
- accumulated stress lowers next-period conversion efficiency,
- the same amount of new credit becomes less productive in a stressed regime.

This is the minimum irreversibility block required to justify the "Thermo"
label in an empirical model.

## 5. Testable Hypotheses

Only a small number of falsifiable hypotheses should be carried at once.

- `H1`: lower `q_t` should predict asset-price acceleration before it predicts
  stronger real growth.
- `H2`: high `C_t` with low `q_t` should show weaker pass-through to inflation
  and real activity.
- `H3`: high `C_t^A` combined with high `S_t` should predict higher future
  volatility, spread widening, or credit contraction risk.

## 6. Identification Strategy

Because the real-versus-asset split is not always observed directly, the model
should be estimated in three layers.

### Level 1: Directly Observed

- sectoral lending flows,
- household housing credit,
- lending to non-financial firms,
- commercial real-estate credit,
- central-bank balance-sheet aggregates,
- official stress and spread data.

### Level 2: Proxy Inferred

- housing-price and equity-price responses,
- equipment investment, inventories, employment,
- funding spreads, volatility measures, FX stress,
- turnover or rollover proxies from existing repo data.

### Level 3: Latent State

- `q_t` as a latent partition share when direct loan-purpose data are incomplete,
- `S_t` as a latent fragility state estimated with a state-space model.

Recommended order:

1. start with Level 1 + Level 2,
2. validate forecasting usefulness,
3. move to Level 3 only after the reduced-form version is stable.

## 7. Relation To Current Repo Outputs

The current repo already computes a useful thermo-monitoring layer:

- `S_M`
- `T_L`
- `p_C`
- `loop_area`
- `U`
- `F_C`
- `X_C`

These should be retained, but repositioned as:

- current thermo diagnostics,
- candidate inputs into the future `S_t` block,
- companion indicators rather than substitutes for `C_t^R` and `C_t^A`.

Important distinction:

`data/allocation_q*.csv` currently stores entropy allocation buckets used for
`S_M`. Those shares are not yet the same object as the v2 real-versus-asset
partition `q_t`.

## 8. Empirical Design

The first empirical version should prioritize prediction over structural purity.

Suggested baseline regressions:

```text
Delta y_{t+h} = b0 + b1 (C_t^R / Y_t^N) + b2 (C_t^A / Y_t^N) + b3 S_t + u_t

pi_{t+h} = c0 + c1 (C_t^R / Y_t^N) + c2 (C_t^A / Y_t^N) + c3 S_t + v_t

Delta log(A_{t+h}) = d0 + d1 (C_t^R / Y_t^N) + d2 (C_t^A / Y_t^N) + d3 S_t + w_t
```

Benchmark models:

- credit-volume-only,
- existing thermo indicators only,
- partitioned credit plus stress.

Success criterion:

The partitioned model must improve directional accuracy, fit, or downside-hit
rates relative to simpler baselines.

## 9. Falsifiability

The model should say clearly how it can fail.

- If the estimated `q_t` does not improve asset-price or fragility forecasts,
  the partition is not pulling its weight.
- If separating `C_t^R` and `C_t^A` does not improve forecast performance over
  total credit alone, the two-bucket split is not empirically justified.
- If adding `S_t` does not improve downside prediction, the thermo extension is
  too weak or too complicated for the available data.

## 10. Repository Deliverables

The model should be published as a recalculable package, not only as a report.

Minimum deliverables:

1. `report.html` for interpretation and narrative,
2. `docs/definitions.md` for variable definitions,
3. `data/data_dictionary.csv` for source and unit mapping,
4. `scripts/` and `tests/` for recalculation and validation.

Planned implementation sequence:

1. define variables and identification rules,
2. build `C_t`, `q_t`, `C_t^R`, `C_t^A`, `S_t`,
3. test `H1` to `H3`,
4. expose results on the site as a recalculable model.
