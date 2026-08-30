# Thermo Credit definitions

This document separates variables that are observed in the current release from
quantities that remain proxies, model diagnostics, or future latent states. The
distinction is part of the claim, not a documentation detail.

## Current measurement layer

| Symbol | Meaning | Evidence class | Operational definition | Claim limit |
| --- | --- | --- | --- | --- |
| `L_t^P` | Included JP loan stock | Directly observed and reconciled | BOJ total loans of domestically licensed banks less local-government loans | Describes the lender population, not domestic borrower geography |
| `Delta L_t^k` | Net stock change for bucket `k` | Derived from observed stocks | `L_t^k - L_(t-1)^k` after aggregation within a fixed bucket | Is not gross lending; repayments, write-offs, and reclassification remain mixed in |
| `P_t^k` | Positive bucket-level stock change | Derived | `max(Delta L_t^k, 0)` | Is a measurement convention, not an observed origination flow |
| `pi_(t,4)^k` | Four-quarter borrower-composition coordinate | Derived measurement bridge | Sum of `P_t^k` over four quarters divided by the common four-bucket sum | Identifies borrower composition only |
| `q_t` | Primary JP scalar coordinate | Derived measurement bridge | `pi_(t,4)^NFB` under the Bezemer-Samarina-Zhang Japan crosswalk | Does not identify GDP-linked use, loan purpose, or final expenditure |
| `g_(t,h)^P` | Matched credit-scale coordinate | Derived | `Delta_h log(L_t^P)` | Measures stock growth over the same included population |

The four primary JP buckets are non-financial business (`NFB`), financial
business (`FIN`), property/mortgage (`PROP`), and household non-housing
(`HHN`). Construction is in `NFB` in the primary crosswalk. The Werner-inspired
and Muller-Verner mappings move it under published alternative classifications
while keeping the included population fixed.

EU and US allocation shares use coarser regional proxies. They are provided as
schema-portability panels and negative controls. They are not cross-country
validation of the Japanese borrower-composition measure.

## Proposed structural layer

These variables state the model we would like to estimate. The current data do
not identify them directly.

| Symbol | Meaning | Current status | Required evidence |
| --- | --- | --- | --- |
| `C_t` | Gross new credit flow | Partly proxied by stock changes | Originations and other flow adjustments on a common population |
| `C_t^R` | Credit financing GDP-linked transactions | Not identified | Purpose-coded originations linked to new production or expenditure |
| `C_t^A` | Credit financing purchases of existing assets | Not identified | Purpose-coded originations linked to existing property or financial assets |
| `q_t^use` | `C_t^R / (C_t^R + C_t^A)` | Latent target | Joint borrower-and-use data or a separately validated latent-state model |
| `Y_t^N` | Nominal output | Observed proxy available | Nominal GDP or value added aligned in units and release time |
| `Y_t^R` | Real activity | Partial | Real GDP or production target with vintage-aware timing |
| `P_t` | General prices | Partial | CPI or GDP deflator with a fixed release convention |
| `A_t` | Asset prices | Partial proxy | Housing, equity, or commercial-property series matched to the credit concept |
| `S_t` | Fragility state | Not identified | A validated observed composite or a separately tested state-space estimate |

When purpose data become available, a possible structural partition is

```text
C_t = C_t^R + C_t^A
q_t^use = C_t^R / C_t
```

Construction should not be forced into either side without evidence. A split
weight may be estimated only from training data or fixed from an external
source before outcome evaluation.

## Experimental dashboard diagnostics

| Symbol | Meaning | Status | Interpretation boundary |
| --- | --- | --- | --- |
| `S_M` | Allocation-dispersion statistic | Implemented diagnostic | Depends on the configured allocation buckets; it is not thermodynamic entropy |
| `T_L` | Liquidity-state index | Implemented diagnostic | A monotone market proxy, not physical temperature |
| `p_C` | Credit-pressure index | Implemented diagnostic | A model transformation, not an observed market price |
| `U` | Internal-energy-like gauge | Implemented diagnostic | Bookkeeping analogy only |
| `F_C` | Free-energy-like gauge | Implemented diagnostic | Model transformation with reference parameters |
| `X_C` | Exergy-like diagnostic | Implemented, not validated | Not a safety margin, policy threshold, or established forecast |
| `loop_area` | Streaming open-path area | Implemented diagnostic | Measures path geometry; a closed-cycle claim needs a registered cycle window |

## Falsification rules

- If borrower composition does not improve a matched-scale out-of-sample
  baseline, it must not be presented as a forecasting result.
- If a calibrated `X_C` does not beat raw `X_C` and simple trailing-change
  baselines on untouched data, calibration has not validated the diagnostic.
- If `C_t^R` and `C_t^A` cannot be distinguished with purpose-coded data, the
  structural two-flow model remains unidentified.
- If `T_L`, `S_M`, or `loop_area` depend mainly on arbitrary preprocessing or
  bucket choices, they should remain audit indicators or be removed.

See `docs/identification_strategy.md` and `docs/calibration_protocol.md` for the
rules used to preserve these boundaries.
