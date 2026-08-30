# Identification strategy

Thermo Credit uses a layered design because public credit statistics do not
usually observe borrower, purpose, recipient, collateral, and final expenditure
in one record. The current release does not fill those gaps by definition.

## Evidence layers

| Layer | What enters this layer | Current use |
| --- | --- | --- |
| Directly observed | Published loan stocks, sector labels, GDP, yields, spreads, balance-sheet totals | Source data and accounting checks |
| Mapped measurement | Literature-fixed borrower taxonomies, positive and signed stock changes, release-lag alignment | Primary JP borrower-composition bridge |
| Proxy | EU/US allocation shares, market liquidity inputs, stress and asset-price substitutes | Portability and sensitivity checks |
| Model diagnostic | `S_M`, `T_L`, `p_C`, `U`, `F_C`, `X_C`, `loop_area` | Exploratory monitoring |
| Latent target | Loan-use share `q_t^use`, fragility state `S_t` | Future work only |

## Primary JP estimand

The current estimand is the four-quarter composition of positive bucket-level
changes in one reconciled BOJ loan-stock population. The primary scalar `q_t`
is the non-financial-business coordinate. This design identifies a borrower
composition statistic under a published crosswalk. It does not identify loan
purpose or the destination of expenditure.

The following checks are required before an empirical row is interpreted:

1. Scale and composition must use the same included population.
2. Signed changes must reconcile to the included stock change.
3. Positive parts must be taken after aggregation within each fixed bucket.
4. Construction placement must be shown under all registered mappings.
5. Release lags and the 2009 classification break must be applied before
   outcome evaluation.

## What would identify credit use

A stronger Real/Asset claim needs gross originations jointly classified by
borrower and use. It must separately retain repayments, refinancing, write-offs,
and reclassifications. Asset-use categories must distinguish new construction
from purchases of existing property. A purpose-coded sample with a different
population can provide partial triangulation, but it cannot validate the full
BOJ bridge by itself.

If direct purpose data remain unavailable, a latent model may be attempted only
after its measurement equation is fixed from training data and tested on a new
vintage. Asset prices or GDP outcomes alone cannot identify the latent share;
using those outcomes both to construct and validate it would be circular.

## Outcome design

The current pseudo-OOS table is a bounded use example. At each forecast origin,
preprocessing is fit on the expanding training window. The candidate adds one
registered composition coordinate to matched-population credit growth. It is
compared with the same-scale baseline on the same complete-case observations.
Current-vintage inputs and fixed release lags do not recreate historical data
vintages, so this exercise is not prospective evidence.

The current main-table candidates have worse point loss than the matched-stock
baseline. That result is retained. A claim of incremental predictability would
require an untouched prospective sequence or a genuinely new vintage under the
frozen protocol.

## Change trigger

The borrower-composition claim may be upgraded to a loan-use claim only when a
joint borrower-and-purpose source covers the same lender, instrument, timing,
and population closely enough to support that mapping. Until then, there is not
enough evidence to require or support the stronger interpretation.
