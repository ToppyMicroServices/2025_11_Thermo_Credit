# mAI Economy: a credit-based research program

Status: research plan, 8 September 2026. The closed-loop simulator described
here is not yet implemented. Existing measurement code remains unchanged.

This is one Toppy research theme, not a replacement for its other projects.

## Question

How do credit creation and its uses shape production, asset transactions, and
financial fragility? Can explicit guidance rules improve these outcomes once
banks and borrowers are allowed to respond to them?

mAI Economy studies this question with AI-assisted model development and
reproducible experiments. The intended system is an AI version of window
guidance: institutions set objectives, compare rules, observe responses, and
revise guidance. It does not assume a fixed pool of money to distribute, or
that a requested allocation becomes actual lending.

Thermo Credit supplies a candidate measurement and state representation within
this program. Its current borrower-composition paper is one empirical input,
not the complete theory of credit or a validation of guidance.

The credit-use distinction draws on Richard A. Werner's Quantity Theory of
Credit: separate credit for GDP transactions from credit for existing-asset
transactions [Werner, 2012](https://doi.org/10.1016/j.irfa.2012.06.002).
GDP-related credit is not synonymous with productive investment. The model
must retain the distinction between consumption and capacity-building uses.
Thermo-Credit's state representation and the AI controller are proposed
extensions, not results established in Werner's work.

## The model to build first

Start with a small stock-flow-consistent model: each flow must update the
corresponding stocks and leave each balance sheet balanced. Include two banks,
households, producing firms, and a settlement authority. A later variant adds
a non-bank funding channel to test substitution outside the guidance perimeter.
Do not collapse capital and settlement liquidity into one capacity score.

Track loan origination, payments, interest, principal repayment, and losses.
Distinguish newly produced goods and investment from transfers of existing
assets and refinancing. A borrower can use credit for more than one purpose;
neither an industry label nor mortgage status identifies the final use.
Unknown use stays unknown.

The first experiment uses one currency and a stated period length. Record all
nominal amounts in that currency. Keep price indices, real quantities, and
credit flows separate; an index change divided by a currency flow is not a
dimensionless efficiency or dissipation rate.

### Accounting before behavior

| Event | Required accounting in the minimal model |
| --- | --- |
| Loan credited to an account | The bank adds a loan asset and a deposit liability; the borrower adds a deposit asset and loan liability. |
| Payment within one bank | Deposits move between customers; aggregate deposits do not change. |
| Payment to another bank | Customer deposits and interbank settlement assets move between banks; aggregate deposits do not change. |
| Principal repayment from a deposit | The loan and deposit fall by the same amount, with settlement entries if banks differ. |
| Interest payment | Transfer income and update bank equity and payer deposits consistently; do not treat interest as principal repayment. |
| Loan write-off | Reduce the bank loan asset and equity, allowing for any prior provision. Deposits held elsewhere do not disappear. Borrower debt changes only under the stated debt-discharge rule. |

Initial assets must have matching funding and counterpart entries, including
the settlement authority. Settlement shortfalls require an explicit funding
transaction or a failed payment, not an unexplained reserve balance. Preserve
the distinction between inability to settle, insolvency, and missed debt service.

### The feedback loop

```text
Institutional objectives and permitted instruments
  -> proposed guidance conditions
  -> bank and borrower responses
  -> lending, spending, repayment, and balance-sheet changes
  -> observed outcomes and model errors
  -> reviewed guidance for the next period
```

Let `x_t` be the full accounting state, `g_t` the guidance conditions, and `e_t`
external shocks. A behavioral model `F_theta` supplies the transition:

```text
x_(t+1) = F_theta(x_t, g_t, e_t)
z_t     = h(x_t, available observations)
```

`z_t` is what a controller can observe, with publication lags and measurement
error. `F_theta` must enforce the accounting identities above. Credit volume
and composition emerge from loan demand and bank decisions under capital,
liquidity, and profitability constraints. They are not set directly by `g_t`.

Begin with a sectoral lending-growth corridor as the guidance instrument.
Specify how each bank responds, including partial compliance and delayed
adjustment. Borrowers may reduce demand or seek another source of finance.
Changes to interest rates, collateral rules, and the guidance perimeter are
separate experiments, not hidden consequences of changing a corridor.

## What AI contributes

In research, AI proposes competing behavioral equations and helps locate
evidence that can distinguish them. An explanation that fits a historical
episode is a candidate hypothesis, not a discovered law. Retain alternative
models when the observations cannot identify a unique mechanism.

In the simulator, an adaptive controller may propose guidance using past
observations. It must not see future shocks or the hidden state used by the
evaluator. Start with a transparent adaptive rule; an AI controller is a
separate comparator, not a new name for that rule.

Institutions choose the objectives and permitted actions. Report output,
prices, debt-service failures, and the distribution of gains and losses
separately before combining them into a score. Record any weights used. AI
does not settle the trade-off between these objectives.

## An AI-readable contract

Extend, rather than silently change, the current measurement interfaces.
The new simulator contract should distinguish these records:

| Record | Minimum content |
| --- | --- |
| Credit contract | Lender, borrower, currency, outstanding principal, interest terms, repayment schedule, and collateral reference. |
| Use evidence | Purpose category, amount covered, source, observation date, and unclassified share. |
| Model run | Model and data versions, behavioral parameters, initial state, shock path, seed, and observation lags. |
| Guidance proposal | Instrument, target corridor, objectives, constraints, model version, and review decision. |
| Result | Realized flows, balance sheets, outcomes by affected group, accounting residuals, and constraint breaches. |

Separate observed facts, imputed quantities, behavioral assumptions, and policy
choices in those records. The first demo uses synthetic aggregate agents, not
personal lending data. The existing MCP `evaluate_scenario` only applies
specified shocks; it does not yet implement this feedback loop.

## Experiments and rejection conditions

Use identical initial conditions and shock paths to compare no guidance,
fixed guidance, and adaptive guidance. Fix the objectives and tuning budget
before testing. Reserve different shock paths and behavioral models for
evaluation; do not retune on them and call the result out-of-sample.

Test delayed responses, weak compliance, collateral-price feedback, funding
substitution, and misspecified behavior. Publish failures and sensitivity to
parameters, not just the setting with the best-looking trajectory. A benefit
inside the simulator remains conditional on its assumptions.

Thermo-Credit variables are useful here only if a controller using them retains
the information needed to compare interventions. Compare them with ordinary
balance-sheet ratios and with a richer observed state under the same
information and tuning budgets. No physical conservation or entropy law is
assumed for the economy.

- If guidance fails under plausible response or substitution assumptions,
  narrow the claim to the conditions where it works.
- If an AI controller does not improve the stated outcomes over fixed and
  transparent adaptive rules, retain the simpler rule.
- If Thermo-Credit compression reverses intervention rankings or increases
  constraint failures, revise or reject that representation for control.

## Delivery order

1. Implement the accounting engine and event-level invariant tests. Publish
   a replayable synthetic example with balanced counterpart accounts.
2. Add behavioral responses and compare no guidance with fixed guidance.
   Publish trajectories and failed cases before tuning an adaptive controller.
3. Add the transparent adaptive baseline, then an AI controller. Evaluate
   both on reserved shocks and alternative behavioral models.
4. Use real data to restrict plausible mechanisms and parameters. Test the
   additional value of Thermo-Credit state variables before connecting an
   experimental guidance interface to the dashboard or MCP.

The current Japanese `q_t` measures borrower composition from sectoral loan
stocks, not gross new lending or final credit use. EU and US panels use coarser
proxies. These series can constrain selected observations; they cannot alone
identify bank responses to a new policy. Existing empirical results and the
[calibration protocol](calibration_protocol.md) remain in force.

## Starting references

- Werner (2012), [Towards a new research programme on "banking and the economy"](https://doi.org/10.1016/j.irfa.2012.06.002): Quantity Theory of Credit and the economic uses of credit.
- Werner (2014), [Can banks individually create money out of nothing?](https://doi.org/10.1016/j.irfa.2014.07.015): a study of lending and deposit creation using a cooperating bank's accounting records.
- Jakab and Kumhof (2015), [Banks are not intermediaries of loanable funds](https://www.bankofengland.co.uk/working-paper/2015/banks-are-not-intermediaries-of-loanable-funds-and-why-this-matters): modeling bank financing through money creation.
- Fukumoto et al. (2010), [Effectiveness of Window Guidance and Financial Environment](https://www.boj.or.jp/en/research/wps_rev/rev_2010/rev10e04.htm): Japan's experience and the limits of guidance as alternative funding channels expanded.
- [Current measurement definitions](definitions.md) and [MCP interface](thermo_credit_mcp_spec.md): existing implementation boundaries.
