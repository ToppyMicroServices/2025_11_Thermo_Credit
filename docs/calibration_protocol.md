# Calibration protocol

This protocol prevents an iterative parameter search from turning the same
holdout sample into training data.

## Fixed sequence

1. Register the target, baseline, candidate formula, parameter bounds, loss,
   training window, holdout window, and random seed.
2. Fit transformations and parameters using the training observations only.
3. Freeze the fitted parameter set and its input-data hashes.
4. Evaluate once on the fixed holdout or on a sequence of new release vintages.
5. Report the baseline, candidate, uncertainty interval, failed variants, and
   every registered gate.

Rolling or expanding estimation is allowed only when the rule is fixed in
advance. At forecast origin `t`, all scaling and fitting must use observations
available by `t`. Future labels and later revisions must not enter the fit.

## Retuning rule

Do not retune a candidate after inspecting its holdout loss and then report the
same holdout as validation. A changed formula, parameter bound, loss, feature,
or preprocessing rule creates a new model version. That version needs a fresh
holdout period or later data vintage.

Exploratory tuning on the full retrospective sample is permitted for model
development, but its output must be labelled in-sample and cannot pass an OOS
or submission-readiness gate.

## Parameter record

Each calibrated release should record:

- model and mapping version;
- training and evaluation dates;
- data-file SHA-256 hashes and release-vintage identifiers;
- objective, parameter bounds, and selected parameters;
- preprocessing and missing-data rules;
- random seed and software versions;
- baseline and candidate losses on the common sample.

`data/calibrated_theory_params.json` is a parameter artifact, not evidence that
the parameters are structurally identified. The current holdout results do not
show that calibrated `X_C` reliably beats raw `X_C` or a simple trailing-change
baseline. The dashboard therefore labels `X_C` as an experimental diagnostic.

## Promotion gate

A calibrated variable may be promoted from diagnostic to predictive indicator
only if it improves a registered baseline on untouched observations, remains
stable across reasonable windows, and keeps the same direction under source and
preprocessing sensitivity checks. Failure is a result and must remain visible.
