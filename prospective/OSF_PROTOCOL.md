# Anonymous prospective BOJ measurement-bridge protocol

## Status and evidence boundary

This document is a protocol for observations released after external
registration. The current analyzed sample predates registration. It is
retrospective current-vintage evidence and must not be described as
preregistered or used in the prospective score sequence.

The protocol tests a falsifiable application of a BOJ borrower-composition
measurement bridge. It does not preregister a claim that incremental
predictability has already been established.

## Frozen measurement

The primary predictor population and crosswalk are the four-way Japan
classification from Bezemer, Samarina, and Zhang (2020): non-financial
business (NFB), financial business (FIN), property/mortgage (PROP), and
household non-housing (HH_NONHOUSING). Scale is growth in
`primary_included_stock`. Allocation is the four-quarter flow-weighted NFB
share,
`sum_4Q(C_NFB)/sum_4Q(C_NFB+C_FIN+C_PROP+C_HH_NONHOUSING)`, using positive
parts after aggregation within those published buckets.

Werner (1997) and Müller and Verner (2024) are frozen as robustness
taxonomies over the same credit population. The author-defined G/B/E grouping
is Appendix-only legacy output and cannot become the prospective primary
mapping. Each release snapshot stores the primary taxonomy ID, vector columns,
scale column, ex-ante taxonomy-selection record, all taxonomy definitions,
mapping version, and every BOJ series ID.

## Prospective eligibility

1. The protocol and mapping are externally time-stamped before a qualifying
   BOJ release.
2. A qualifying release timestamp is strictly later than that registration.
3. The response is obtained directly from the BOJ API and captured without
   overwriting earlier records.
4. The manifest contains the exact source URL, release and retrieval times,
   raw-payload checksum, mapping and protocol checksums, and code identity.
5. A missed release may be retained as an audit but cannot be backfilled into
   the primary prospective sequence.

A local seed made from the current panel is permanently marked
`protocol_baseline_current_vintage_seed` and is ineligible for prospective
scoring.

## Evaluation

The frozen outcomes, horizons, benchmark, candidate, training rule, loss, and
revision policy are defined in `protocol.json` and copied into every manifest.
The primary comparison is candidate loss minus matched-stock benchmark loss.
Negative values are lower point loss. Inference and any stopping rule must be
specified before interpreting an accumulated score sequence; individual
negative losses alone do not establish incremental predictability.

## Revisions and preservation

The first successfully captured post-registration response is the primary
predictor vintage for that release. Later content differences are retained as
linked revision vintages. The first official outcome release is the primary
realization; later revisions are separate sensitivity scores. Every completed
record is content addressed and checksum verified.

## Registration procedure

Remove no limitations from this document. Review the anonymous files, deposit
or register them manually in OSF or another public time-stamping repository,
and record the returned UTC timestamp in `protocol.json`. Do not add author
names, affiliations, local paths, account IDs, or private repository metadata
to the anonymous deposit.

No script in this repository uploads to OSF or another external service.
