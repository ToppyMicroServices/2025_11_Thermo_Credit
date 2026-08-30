# BOJ release-vintage protocol

This directory adds a prospective evidence layer to the BOJ measurement
bridge. It does not relabel the current analysis as preregistered and does not
replace the manuscript's existing pseudo-OOS application.

## Evidence classes

| Manifest `capture_class` | Meaning | Prospective scoring |
|---|---|---|
| `protocol_baseline_current_vintage_seed` | A frozen copy of data already available when the protocol was written. It is useful for checking the archive and preserving the current-vintage baseline. | Never eligible |
| `prospective_release_vintage` | The exact BOJ API response first captured for a release strictly after the external protocol-registration timestamp. | Eligible under the frozen protocol |

The distinction is enforced by
`eligible_for_prospective_scoring`. The score writer rejects a seed even if it
is otherwise complete.

## 1. Create the current-vintage seed

The existing processed BOJ panel can initialize and test the archive. Its BOJ
`LAST_UPDATE` date is recorded with date-only precision; it is not treated as
an exact historical release time or as prospective evidence.

```sh
python3 scripts/24_capture_boj_vintage.py \
  --capture-class protocol_baseline_current_vintage_seed \
  --input data/credit_destination_jp.csv
```

Re-running the same command is idempotent. It returns the existing vintage
rather than writing another copy.

## 2. Register before collecting prospective evidence

Review [OSF_PROTOCOL.md](OSF_PROTOCOL.md) and `protocol.json`. Register or
time-stamp those files manually in the chosen repository. This code does not
create an OSF project and does not upload anything.

Only after registration, replace the null
`registration_timestamp_utc` in `protocol.json` with the exact public
registration timestamp and commit that metadata update. A release at or before
that timestamp remains non-prospective. Do not backdate the timestamp and do
not call any observation already analyzed preregistered.

## 3. Capture each future BOJ release

Run the capture promptly after a scheduled release. Supply the publisher's
release timestamp when it is available. If only the observed retrieval time is
known, label the timestamp as an observed upper bound.

```sh
python3 scripts/24_capture_boj_vintage.py \
  --capture-class prospective_release_vintage \
  --fetch \
  --release-timestamp 2026-08-20T08:50:00+09:00 \
  --release-timestamp-source "BOJ release calendar" \
  --release-timestamp-precision minute \
  --end-date 202602
```

Each completed directory contains:

- the exact response bytes and their SHA-256 checksum;
- the complete BOJ series-ID list and exact source URL;
- release and retrieval timestamps with source and precision;
- the frozen Bezemer four-way primary mapping, primary vector and scale
  columns, Werner and Müller–Verner robustness mappings, Appendix-only legacy
  G/B/E mapping, and mapping version;
- the frozen protocol, outcome, horizon, benchmark, loss, and revision rules;
- the Git commit, dirty-worktree flag, and capture-script checksums.

Records are content addressed. Identical recapture is idempotent. If content at
the same reported release timestamp differs, a new vintage is created with a
revision sequence and links to the prior vintage; no completed record is
overwritten.

## 4. Verify and score

Verify all vintage, mapping, protocol, and score checksums:

```sh
python3 scripts/26_verify_boj_vintage_archive.py
```

After a declared outcome is released, append a score using values on the scale
frozen by the protocol. The realization checksum should identify the exact
outcome-source payload used.

```sh
python3 scripts/25_score_boj_vintage.py \
  --vintage-id BOJ_VINTAGE_ID \
  --outcome-id jgb_yield_change \
  --horizon-quarters 4 \
  --target-period 2027Q2 \
  --benchmark-forecast 0.48 \
  --candidate-forecast 0.44 \
  --realized-value 0 \
  --realization-release-timestamp 2027-08-16T08:50:00+09:00 \
  --realization-source-url "https://example.invalid/exact-official-source" \
  --realization-payload-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

The example URL and checksum are placeholders and must not be reused. Score
records are also append-only and content addressed. A changed official
realization becomes a linked revision rather than replacing the first-release
score.

## Limits

The archive makes the information set auditable; it does not itself establish
incremental predictability. Ordinary filesystem permissions cannot prevent a
deliberate deletion, so the archive should also be backed up in a
write-once/versioned store and its checksums included in later anonymous
replication releases. Capture manifests omit author names, account identifiers,
and absolute local paths.
