# External partial-validation snapshots

These small extracts support two deliberately limited checks.

- `mof_manufacturing_bank_borrowing.csv` contains the manufacturing-column
  totals from sheets 26 and 32 of the Ministry of Finance quarterly historical
  liabilities workbook. The total is short-term plus long-term bank
  borrowings. It is a debtor-side, cross-source check on the manufacturing
  branch of the BOJ NFB borrower bucket. It is not loan-purpose validation.
- `mlit_private_housing_2024.csv` contains published FY2024 aggregates from the
  FY2025 Ministry of Land, Infrastructure, Transport and Tourism survey of
  private housing loans. The purpose shares are reported by institutions that
  supplied all three purpose categories. It directly validates only that the
  observed housing-loan branch is purpose-coded as housing acquisition or
  refinancing; it does not validate corporate real-estate loans or the full
  BOJ PROP bucket.

The original MOF workbook and MLIT PDF are not committed because the former is
an in-place, periodically revised historical workbook and the latter is much
larger than the extracted table. `metadata.json` records their official URLs,
retrieval date, source checksums, exact extraction locations, populations, and
claim boundaries. A changed upstream checksum means that the official file has
changed and the committed extract must not be silently treated as current.

The MOF validation uses four-quarter differences. This reduces sensitivity to
the April--June sample replacement described by MOF, but it does not eliminate
differences in samples, reporting populations, or lender coverage.
