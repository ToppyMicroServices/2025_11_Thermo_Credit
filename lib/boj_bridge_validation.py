from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.boj_credit_taxonomies import (
    LEGACY_TAXONOMY_ID,
    MULLER_VERNER_TAXONOMY_ID,
    PRIMARY_BUCKETS,
    PRIMARY_TAXONOMY_ID,
    TAXONOMY_METADATA,
    TAXONOMY_SELECTION_RULE,
    WERNER_TAXONOMY_ID,
)


LAMBDA_B_DEFAULT = 0.5
SIGNED_DENOMINATOR_ATOL = 1e-9
MAIN_VALIDATION_AUDITS = (
    "One-quarter signed-change diagnostic",
    "Aggregation-order sensitivity",
    "Quarter-of-year denominator availability",
    "Four-quarter within-BOJ purpose-coded comparator",
    "Construction-borrower reclassification",
    "Real-estate borrower reclassification",
    "Finance and household borrower alternatives",
    "Series availability and aggregate reconciliation",
)
PRIMARY_VALIDATION_AUDITS = (
    "Primary same-population identity",
    "Published residual NFB decomposition",
    "Household non-housing split",
    "Explicit official-population coverage",
    "Literature-anchored taxonomy population identity",
    "Primary four-quarter vector availability",
)


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _validation_latex(value: Any) -> str:
    text = _latex_escape(value)
    return (
        text.replace("|delta q|", r"$|\Delta q|$")
        .replace("q\\_t", r"$q_t$")
        .replace(" q ", r" $q$ ")
    )


def _format_pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{100.0 * float(value):.{digits}f}%"


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _net_bucket(frame: pd.DataFrame, bucket: str) -> pd.Series:
    direct = _numeric(frame, f"C_{bucket}_net")
    if direct.notna().any():
        return direct
    stock = _numeric(frame, f"stock_{bucket}")
    if stock.notna().any():
        return stock.diff().fillna(0.0)
    return direct


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator > 0)
    return numerator / denom


def _four_quarter_composition(
    group_g: pd.Series,
    group_b: pd.Series,
    group_e: pd.Series,
) -> pd.DataFrame:
    """Return G/B/E coordinates formed with one common four-quarter denominator."""
    rolling_g = group_g.rolling(window=4, min_periods=4).sum()
    rolling_b = group_b.rolling(window=4, min_periods=4).sum()
    rolling_e = group_e.rolling(window=4, min_periods=4).sum()
    rolling_total = rolling_g + rolling_b + rolling_e
    return pd.DataFrame(
        {
            "G": _safe_div(rolling_g, rolling_total),
            "B": _safe_div(rolling_b, rolling_total),
            "E": _safe_div(rolling_e, rolling_total),
        }
    )


def _signed_div(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    atol: float = SIGNED_DENOMINATOR_ATOL,
) -> pd.Series:
    denom = denominator.where(denominator.abs() > atol)
    return numerator / denom


def _latest(series: pd.Series) -> float:
    finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float("nan")
    return float(finite.iloc[-1])


def _median(series: pd.Series) -> float:
    finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float("nan")
    return float(finite.median())


def _mean(series: pd.Series) -> float:
    finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float("nan")
    return float(finite.mean())


def _corr(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return len(pair), float("nan")
    if pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
        return len(pair), float("nan")
    return len(pair), float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _spearman(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return len(pair), float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return len(pair), float("nan")
    ranked = pair.rank(method="average")
    return len(pair), float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def _median_abs_diff(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return float("nan")
    return float((pair.iloc[:, 0] - pair.iloc[:, 1]).abs().median())


def _share(mask: pd.Series) -> float:
    valid = mask.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.astype(bool).mean())


def _sum_share(frame: pd.DataFrame, column: str) -> float:
    numerator = _numeric(frame, column).sum(skipna=True)
    denominator = _numeric(frame, "total_positive_flow").sum(skipna=True)
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _sum_ratio(frame: pd.DataFrame, numerator_column: str, denominator_column: str) -> float:
    numerator = _numeric(frame, numerator_column).sum(skipna=True)
    denominator = _numeric(frame, denominator_column).sum(skipna=True)
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _has_primary_taxonomy(frame: pd.DataFrame) -> bool:
    return all(f"C_{bucket}" in frame.columns for bucket in PRIMARY_BUCKETS)


def _primary_sum_share(frame: pd.DataFrame, column: str) -> float:
    denominator = sum(_numeric(frame, f"C_{bucket}") for bucket in PRIMARY_BUCKETS).sum(
        skipna=True
    )
    numerator = _numeric(frame, column).sum(skipna=True)
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _common_availability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validity_column = (
        "primary_taxonomy_delta_valid"
        if _has_primary_taxonomy(frame) and "primary_taxonomy_delta_valid" in frame.columns
        else "common_taxonomy_delta_valid"
    )
    if validity_column in frame.columns:
        valid = frame[validity_column]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame.loc[valid].copy()
    stock_columns = [
        column
        for column in frame.columns
        if column.startswith("stock_")
        and column not in {"stock_G", "stock_B", "stock_E", "stock_total"}
    ]
    if not stock_columns:
        return frame
    complete = frame[stock_columns].notna().all(axis=1)
    return frame.loc[complete].copy() if complete.any() else frame


def _display_code(code: str) -> str:
    return code.replace("_", " / ")


def _code_summary(codes: Sequence[str]) -> str:
    cleaned = [_display_code(str(code)) for code in codes]
    if len(cleaned) <= 4:
        return "; ".join(cleaned)
    examples = "; ".join(cleaned[:2] + [cleaned[-1]])
    return f"{len(cleaned)} codes in replication metadata; examples: {examples}"


def _codes(rows: Sequence[Mapping[str, Any]]) -> str:
    return _code_summary([str(row["stock_code"]) for row in rows])


def mapping_rows(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    frame = _common_availability_frame(frame)
    bucket_mapping = list(metadata.get("bucket_mapping", []))
    g_rows = [row for row in bucket_mapping if row.get("bucket") == "G"]
    b_rows = [row for row in bucket_mapping if row.get("bucket") == "B"]
    e_rows = [row for row in bucket_mapping if row.get("bucket") == "E"]
    u_rows = [row for row in bucket_mapping if row.get("bucket") == "U"]
    fixed_codes = [str(row["stock_code"]).replace("DLLI5DS2", "DLLI5DS5", 1) for row in bucket_mapping]

    return [
        {
            "source": f"Outstanding loans by sector, {len(g_rows)} author-defined Group G series: {_codes(g_rows)}",
            "borrower": "Manufacturing, trade, services, utilities, transport, local governments, and other organisations",
            "bucket": "Group G: broad sectors, local governments, and other organisations",
            "rationale": "Author-defined broad borrower grouping used to construct the composition share.",
            "ambiguous": "Borrower sector does not reveal whether funds finance operations, fixed assets, refinancing, or financial assets.",
            "type": "sectoral stock-difference",
            "bias": "Measures borrower composition, not loan purpose or macroeconomic use.",
            "negative": "Sector stocks are summed within the bucket before the positive part; the signed audit keeps the bucket net change.",
            "lag": "Quarterly BOJ LA01 release; forecast panels apply release-lag dating.",
            "coverage": _format_pct(_sum_share(frame, "C_G")),
        },
        {
            "source": f"Construction outstanding: {_codes(b_rows)}",
            "borrower": "Construction",
            "bucket": "Group B: construction",
            "rationale": "Author-defined construction borrower group kept separate without fractional allocation.",
            "ambiguous": "Building finance, land purchase, working capital, and refinancing are not separated.",
            "type": "sectoral stock-difference",
            "bias": "A separate borrower bucket does not identify construction loan purpose.",
            "negative": "The bucket net change is computed before the positive part; the signed audit keeps the sign.",
            "lag": "Quarterly BOJ LA01 release; forecast panels apply release-lag dating.",
            "coverage": _format_pct(_sum_share(frame, "C_B")),
        },
        {
            "source": f"Finance, real-estate, and household outstanding sectors: {_codes(e_rows)}",
            "borrower": "Finance and insurance; real estate; households",
            "bucket": "Group E: finance, real estate, and households",
            "rationale": "Author-defined grouping of the remaining mapped domestic borrower sectors without a common purpose assignment.",
            "ambiguous": "Real-estate, finance, housing, and household-consumption uses are mixed.",
            "type": "sectoral stock-difference",
            "bias": "The group is heterogeneous and must not be read as an asset-purpose measure.",
            "negative": "Sector stocks are summed within the bucket before the positive part; the signed audit keeps the bucket net change.",
            "lag": "Quarterly BOJ LA01 release; forecast panels apply release-lag dating.",
            "coverage": _format_pct(_sum_share(frame, "C_E")),
        },
        {
            "source": f"Unclassified outstanding sector: {_codes(u_rows)}",
            "borrower": "Overseas yen loans and domestic loans transferred overseas",
            "bucket": "Excluded from domestic borrower composition",
            "rationale": "The reported category is not a domestic borrower-sector group comparable with G, B, or E.",
            "ambiguous": "Could contain many borrower sectors and loan uses outside the domestic map.",
            "type": "sectoral stock-difference",
            "bias": "Exclusion narrows the mapped domestic denominator; direction for the Group G share is unknown.",
            "negative": "Its bucket-positive change enters the selected-series coverage denominator; the signed audit keeps the sign.",
            "lag": "Quarterly BOJ LA01 release; forecast panels apply release-lag dating.",
            "coverage": _format_pct(_sum_share(frame, "unclassified_positive_flow")),
        },
        {
            "source": "New loans for fixed investment by sector: " + _code_summary(fixed_codes),
            "borrower": "Same sector map as the outstanding-loan series",
            "bucket": "Within-BOJ purpose-coded comparator",
            "rationale": "Provides a purpose-coded fixed-investment comparator within the same BOJ sector map.",
            "ambiguous": "Excludes working capital, refinancing, rollovers, and many asset purchases.",
            "type": "claim-boundary audit",
            "bias": "Narrower than total lending and tilted toward investment-intensive borrowers.",
            "negative": "Used as reported and kept separate from the borrower-stock-change measure.",
            "lag": "Quarterly BOJ LA01 release; forecast panels apply release-lag dating.",
            "coverage": "audit only",
        },
    ]


def primary_mapping_rows(
    frame: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> list[dict[str, str]]:
    frame = _common_availability_frame(frame)
    citation = (
        "Bezemer, Samarina, and Zhang (2020), Journal of Banking & Finance 113, "
        "105760; Japan crosswalk in DNB Working Paper 559."
    )
    return [
        {
            "source": "Published Japan residual NFB",
            "role": r"$P_t^{NFB}$",
            "rationale": (
                "Official total less finance/insurance, real estate, local government, "
                "and total household loans; construction is included."
            ),
            "risk": (
                "The residual also absorbs the separately disclosed overseas-linked and "
                "unresolved components."
            ),
            "share": _format_pct(_primary_sum_share(frame, "C_NFB")),
        },
        {
            "source": "Finance and insurance",
            "role": r"$P_t^{FIN}$",
            "rationale": "Direct BOJ borrower-sector match to financial business.",
            "risk": "Borrower sector does not identify the use of funds.",
            "share": _format_pct(_primary_sum_share(frame, "C_FIN")),
        },
        {
            "source": "Real estate plus household housing",
            "role": r"$P_t^{PROP}$",
            "rationale": "Published Japan property/mortgage crosswalk.",
            "risk": "Combines a borrower-sector stock with a household-purpose stock.",
            "share": _format_pct(_primary_sum_share(frame, "C_PROP")),
        },
        {
            "source": "Household total less household housing",
            "role": r"$P_t^{HHN}$",
            "rationale": "Published residual construction for household non-housing credit.",
            "risk": "Includes tax and other household borrowing; it is not labelled consumer credit.",
            "share": _format_pct(_primary_sum_share(frame, "C_HH_NONHOUSING")),
        },
        {
            "source": "Local governments",
            "role": "excluded, explicit",
            "rationale": "Removed by the published Japan residual formula.",
            "risk": "Exclusion changes the population relative to the official total.",
            "share": _format_pct(
                _sum_ratio(
                    frame,
                    "local_governments_positive_flow",
                    "explicit_scope_positive_flow",
                )
            ),
        },
        {
            "source": "Overseas-linked loans",
            "role": "NFB component, explicit",
            "rationale": "Disclosed because the current BOJ total makes it part of residual NFB.",
            "risk": "Not a domestic borrower-sector observation.",
            "share": _format_pct(
                _sum_ratio(frame, "overseas_positive_flow", "primary_positive_flow")
            ),
        },
        {
            "source": "Unresolved residual",
            "role": "NFB component, explicit",
            "rationale": "Residual after mapped NFB industries and overseas-linked loans.",
            "risk": "Has no independent borrower or purpose label.",
            "share": _format_pct(
                _sum_ratio(frame, "unresolved_residual_positive_flow", "primary_positive_flow")
            ),
        },
        {
            "source": "BOJ consumer-purpose loans",
            "role": "validation comparator",
            "rationale": "Checks, but does not define, the household non-housing residual.",
            "risk": "Same provider and narrower concept; not independent external validation.",
            "share": "validation audit",
        },
        {
            "source": citation,
            "role": "literature source",
            "rationale": (
                "Taxonomy definitions are literature-anchored and all three are "
                "reported jointly."
            ),
            "risk": "A published crosswalk does not eliminate BOJ measurement error.",
            "share": "not selected on fit",
        },
    ]


def taxonomy_robustness_rows(metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    taxonomies = metadata.get("taxonomies", TAXONOMY_METADATA)
    selection = metadata.get("taxonomy_selection", TAXONOMY_SELECTION_RULE)
    ids = (
        PRIMARY_TAXONOMY_ID,
        WERNER_TAXONOMY_ID,
        MULLER_VERNER_TAXONOMY_ID,
        LEGACY_TAXONOMY_ID,
    )
    rows: list[dict[str, str]] = []
    for taxonomy_id in ids:
        entry = taxonomies.get(taxonomy_id, TAXONOMY_METADATA[taxonomy_id])
        rows.append(
            {
                "taxonomy_id": taxonomy_id,
                "role": str(entry.get("role", "")),
                "population": str(entry.get("population", "")),
                "construction": str(entry.get("construction_placement", "")),
                "selection": (
                    str(selection.get("selection_basis", ""))
                    if taxonomy_id != LEGACY_TAXONOMY_ID
                    else "Appendix compatibility only."
                ),
            }
        )
    return rows


def main_mapping_rows(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    if str(metadata.get("primary_taxonomy_id", "")) == PRIMARY_TAXONOMY_ID:
        return primary_mapping_rows(frame, metadata)[:7]
    rows = mapping_rows(frame, metadata)
    by_bucket = {row["bucket"]: row for row in rows}
    return [
        {
            "source": "Group G: broad sectors, local governments, and other organisations",
            "role": r"$P_t^G$",
            "rationale": "Author-defined broad borrower group.",
            "risk": "Sector labels do not identify loan use.",
            "share": by_bucket[
                "Group G: broad sectors, local governments, and other organisations"
            ]["coverage"],
        },
        {
            "source": "Group B: construction",
            "role": r"$P_t^B$",
            "rationale": "Author-defined construction borrower group.",
            "risk": "Loan purpose is mixed within the borrower sector.",
            "share": by_bucket["Group B: construction"]["coverage"],
        },
        {
            "source": "Group E: finance, real estate, households",
            "role": r"$P_t^E$",
            "rationale": "Author-defined remaining mapped borrower group.",
            "risk": "Finance, real-estate, and household uses are heterogeneous.",
            "share": by_bucket["Group E: finance, real estate, and households"]["coverage"],
        },
        {
            "source": "Overseas-linked unclassified",
            "role": "excluded",
            "rationale": "Not comparable with a domestic borrower-sector group.",
            "risk": "Exclusion narrows the mapped denominator; share bias is unknown.",
            "share": by_bucket["Excluded from domestic borrower composition"]["coverage"],
        },
        {
            "source": "Fixed-investment new lending",
            "role": "claim-boundary audit",
            "rationale": "Within-BOJ purpose-coded comparator.",
            "risk": "Narrower than total lending; not independent external evidence.",
            "share": "claim-boundary audit",
        },
    ]


def render_main_mapping_table(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> str:
    rows = main_mapping_rows(frame, metadata)
    is_primary = str(metadata.get("primary_taxonomy_id", "")) == PRIMARY_TAXONOMY_ID
    valid_share_frame = _common_availability_frame(frame)
    valid_dates = (
        pd.to_datetime(valid_share_frame["date"], errors="coerce").dropna()
        if "date" in valid_share_frame.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    share_period = ""
    if not valid_dates.empty:
        first_date = valid_dates.min()
        last_date = valid_dates.max()
        share_period = (
            f" ({first_date.year}Q{first_date.quarter}--"
            f"{last_date.year}Q{last_date.quarter})"
        )
    if is_primary:
        share_note = (
            rf"For the four primary buckets, the displayed share is "
            rf"$\sum_t P_t^k/\sum_t(P_t^{{NFB}}+P_t^{{FIN}}+"
            rf"P_t^{{PROP}}+P_t^{{HHN}})$ over the "
            rf"{len(valid_share_frame)} valid within-taxonomy quarterly changes"
            rf"{share_period}. Explicit component rows use the denominator stated by "
            rf"their construction."
        )
        caption = "Literature-anchored BOJ credit-allocation crosswalk."
    else:
        share_note = (
            rf"For $k\in\{{G,B,E,U\}}$, the displayed share is "
            rf"$\sum_t P_t^k/\sum_t(P_t^G+P_t^B+P_t^E+P_t^U)$ over the "
            rf"{len(valid_share_frame)} valid within-taxonomy quarterly changes"
            rf"{share_period}."
        )
        caption = "Author-defined BOJ borrower-composition groups."
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"  \label{tab:boj_bridge_mapping}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular}{@{}p{0.18\textwidth}p{0.13\textwidth}p{0.24\textwidth}p{0.28\textwidth}p{0.08\textwidth}@{}}",
        r"    \toprule",
        (
            r"    BOJ input / component & Bridge role & Construction & Main measurement risk & Primary / explicit flow share \\"
            if is_primary
            else r"    BOJ borrower group & Bridge role & Grouping rationale & Main measurement risk & Selected-series share (including U) \\"
        ),
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                (
                    row["role"]
                    if key == "role"
                    else _validation_latex(row[key])
                )
                for key in ("source", "role", "rationale", "risk", "share")
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            (
                r"  \par\smallskip\raggedright\footnotesize The primary four-bucket mapping follows Bezemer, Samarina, and Zhang (2020; Japan crosswalk in DNB Working Paper 559); \texttt{q\_t} is the four-quarter NFB coordinate. Construction is an NFB detail. Overseas-linked and unresolved amounts absorbed by the published residual are shown explicitly. "
                if is_primary
                else r"  \par\smallskip\raggedright\footnotesize The G/B/E labels and $P_t^G/P_t^B/P_t^E$ symbols are author-defined, not official BOJ categories; they correspond to replication columns \texttt{C\_G}/\texttt{C\_B}/\texttt{C\_E}. "
            )
            + share_note
            + (
                r" Positive parts are taken after net changes are aggregated within each fixed bucket. The current-sample mappings are literature-anchored and reported jointly; this declaration is not an externally time-stamped preregistration."
                if is_primary
                else r" Positive parts are taken after net changes are aggregated within borrower groups. Stock levels are jointly observed from 2009Q2; the first valid within-taxonomy change is 2009Q3. The overseas-linked series enters only the selected-series coverage denominator. These shares are not coverage of all BOJ credit."
            ),
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_primary_mapping_table(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> str:
    rows = primary_mapping_rows(frame, metadata)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Published-taxonomy BOJ crosswalk and explicit residuals.}",
        r"  \label{tab:boj_primary_mapping_detail}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}p{0.20\textwidth}p{0.14\textwidth}p{0.27\textwidth}p{0.30\textwidth}p{0.09\textwidth}@{}}",
        r"    \toprule",
        r"    BOJ input / source & Bridge role & Construction & Remaining measurement risk & Flow share \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                row["role"] if key == "role" else _validation_latex(row[key])
                for key in ("source", "role", "rationale", "risk", "share")
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"  \par\smallskip\raggedright\footnotesize The four primary flow coordinates share one included population. Local government is outside that population. Overseas-linked and unresolved amounts are disclosed as components absorbed by the published NFB residual, not silently discarded.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_taxonomy_robustness_table(metadata: Mapping[str, Any]) -> str:
    rows = taxonomy_robustness_rows(metadata)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Literature-anchored credit-taxonomy placements.}",
        r"  \label{tab:boj_taxonomy_robustness}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}p{0.20\textwidth}p{0.14\textwidth}p{0.25\textwidth}p{0.25\textwidth}p{0.16\textwidth}@{}}",
        r"    \toprule",
        r"    Taxonomy identifier & Role & Population & Construction placement & Selection rule \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                _validation_latex(row[key])
                for key in ("taxonomy_id", "role", "population", "construction", "selection")
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"  \par\smallskip\raggedright\footnotesize The primary and robustness mappings are defined in replication metadata and reported jointly. The current-sample declaration is not an externally time-stamped preregistration; the prospective archive freezes these definitions only for future releases.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_mapping_table(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> str:
    rows = mapping_rows(frame, metadata)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Detailed BOJ source series to borrower-composition buckets.}",
        r"  \label{tab:boj_bridge_mapping_detail}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}p{0.15\textwidth}p{0.12\textwidth}p{0.10\textwidth}p{0.13\textwidth}p{0.13\textwidth}p{0.08\textwidth}p{0.12\textwidth}p{0.10\textwidth}p{0.08\textwidth}p{0.06\textwidth}@{}}",
        r"    \toprule",
        r"    BOJ series name / code & Borrower sector & Borrower bucket & Grouping rationale & Ambiguous cases & Type & Expected bias direction & Negative quarterly changes & Assumed release lag & Series share (incl.\ U) \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                _latex_escape(row[key])
                for key in (
                    "source",
                    "borrower",
                    "bucket",
                    "rationale",
                    "ambiguous",
                    "type",
                    "bias",
                    "negative",
                    "lag",
                    "coverage",
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"  \par\smallskip\raggedright\footnotesize Shares use positive parts taken after net changes are aggregated within borrower buckets. Stock levels are jointly observed from 2009Q2; the first valid within-taxonomy change is 2009Q3. They are selected-series composition shares, not coverage of the official BOJ aggregate.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _legacy_validation_rows(
    frame: pd.DataFrame,
    lambda_b: float = LAMBDA_B_DEFAULT,
) -> list[dict[str, str]]:
    availability_frame = frame.copy()
    frame = _common_availability_frame(frame)
    _ = lambda_b  # Kept for API compatibility; the primary borrower-composition metric has no weight.
    c_t = _numeric(frame, "C_t")
    c_g = _numeric(frame, "C_G")
    c_b = _numeric(frame, "C_B")
    c_e = _numeric(frame, "C_E")
    bucket_positive_total = c_g + c_b + c_e
    q_positive = _safe_div(c_g, bucket_positive_total)
    composition_4q = _four_quarter_composition(c_g, c_b, c_e)
    q_4q = composition_4q["G"]

    c_g_series_positive = _numeric(frame, "C_G_series_positive")
    c_b_series_positive = _numeric(frame, "C_B_series_positive")
    c_e_series_positive = _numeric(frame, "C_E_series_positive")
    series_positive_total = c_g_series_positive + c_b_series_positive + c_e_series_positive
    q_series_positive = _safe_div(
        c_g_series_positive.rolling(window=4, min_periods=4).sum(),
        series_positive_total.rolling(window=4, min_periods=4).sum(),
    )
    aggregation_n, aggregation_corr = _corr(q_4q, q_series_positive)
    aggregation_mad = _median_abs_diff(q_4q, q_series_positive)
    aggregation_pair = pd.concat(
        [
            q_4q.rename("q_bucket_positive"),
            q_series_positive.rename("q_series_positive"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    aggregation_abs_diff = (
        aggregation_pair["q_bucket_positive"] - aggregation_pair["q_series_positive"]
    ).abs()
    aggregation_p90 = (
        float(aggregation_abs_diff.quantile(0.90))
        if not aggregation_abs_diff.empty
        else float("nan")
    )
    aggregation_max = (
        float(aggregation_abs_diff.max())
        if not aggregation_abs_diff.empty
        else float("nan")
    )
    aggregation_bucket_median = _median(aggregation_pair["q_bucket_positive"])
    aggregation_series_median = _median(aggregation_pair["q_series_positive"])

    c_g_net = _net_bucket(frame, "G")
    c_b_net = _net_bucket(frame, "B")
    c_e_net = _net_bucket(frame, "E")
    net_total = c_g_net + c_b_net + c_e_net
    q_net = _signed_div(c_g_net, net_total)
    net_paired_n, net_corr = _corr(q_positive, q_net)
    net_mad = _median_abs_diff(q_positive, q_net)
    signed_frame = pd.concat(
        [q_net.rename("q_net"), net_total.rename("net_total")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    net_n = int(len(signed_frame))
    net_contractions = int((signed_frame["net_total"] < 0.0).sum())
    net_outside = int(
        ((signed_frame["q_net"] < 0.0) | (signed_frame["q_net"] > 1.0)).sum()
    )
    net_pair = pd.concat(
        [q_positive.rename("q_positive"), q_net.rename("q_net")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    net_abs_diff = (net_pair["q_positive"] - net_pair["q_net"]).abs()
    net_p90 = float(net_abs_diff.quantile(0.90)) if not net_abs_diff.empty else float("nan")
    net_min = float(signed_frame["q_net"].min()) if not signed_frame.empty else float("nan")
    net_max = float(signed_frame["q_net"].max()) if not signed_frame.empty else float("nan")

    fixed_total = (
        _numeric(frame, "fixed_investment_new_G")
        + _numeric(frame, "fixed_investment_new_B")
        + _numeric(frame, "fixed_investment_new_E")
    )
    metric_dates = (
        pd.to_datetime(frame["date"], errors="coerce")
        if "date" in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    metric_quarters = metric_dates.dt.quarter
    zero_denominator = bucket_positive_total.eq(0.0) & bucket_positive_total.notna()
    q_4q_valid = q_4q.notna()
    q_4q_n = int(q_4q_valid.sum())
    q_4q_first = metric_dates[q_4q_valid].min() if q_4q_valid.any() else pd.NaT
    quarter_denominator_parts: list[str] = []
    for quarter in range(1, 5):
        quarter_rows = metric_quarters.eq(quarter)
        quarter_n = int(quarter_rows.sum())
        quarter_zero = int((zero_denominator & quarter_rows).sum())
        quarter_denominator_parts.append(f"Q{quarter}={quarter_zero}/{quarter_n}")
    zero_denominator_n = int(zero_denominator.sum())
    q_fixed_4q = _safe_div(
        _numeric(frame, "fixed_investment_new_G").rolling(window=4, min_periods=4).sum(),
        fixed_total.rolling(window=4, min_periods=4).sum(),
    )
    fixed_n, fixed_corr = _corr(q_4q, q_fixed_4q)
    _, fixed_spearman = _spearman(q_4q, q_fixed_4q)
    fixed_mad = _median_abs_diff(q_4q, q_fixed_4q)
    fixed_pair = pd.concat(
        [q_4q.rename("stock_change_share"), q_fixed_4q.rename("purpose_share")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    fixed_change_pair = fixed_pair.diff().dropna()
    fixed_change_n, fixed_change_spearman = _spearman(
        fixed_change_pair["stock_change_share"],
        fixed_change_pair["purpose_share"],
    )
    fixed_direction_agreement = (
        float(
            (
                np.sign(fixed_change_pair["stock_change_share"])
                == np.sign(fixed_change_pair["purpose_share"])
            ).mean()
        )
        if not fixed_change_pair.empty
        else float("nan")
    )
    fixed_stock_mean = _mean(fixed_pair["stock_change_share"])
    fixed_purpose_mean = _mean(fixed_pair["purpose_share"])
    fixed_mean_bias = _mean(
        fixed_pair["stock_change_share"] - fixed_pair["purpose_share"]
    )

    def reclassified_composition(
        source_net: pd.Series,
        *,
        target_bucket: str,
    ) -> pd.DataFrame:
        if target_bucket == "G":
            g_alt = (c_g_net + source_net).clip(lower=0.0)
            b_alt = c_b
        elif target_bucket == "B":
            g_alt = c_g
            b_alt = (c_b_net + source_net).clip(lower=0.0)
        else:
            raise ValueError(f"Unsupported target bucket: {target_bucket}")
        e_alt = (c_e_net - source_net).clip(lower=0.0)
        return _four_quarter_composition(g_alt, b_alt, e_alt)

    zero_flow = pd.Series(0.0, index=frame.index, dtype=float)
    construction_as_group_g = _four_quarter_composition(
        (c_g_net + c_b_net).clip(lower=0.0),
        zero_flow,
        c_e,
    )["G"]
    construction_shift = (construction_as_group_g - q_4q).abs()
    real_estate_net = _numeric(frame, "delta_net_real_estate")
    real_estate_to_construction = reclassified_composition(
        real_estate_net,
        target_bucket="B",
    )["G"]
    real_estate_shift = (real_estate_to_construction - q_4q).abs()
    finance_net = _numeric(frame, "delta_net_finance_insurance")
    households_net = _numeric(frame, "delta_net_households_housing_consumer_tax")
    finance_shift = (
        reclassified_composition(finance_net, target_bucket="G")["G"] - q_4q
    ).abs()
    household_shift = (
        reclassified_composition(households_net, target_bucket="G")["G"] - q_4q
    ).abs()
    valid_flow = c_t.where(c_t > 0).replace([np.inf, -np.inf], np.nan).dropna()
    flow_p20 = float(valid_flow.quantile(0.20)) if not valid_flow.empty else float("nan")
    q_boundary = (q_positive <= 0.05) | (q_positive >= 0.95)
    low_flow = (
        (c_t > 0) & (c_t <= flow_p20)
        if np.isfinite(flow_p20)
        else pd.Series(False, index=frame.index)
    )
    low_flow_boundary = _share(q_boundary[low_flow])
    high_flow_boundary = _share(q_boundary[c_t > flow_p20])
    q_flow_floor = _safe_div(c_g, c_t.where(c_t >= flow_p20, flow_p20))
    floor_diff = (q_positive - q_flow_floor).abs()[low_flow].replace([np.inf, -np.inf], np.nan).dropna()
    floor_median = float(floor_diff.median()) if not floor_diff.empty else float("nan")
    floor_p90 = float(floor_diff.quantile(0.90)) if not floor_diff.empty else float("nan")
    floor_max = float(floor_diff.max()) if not floor_diff.empty else float("nan")
    interval_n, interval_corr = _corr(q_positive, q_4q)
    interval_mad = _median_abs_diff(q_positive, q_4q)

    stock_columns = [
        column
        for column in availability_frame.columns
        if column.startswith("stock_")
        and column not in {"stock_G", "stock_B", "stock_E", "stock_total"}
    ]
    dates = (
        pd.to_datetime(availability_frame["date"], errors="coerce")
        if "date" in availability_frame.columns
        else pd.Series(pd.NaT, index=availability_frame.index)
    )
    available_count = (
        availability_frame[stock_columns].notna().sum(axis=1)
        if stock_columns
        else pd.Series(0, index=availability_frame.index)
    )
    full_count = len(stock_columns)
    full_rows = available_count.eq(full_count) if stock_columns else pd.Series(False, index=frame.index)
    full_start = dates[full_rows].min() if full_rows.any() else pd.NaT
    selected_stock_total = (
        availability_frame[stock_columns]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, min_count=1)
        if stock_columns
        else pd.Series(np.nan, index=availability_frame.index)
    )
    official_stock_total = _numeric(availability_frame, "stock_total")
    stock_reconciliation = selected_stock_total / official_stock_total.where(lambda s: s > 0)
    full_reconciliation = stock_reconciliation[full_rows].replace([np.inf, -np.inf], np.nan).dropna()
    reconciliation_median = (
        float(full_reconciliation.median()) if not full_reconciliation.empty else float("nan")
    )
    reconciliation_min = float(full_reconciliation.min()) if not full_reconciliation.empty else float("nan")
    reconciliation_max = float(full_reconciliation.max()) if not full_reconciliation.empty else float("nan")
    mapped_domestic_stock = _numeric(availability_frame, "mapped_domestic_stock")
    mapped_stock_ratio = (
        mapped_domestic_stock / official_stock_total.where(lambda s: s > 0)
    )
    mapped_stock_ratio = mapped_stock_ratio[full_rows].replace([np.inf, -np.inf], np.nan).dropna()
    mapped_ratio_median = _median(mapped_stock_ratio)
    mapped_ratio_min = float(mapped_stock_ratio.min()) if not mapped_stock_ratio.empty else float("nan")
    mapped_ratio_max = float(mapped_stock_ratio.max()) if not mapped_stock_ratio.empty else float("nan")
    flow_pair = pd.concat(
        [
            selected_stock_total.diff().rename("selected_change"),
            official_stock_total.diff().rename("official_change"),
        ],
        axis=1,
    )
    if "common_taxonomy_delta_valid" in availability_frame.columns:
        flow_valid = availability_frame["common_taxonomy_delta_valid"]
        if flow_valid.dtype != bool:
            flow_valid = flow_valid.astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        flow_valid = dates >= pd.Timestamp("2009-09-30")
    flow_pair = (
        flow_pair.loc[full_rows & flow_valid]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    flow_n, flow_corr = _corr(flow_pair["selected_change"], flow_pair["official_change"])
    flow_gap = (flow_pair["selected_change"] - flow_pair["official_change"]).abs()
    flow_gap_median = float(flow_gap.median()) if not flow_gap.empty else float("nan")
    flow_relative_gap = (
        flow_gap
        / flow_pair["official_change"].abs().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    flow_relative_gap_median = (
        float(flow_relative_gap.median()) if not flow_relative_gap.empty else float("nan")
    )

    rows = [
        {
            "audit": "One-quarter signed-change diagnostic",
            "readout": (
                f"1Q signed N={net_n}; aggregate contractions={net_contractions}; "
                f"paired N={net_paired_n}; corr={_format_float(net_corr)}; "
                f"median |1Q difference|={_format_float(net_mad)}; "
                f"p90={_format_float(net_p90)}; outside [0,1]={net_outside}; "
                f"range=[{_format_float(net_min)}, {_format_float(net_max)}]"
            ),
            "interpretation": "This one-quarter diagnostic compares the bounded Group-G coordinate with a signed contribution that retains aggregate-contraction denominators, excluding only near-zero totals. It is not the primary four-quarter q_t and is not a bounded share.",
        },
        {
            "audit": "Aggregation-order sensitivity",
            "readout": (
                f"4Q N={aggregation_n}; corr={_format_float(aggregation_corr)}; "
                f"median |delta q|={_format_float(aggregation_mad)}; "
                f"p90={_format_float(aggregation_p90)}; max={_format_float(aggregation_max)}; "
                f"median bucket-positive={_format_float(aggregation_bucket_median)}, "
                f"series-positive={_format_float(aggregation_series_median)}"
            ),
            "interpretation": "The primary measure nets within each borrower bucket before taking the positive part and is therefore invariant to subdivisions within a fixed bucket; the series-positive measure is retained only as a legacy sensitivity.",
        },
        {
            "audit": "Quarter-of-year denominator availability",
            "readout": (
                f"zero bucket-positive denominator={zero_denominator_n}/{len(frame)}; "
                + "; ".join(quarter_denominator_parts)
                + f"; 4Q valid={q_4q_n}/{len(frame)}, "
                + f"first={q_4q_first.date() if pd.notna(q_4q_first) else 'n/a'}"
            ),
            "interpretation": "A zero quarterly denominator is reported as missing for the quarterly share but remains a zero contribution in the primary four-quarter flow-weighted share.",
        },
        {
            "audit": "Low-denominator stress",
            "readout": (
                f"p20 positive-flow threshold={_format_float(flow_p20)}; "
                f"boundary q share low-denominator={_format_pct(low_flow_boundary)}, "
                f"other={_format_pct(high_flow_boundary)}"
            ),
            "interpretation": "Small bucket-positive denominators can push the Group G share toward 0 or 1; these rows are measurement stress, not strong economic timing.",
        },
        {
            "audit": "Flow-floor sensitivity",
            "readout": (
                f"Among low-denominator rows: median |delta q|={_format_float(floor_median)}; "
                f"p90={_format_float(floor_p90)}; max={_format_float(floor_max)}"
            ),
            "interpretation": "The denominator-floor diagnostic is summarized only over rows changed by the floor.",
        },
        {
            "audit": "Four-quarter measurement interval",
            "readout": (
                f"N={interval_n}; corr={_format_float(interval_corr)}; "
                f"median |delta q|={_format_float(interval_mad)}"
            ),
            "interpretation": "Rolling four-quarter aggregation smooths quarter-level stock-difference noise in the borrower-composition measure.",
        },
        {
            "audit": "Four-quarter within-BOJ purpose-coded comparator",
            "readout": (
                f"N={fixed_n}; mean stock-change share={_format_float(fixed_stock_mean)}, "
                f"purpose share={_format_float(fixed_purpose_mean)}; "
                f"mean bias={_format_float(fixed_mean_bias)}; "
                f"median |gap|={_format_float(fixed_mad)}; "
                f"level Pearson={_format_float(fixed_corr)}, "
                f"Spearman={_format_float(fixed_spearman)}; "
                f"change N={fixed_change_n}; change Spearman={_format_float(fixed_change_spearman)}; "
                f"direction agreement={_format_pct(fixed_direction_agreement)}"
            ),
            "interpretation": "This claim-boundary audit does not support a fixed-investment-purpose interpretation of the borrower-composition measure and is not independent external validation.",
        },
        {
            "audit": "Construction-borrower reclassification",
            "readout": (
                f"4Q median |share shift| if construction is reassigned to Group G is "
                f"{_format_float(_median(construction_shift))}; "
                f"latest={_format_float(_latest(construction_shift))}"
            ),
            "interpretation": "This is a borrower-classification sensitivity. No fractional construction-purpose weight is selected or estimated.",
        },
        {
            "audit": "Real-estate borrower reclassification",
            "readout": (
                f"4Q median |share shift| if real estate is grouped with construction is "
                f"{_format_float(_median(real_estate_shift))}; "
                f"latest={_format_float(_latest(real_estate_shift))}"
            ),
            "interpretation": "The alternative recomputes bucket net changes before taking positive parts; it does not assert the purpose of real-estate borrowing.",
        },
        {
            "audit": "Finance and household borrower alternatives",
            "readout": (
                f"4Q median |share shift| if finance is reassigned to Group G={_format_float(_median(finance_shift))}; "
                f"if households are reassigned to Group G={_format_float(_median(household_shift))}"
            ),
            "interpretation": "These extreme borrower-bucket reallocations expose label sensitivity without claiming to identify loan purpose.",
        },
        {
            "audit": "Series availability and aggregate reconciliation",
            "readout": (
                f"{full_count} selected sector stocks all observed from "
                f"{full_start.date() if pd.notna(full_start) else 'n/a'}; "
                f"24-series/official stock median={_format_float(reconciliation_median, 3)}, "
                f"range=[{_format_float(reconciliation_min, 3)}, {_format_float(reconciliation_max, 3)}]; "
                f"mapped-domestic/official median={_format_float(mapped_ratio_median, 3)}, "
                f"range=[{_format_float(mapped_ratio_min, 3)}, {_format_float(mapped_ratio_max, 3)}]; "
                f"net-change N={flow_n}, corr={_format_float(flow_corr, 4)}, "
                f"median gap={_format_float(flow_gap_median, 0)} (JPY 100m), "
                f"median relative gap={_format_pct(flow_relative_gap_median)}"
            ),
            "interpretation": "Stock and net-change agreement supports selected-series accounting coverage, not loan-purpose validity. The 2009Q2 classification-break difference is excluded from the common-taxonomy flow sample.",
        },
        {
            "audit": "Refinancing, rollovers, write-offs, and reclassification",
            "readout": "Not directly observed in LA01 outstanding-loan stocks.",
            "interpretation": "These items can move stock differences without new lending; they remain measurement error in the borrower-composition proxy.",
        },
    ]
    return rows


def _primary_validation_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    availability_frame = frame.copy()
    frame = _common_availability_frame(frame)
    components = {
        bucket: _numeric(frame, f"C_{bucket}")
        for bucket in PRIMARY_BUCKETS
    }
    component_total = sum(components.values())
    reported_total = _numeric(frame, "C_t_primary")
    if not reported_total.notna().any():
        reported_total = _numeric(frame, "C_t")
    flow_identity_gap = (reported_total - component_total).abs()

    primary_stocks = {
        bucket: _numeric(frame, f"stock_primary_{bucket.lower()}")
        for bucket in PRIMARY_BUCKETS
    }
    primary_stock_sum = sum(primary_stocks.values())
    included_stock = _numeric(frame, "primary_included_stock")
    stock_identity_gap = (included_stock - primary_stock_sum).abs()
    included_net = _numeric(frame, "primary_included_net_flow")
    raw_delta = _numeric(frame, "C_t_raw_delta")
    signed_scale_gap = (included_net - raw_delta).abs()

    mapped_nfb = _numeric(frame, "stock_primary_nfb_mapped_detail")
    nfb_stock = primary_stocks["NFB"]
    overseas_stock = _numeric(frame, "stock_overseas_explicit")
    unresolved_stock = _numeric(frame, "stock_unresolved_residual")
    nfb_identity_gap = _numeric(frame, "primary_nfb_residual_identity_gap_stock").abs()
    overseas_share = overseas_stock / nfb_stock.replace(0.0, np.nan)
    mapped_nfb_share = mapped_nfb / nfb_stock.replace(0.0, np.nan)
    unresolved_share = unresolved_stock / nfb_stock.replace(0.0, np.nan)

    household_total = _numeric(frame, "stock_household_total")
    household_housing = _numeric(frame, "stock_household_housing")
    household_nonhousing = _numeric(frame, "stock_household_nonhousing")
    household_consumer = _numeric(frame, "stock_household_consumer_purpose")
    household_identity_gap = (
        household_total - household_housing - household_nonhousing
    ).abs()
    household_invalid = int(
        (
            household_total.notna()
            & household_housing.notna()
            & (household_total - household_housing < 0.0)
        ).sum()
    )
    purpose_n, purpose_corr = _corr(household_nonhousing, household_consumer)
    purpose_gap = (household_nonhousing - household_consumer).abs()

    official_stock = _numeric(frame, "stock_total")
    primary_stock_coverage = included_stock / official_stock.replace(0.0, np.nan)
    local_stock = _numeric(frame, "stock_local_governments_explicit")
    local_stock_share = local_stock / official_stock.replace(0.0, np.nan)
    explicit_scope_gap = _numeric(frame, "explicit_scope_gap_to_official_stock").abs()

    werner_gap = _numeric(frame, "werner_population_gap_stock").abs()
    muller_verner_gap = _numeric(frame, "muller_verner_population_gap_stock").abs()

    rolling = {
        bucket: components[bucket].rolling(window=4, min_periods=4).sum()
        for bucket in PRIMARY_BUCKETS
    }
    rolling_total = sum(rolling.values())
    rolling_compositions = {
        bucket: rolling[bucket] / rolling_total.replace(0.0, np.nan)
        for bucket in PRIMARY_BUCKETS
    }
    vector_sum = sum(rolling_compositions.values())
    q_t = _numeric(frame, "q_t")
    q_expected = rolling_compositions["NFB"]
    q_alias_gap = (q_t - q_expected).abs()
    valid_q = q_expected.notna()
    dates = (
        pd.to_datetime(frame["date"], errors="coerce")
        if "date" in frame.columns
        else pd.Series(pd.NaT, index=frame.index)
    )
    first_q = dates[valid_q].min() if valid_q.any() else pd.NaT

    def maximum(series: pd.Series) -> float:
        finite = pd.to_numeric(series, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        return float(finite.max()) if not finite.empty else float("nan")

    return [
        {
            "audit": "Primary same-population identity",
            "readout": (
                f"max |C_t-sum four buckets|={_format_float(maximum(flow_identity_gap), 6)}; "
                f"max |included stock-sum four stocks|={_format_float(maximum(stock_identity_gap), 6)}; "
                f"max |signed-scale gap|={_format_float(maximum(signed_scale_gap), 6)}"
            ),
            "interpretation": (
                "Allocation and scale are constructed from one included population. "
                "This is an accounting identity, not purpose validation."
            ),
        },
        {
            "audit": "Published residual NFB decomposition",
            "readout": (
                f"median mapped-NFB/NFB={_format_pct(_median(mapped_nfb_share))}; "
                f"median overseas/NFB={_format_pct(_median(overseas_share))}; "
                f"median unresolved/NFB={_format_pct(_median(unresolved_share), 3)}; "
                f"max decomposition gap={_format_float(maximum(nfb_identity_gap), 6)}"
            ),
            "interpretation": (
                "The primary NFB series follows the published Japan residual exactly. "
                "Overseas-linked and unresolved components absorbed by that residual are "
                "visible rather than silently treated as mapped domestic industries."
            ),
        },
        {
            "audit": "Household non-housing split",
            "readout": (
                f"negative residual rows={household_invalid}; "
                f"max split-identity gap={_format_float(maximum(household_identity_gap), 6)}; "
                f"consumer-purpose pair N={purpose_n}, corr={_format_float(purpose_corr, 3)}, "
                f"median absolute stock gap={_format_float(_median(purpose_gap), 0)} (JPY 100m)"
            ),
            "interpretation": (
                "Household non-housing is total household credit less housing credit. "
                "The narrower consumer-purpose series is a same-provider comparator, not "
                "independent external validation."
            ),
        },
        {
            "audit": "Explicit official-population coverage",
            "readout": (
                f"median included/official stock={_format_pct(_median(primary_stock_coverage))}; "
                f"median local-government/official={_format_pct(_median(local_stock_share))}; "
                f"max explicit-scope gap={_format_float(maximum(explicit_scope_gap), 6)}"
            ),
            "interpretation": (
                "The primary population plus the explicit local-government exclusion "
                "reconciles to the official total; coverage does not validate loan purpose."
            ),
        },
        {
            "audit": "Literature-anchored taxonomy population identity",
            "readout": (
                f"Werner max population gap={_format_float(maximum(werner_gap), 6)}; "
                f"Muller-Verner max population gap={_format_float(maximum(muller_verner_gap), 6)}"
            ),
            "interpretation": (
                "The Werner-inspired and Muller-Verner BOJ adaptations are defined in "
                "metadata and exhaust the same included population. They are reported "
                "jointly; this is not a claim of externally time-stamped preregistration."
            ),
        },
        {
            "audit": "Primary four-quarter vector availability",
            "readout": (
                f"valid q_t={int(valid_q.sum())}/{len(frame)}; "
                f"first={first_q.date() if pd.notna(first_q) else 'n/a'}; "
                f"max |vector sum-1|={_format_float(maximum((vector_sum - 1.0).abs()), 6)}; "
                f"max |q_t-NFB alias gap|={_format_float(maximum(q_alias_gap), 6)}"
            ),
            "interpretation": (
                "q_t is only the displayed NFB coordinate of the four-part, common-denominator "
                "borrower-composition vector."
            ),
        },
        {
            "audit": "Current-vintage and stock-flow boundary",
            "readout": (
                f"rows={len(availability_frame)}; refinancing, rollovers, write-offs, "
                "and reclassifications are not separately observed"
            ),
            "interpretation": (
                "The bridge remains a current-vintage stock-difference measure and does not "
                "establish incremental predictability or loan-purpose identification."
            ),
        },
    ]


def validation_rows(
    frame: pd.DataFrame,
    lambda_b: float = LAMBDA_B_DEFAULT,
) -> list[dict[str, str]]:
    if _has_primary_taxonomy(frame):
        return _primary_validation_rows(frame)
    return _legacy_validation_rows(frame, lambda_b=lambda_b)


def render_validation_table(frame: pd.DataFrame, lambda_b: float = LAMBDA_B_DEFAULT) -> str:
    all_rows = validation_rows(frame, lambda_b=lambda_b)
    by_audit = {row["audit"]: row for row in all_rows}
    audit_order = PRIMARY_VALIDATION_AUDITS if _has_primary_taxonomy(frame) else MAIN_VALIDATION_AUDITS
    rows = [by_audit[audit] for audit in audit_order]
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        (
            r"  \caption{Published-taxonomy BOJ bridge audits.}"
            if _has_primary_taxonomy(frame)
            else r"  \caption{BOJ borrower-composition audits.}"
        ),
        r"  \label{tab:boj_bridge_validation}",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}p{0.22\textwidth}p{0.26\textwidth}p{0.46\textwidth}@{}}",
        r"    \toprule",
        r"    Audit & Current readout & Interpretation \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                _validation_latex(row[key])
                for key in ("audit", "readout", "interpretation")
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_boj_bridge_validation(
    *,
    data_path: Path,
    metadata_path: Path,
    lambda_b: float = LAMBDA_B_DEFAULT,
) -> tuple[str, str, dict[str, Any]]:
    frame = pd.read_csv(data_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping_tex = render_mapping_table(frame, metadata)
    validation_tex = render_validation_table(frame, lambda_b=lambda_b)
    if _has_primary_taxonomy(frame):
        summary = {
            "primary_taxonomy_id": metadata.get(
                "primary_taxonomy_id",
                PRIMARY_TAXONOMY_ID,
            ),
            "primary_metric": (
                "four-quarter borrower-composition vector with NFB/FIN/PROP/"
                "HH_NONHOUSING coordinates formed from one denominator"
            ),
            "primary_vector_columns": [
                f"borrower_composition_{bucket}_4q"
                for bucket in PRIMARY_BUCKETS
            ],
            "display_coordinate": (
                "q_t is an alias for borrower_composition_NFB_4q = "
                "sum_4q(C_NFB) / sum_4q(C_NFB + C_FIN + C_PROP + C_HH_NONHOUSING)"
            ),
            "same_population_scale": {
                "stock": "primary_included_stock",
                "signed_flow": "C_t_raw_delta",
                "positive_bucket_flow": "C_t",
            },
            "aggregation_order": (
                "sum stocks within each reported taxonomy bucket, difference, then take "
                "the positive part"
            ),
            "taxonomy_selection": metadata.get(
                "taxonomy_selection",
                TAXONOMY_SELECTION_RULE,
            ),
            "taxonomies": metadata.get("taxonomies", TAXONOMY_METADATA),
            "legacy_appendix_columns": [
                "C_G",
                "C_B",
                "C_E",
                "legacy_C_t",
                "legacy_q_t",
            ],
            "legacy_lambda_argument_ignored": True,
            "rows": int(len(frame)),
            "primary_mapping_rows": primary_mapping_rows(frame, metadata),
            "legacy_mapping_rows": mapping_rows(frame, metadata),
            "taxonomy_robustness_rows": taxonomy_robustness_rows(metadata),
            "validation_rows": validation_rows(frame, lambda_b=lambda_b),
        }
    else:
        summary = {
            "primary_metric": (
                "four-quarter borrower-composition vector with G/B/E coordinates "
                "formed from one denominator: sum_4q(C_G + C_B + C_E)"
            ),
            "primary_vector_columns": [
                "borrower_composition_G_4q",
                "borrower_composition_B_4q",
                "borrower_composition_E_4q",
            ],
            "display_coordinate": (
                "q_t is an alias for borrower_composition_G_4q = "
                "sum_4q(C_G) / sum_4q(C_G + C_B + C_E)"
            ),
            "quarterly_diagnostic": (
                "one-quarter G/B/E borrower-composition vector with the common "
                "denominator C_G + C_B + C_E"
            ),
            "aggregation_order": (
                "sum net changes within each borrower bucket, then take the positive part"
            ),
            "legacy_lambda_argument_ignored": True,
            "rows": int(len(frame)),
            "mapping_rows": mapping_rows(frame, metadata),
            "validation_rows": validation_rows(frame, lambda_b=lambda_b),
        }
    return mapping_tex, validation_tex, summary


def write_boj_bridge_validation_outputs(root: Path) -> dict[str, Path]:
    mapping_tex, validation_tex, summary = run_boj_bridge_validation(
        data_path=root / "data" / "credit_destination_jp.csv",
        metadata_path=root / "data" / "credit_destination_jp_metadata.json",
    )
    frame = pd.read_csv(root / "data" / "credit_destination_jp.csv")
    metadata = json.loads((root / "data" / "credit_destination_jp_metadata.json").read_text(encoding="utf-8"))
    main_mapping_tex = render_main_mapping_table(frame, metadata)
    primary_mapping_tex = render_primary_mapping_table(frame, metadata)
    taxonomy_robustness_tex = render_taxonomy_robustness_table(metadata)
    tex_dir = root / "tex" / "generated"
    data_dir = root / "data"
    tex_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    main_mapping_path = tex_dir / "theory_boj_bridge_mapping_main.tex"
    primary_mapping_path = tex_dir / "theory_boj_primary_mapping.tex"
    taxonomy_robustness_path = tex_dir / "theory_boj_taxonomy_robustness.tex"
    mapping_path = tex_dir / "theory_boj_bridge_mapping.tex"
    validation_path = tex_dir / "theory_boj_bridge_validation.tex"
    summary_path = data_dir / "boj_bridge_validation_summary.json"
    main_mapping_path.write_text(main_mapping_tex, encoding="utf-8")
    primary_mapping_path.write_text(primary_mapping_tex, encoding="utf-8")
    taxonomy_robustness_path.write_text(taxonomy_robustness_tex, encoding="utf-8")
    mapping_path.write_text(mapping_tex, encoding="utf-8")
    validation_path.write_text(validation_tex, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "main_mapping_tex": main_mapping_path,
        "primary_mapping_tex": primary_mapping_path,
        "taxonomy_robustness_tex": taxonomy_robustness_path,
        "mapping_tex": mapping_path,
        "validation_tex": validation_path,
        "summary": summary_path,
    }


def write_main_boj_bridge_tables(root: Path) -> dict[str, Path]:
    frame = pd.read_csv(root / "data" / "credit_destination_jp.csv")
    metadata = json.loads(
        (root / "data" / "credit_destination_jp_metadata.json").read_text(encoding="utf-8")
    )
    tex_dir = root / "tex" / "generated"
    tex_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = tex_dir / "theory_boj_bridge_mapping_main.tex"
    primary_mapping_path = tex_dir / "theory_boj_primary_mapping.tex"
    taxonomy_robustness_path = tex_dir / "theory_boj_taxonomy_robustness.tex"
    validation_path = tex_dir / "theory_boj_bridge_validation.tex"
    mapping_path.write_text(render_main_mapping_table(frame, metadata), encoding="utf-8")
    primary_mapping_path.write_text(render_primary_mapping_table(frame, metadata), encoding="utf-8")
    taxonomy_robustness_path.write_text(
        render_taxonomy_robustness_table(metadata),
        encoding="utf-8",
    )
    validation_path.write_text(render_validation_table(frame), encoding="utf-8")
    return {
        "main_mapping_tex": mapping_path,
        "primary_mapping_tex": primary_mapping_path,
        "taxonomy_robustness_tex": taxonomy_robustness_path,
        "validation_tex": validation_path,
    }


__all__ = [
    "main_mapping_rows",
    "mapping_rows",
    "primary_mapping_rows",
    "render_main_mapping_table",
    "render_mapping_table",
    "render_primary_mapping_table",
    "render_taxonomy_robustness_table",
    "render_validation_table",
    "run_boj_bridge_validation",
    "validation_rows",
    "taxonomy_robustness_rows",
    "write_boj_bridge_validation_outputs",
    "write_main_boj_bridge_tables",
]
