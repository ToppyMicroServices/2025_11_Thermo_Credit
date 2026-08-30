from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


METADATA_RELATIVE_PATH = Path("data/external_validation/metadata.json")
BOJ_RELATIVE_PATH = Path("data/credit_destination_jp.csv")
SUMMARY_RELATIVE_PATH = Path("data/external_validation_summary.json")
TEX_RELATIVE_PATH = Path("tex/generated/theory_external_partial_validation.tex")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(root: Path) -> dict[str, Any]:
    path = root / METADATA_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def verify_snapshot_checksums(
    root: Path,
    metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    sources = metadata.get("sources", {})
    if not isinstance(sources, Mapping):
        raise ValueError("External-validation metadata must contain a sources object.")
    for source_id, source in sources.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"Metadata for {source_id} must be an object.")
        relative = Path(str(source["snapshot_file"]))
        path = root / relative
        expected = str(source["snapshot_sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Snapshot checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
        results[str(source_id)] = {
            "path": relative.as_posix(),
            "sha256": actual,
            "verified": True,
            "official_file_url": str(source["official_file_url"]),
            "official_file_sha256_at_retrieval": str(
                source["official_file_sha256_at_retrieval"]
            ),
            "raw_file_storage": str(source["raw_file_storage"]),
        }
    return results


def _correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return float("nan")
    ranked = pair.rank(method="average")
    return _correlation(ranked.iloc[:, 0], ranked.iloc[:, 1])


def _iso_quarter(value: pd.Timestamp) -> str:
    return f"{value.year}Q{value.quarter}"


def validate_mof_manufacturing(
    mof: pd.DataFrame,
    boj: pd.DataFrame,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    required_mof = {"date", "total_bank_borrowing_100m_yen"}
    required_boj = {"date", "stock_manufacturing"}
    if not required_mof.issubset(mof.columns):
        missing = sorted(required_mof - set(mof.columns))
        raise ValueError(f"MOF snapshot is missing columns: {missing}")
    if not required_boj.issubset(boj.columns):
        missing = sorted(required_boj - set(boj.columns))
        raise ValueError(f"BOJ panel is missing columns: {missing}")

    mof_dates = pd.to_datetime(mof["date"].to_numpy(), errors="coerce")
    boj_dates = pd.to_datetime(boj["date"].to_numpy(), errors="coerce")
    mof_order = np.argsort(mof_dates.to_numpy())
    boj_order = np.argsort(boj_dates.to_numpy())
    mof_stock = pd.Series(
        pd.to_numeric(
            mof["total_bank_borrowing_100m_yen"].to_numpy(), errors="coerce"
        )[mof_order],
        dtype=float,
    )
    boj_stock = pd.Series(
        pd.to_numeric(boj["stock_manufacturing"].to_numpy(), errors="coerce")[
            boj_order
        ],
        dtype=float,
    )
    mof_work = pd.DataFrame(
        {
            "date": mof_dates.to_numpy()[mof_order],
            "mof_change_4q": mof_stock.diff(4).to_numpy(),
        }
    )
    boj_work = pd.DataFrame(
        {
            "date": boj_dates.to_numpy()[boj_order],
            "boj_change_4q": boj_stock.diff(4).to_numpy(),
        }
    )

    common = mof_work[["date", "mof_change_4q"]].merge(
        boj_work[["date", "boj_change_4q"]],
        on="date",
        how="inner",
        validate="one_to_one",
    )
    common = common.replace([np.inf, -np.inf], np.nan).dropna()
    if common.empty:
        raise ValueError("MOF and BOJ manufacturing series have no common 4Q changes.")

    direction_agreement = float(
        (
            np.sign(common["mof_change_4q"])
            == np.sign(common["boj_change_4q"])
        ).mean()
    )
    first = pd.Timestamp(common["date"].min())
    last = pd.Timestamp(common["date"].max())
    return {
        "status": "computed",
        "validation_type": str(source_metadata["validation_type"]),
        "supported_primary_branch": str(source_metadata["supported_primary_branch"]),
        "purpose_status": str(source_metadata["purpose_status"]),
        "estimand": str(source_metadata["estimand"]),
        "measurement_interval": "four-quarter change",
        "common_period": f"{_iso_quarter(first)}-{_iso_quarter(last)}",
        "n_common_changes": int(len(common)),
        "pearson_correlation": _correlation(
            common["mof_change_4q"], common["boj_change_4q"]
        ),
        "spearman_correlation": _spearman(
            common["mof_change_4q"], common["boj_change_4q"]
        ),
        "direction_agreement": direction_agreement,
        "mof_latest_change_100m_yen": round(
            float(common["mof_change_4q"].iloc[-1]), 2
        ),
        "boj_latest_change_100m_yen": round(
            float(common["boj_change_4q"].iloc[-1]), 2
        ),
        "population": str(source_metadata["population"]),
        "frequency": str(source_metadata["frequency"]),
        "sample_change_control": str(source_metadata["sample_change_control"]),
        "does_not_validate": list(source_metadata["does_not_validate"]),
        "official_file_url": str(source_metadata["official_file_url"]),
    }


def validate_mlit_housing(
    mlit: pd.DataFrame,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"series_id", "value", "reference_period"}
    if not required.issubset(mlit.columns):
        missing = sorted(required - set(mlit.columns))
        raise ValueError(f"MLIT snapshot is missing columns: {missing}")
    values = (
        mlit.assign(value=pd.to_numeric(mlit["value"], errors="coerce"))
        .set_index("series_id")["value"]
        .to_dict()
    )
    needed = {
        "new_housing_purpose_share",
        "existing_housing_purpose_share",
        "refinancing_purpose_share",
        "apartment_loan_new_lending",
        "apartment_loan_outstanding",
    }
    missing_series = sorted(needed - set(values))
    if missing_series:
        raise ValueError(f"MLIT snapshot is missing series: {missing_series}")

    acquisition_share = round(
        float(
            values["new_housing_purpose_share"]
            + values["existing_housing_purpose_share"]
        ),
        3,
    )
    classified_share_sum = round(
        float(acquisition_share + values["refinancing_purpose_share"]),
        3,
    )
    reference_periods = sorted(set(mlit["reference_period"].dropna().astype(str)))
    return {
        "status": "published_cross_section",
        "validation_type": str(source_metadata["validation_type"]),
        "supported_primary_branch": str(source_metadata["supported_primary_branch"]),
        "purpose_status": str(source_metadata["purpose_status"]),
        "estimand": str(source_metadata["estimand"]),
        "reference_period": ", ".join(reference_periods),
        "new_housing_purpose_share": float(values["new_housing_purpose_share"]),
        "existing_housing_purpose_share": float(
            values["existing_housing_purpose_share"]
        ),
        "acquisition_purpose_share": acquisition_share,
        "refinancing_purpose_share": float(values["refinancing_purpose_share"]),
        "classified_share_sum": classified_share_sum,
        "classified_share_rounding_gap": round(1.0 - classified_share_sum, 3),
        "apartment_loan_new_lending_100m_yen": float(
            values["apartment_loan_new_lending"]
        ),
        "apartment_loan_outstanding_100m_yen": float(
            values["apartment_loan_outstanding"]
        ),
        "population": str(source_metadata["population"]),
        "frequency": str(source_metadata["frequency"]),
        "does_not_validate": list(source_metadata["does_not_validate"]),
        "official_file_url": str(source_metadata["official_file_url"]),
    }


def _format_float(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _format_pct(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.{digits}f}%"


def validation_rows(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    mof = summary["mof_manufacturing"]
    mlit = summary["mlit_private_housing"]
    return [
        {
            "source": "MOF corporate survey",
            "branch": "NFB: manufacturing",
            "estimand": "Correlation of debtor-side and BOJ lender-side 4Q changes in manufacturing borrowing stocks.",
            "period_readout": (
                f"{mof['common_period']}; N={mof['n_common_changes']}; "
                f"Pearson={_format_float(mof['pearson_correlation'])}; "
                f"Spearman={_format_float(mof['spearman_correlation'])}; "
                f"direction={_format_pct(mof['direction_agreement'])}"
            ),
            "boundary": (
                "Borrower-side convergent evidence only; populations and lender "
                "coverage differ, and no loan purpose is observed."
            ),
        },
        {
            "source": "MLIT private housing-loan survey",
            "branch": "PROP: household housing",
            "estimand": (
                "Published purpose composition of individual housing-loan "
                "originations and apartment-loan amounts."
            ),
            "period_readout": (
                f"{mlit['reference_period']}; acquisition="
                f"{_format_pct(mlit['acquisition_purpose_share'])}; "
                f"refinancing={_format_pct(mlit['refinancing_purpose_share'])}; "
                "apartment originations=JPY "
                f"{mlit['apartment_loan_new_lending_100m_yen'] / 10000.0:.2f}tn"
            ),
            "boundary": (
                "Direct purpose only for the annual housing branch; it does not "
                "validate corporate real-estate loans or quarterly BOJ stock changes."
            ),
        },
    ]


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def render_validation_table(summary: Mapping[str, Any]) -> str:
    rows = validation_rows(summary)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Partial external validation of published-bucket branches.}",
        r"  \label{tab:external_partial_validation}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}p{0.15\textwidth}p{0.13\textwidth}p{0.25\textwidth}p{0.22\textwidth}p{0.25\textwidth}@{}}",
        r"    \toprule",
        r"    External source & Primary branch & Estimand & Period and readout & Claim boundary \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                _latex_escape(row[key])
                for key in (
                    "source",
                    "branch",
                    "estimand",
                    "period_readout",
                    "boundary",
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"  \par\smallskip\raggedright\footnotesize MOF uses four-quarter differences to reduce sensitivity to its annual April--June sample replacement; this does not make the two credit populations identical. MLIT is annual origination evidence and is therefore not correlated with the quarterly stock-change bridge.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_external_validation(root: Path) -> dict[str, Any]:
    metadata = load_metadata(root)
    sources = metadata["sources"]
    checksum_results = verify_snapshot_checksums(root, metadata)
    mof = pd.read_csv(root / sources["mof_manufacturing"]["snapshot_file"])
    mlit = pd.read_csv(root / sources["mlit_private_housing"]["snapshot_file"])
    boj = pd.read_csv(root / BOJ_RELATIVE_PATH)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "claim_boundary": (
            "MOF supplies borrower-side convergent evidence for the manufacturing "
            "branch of NFB, not purpose validation. MLIT supplies direct-purpose "
            "evidence only for the housing branch of PROP. Neither validates the "
            "full four-bucket composition."
        ),
        "retrieved_on": metadata["retrieved_on"],
        "snapshot_verification": checksum_results,
        "mof_manufacturing": validate_mof_manufacturing(
            mof, boj, sources["mof_manufacturing"]
        ),
        "mlit_private_housing": validate_mlit_housing(
            mlit, sources["mlit_private_housing"]
        ),
    }
    summary["table_rows"] = validation_rows(summary)
    return summary


def write_external_validation_outputs(root: Path) -> dict[str, Path]:
    summary = run_external_validation(root)
    summary_path = root / SUMMARY_RELATIVE_PATH
    tex_path = root / TEX_RELATIVE_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tex_path.write_text(render_validation_table(summary), encoding="utf-8")
    return {"summary": summary_path, "tex": tex_path}


__all__ = [
    "load_metadata",
    "render_validation_table",
    "run_external_validation",
    "sha256_file",
    "validate_mlit_housing",
    "validate_mof_manufacturing",
    "validation_rows",
    "verify_snapshot_checksums",
    "write_external_validation_outputs",
]
