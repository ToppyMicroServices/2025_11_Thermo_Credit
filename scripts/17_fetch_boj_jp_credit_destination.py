from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.boj_credit_taxonomies import (
    HOUSEHOLD_CONSUMER_STOCK_CODE,
    HOUSEHOLD_HOUSING_STOCK_CODE,
    HOUSEHOLD_TOTAL_STOCK_CODE,
    LEGACY_TAXONOMY_ID,
    MAPPING_VERSION,
    MULLER_VERNER_BUCKETS,
    MULLER_VERNER_TAXONOMY_ID,
    OFFICIAL_TOTAL_FIXED_CODE,
    OFFICIAL_TOTAL_STOCK_CODE,
    PRIMARY_BUCKETS,
    PRIMARY_TAXONOMY_ID,
    SECTOR_MAP,
    TAXONOMY_METADATA,
    TAXONOMY_SELECTION_RULE,
    WERNER_BUCKETS,
    WERNER_TAXONOMY_ID,
)

API_ENDPOINT = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"
DB_NAME = "LA01"
UNIT = "100 million yen"
COMMON_TAXONOMY_STOCK_START = pd.Timestamp("2009-06-30")
FIRST_VALID_COMMON_TAXONOMY_FLOW = pd.Timestamp("2009-09-30")


def _fixed_investment_code(stock_code: str) -> str:
    return stock_code.replace("DLLI5DS2", "DLLI5DS5", 1)


def _quarter_end(period: Any) -> pd.Timestamp:
    value = int(period)
    year = value // 100
    quarter = value % 100
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _fetch_codes(codes: list[str], *, start_date: str, end_date: str) -> dict[str, Any]:
    params = {
        "format": "json",
        "lang": "en",
        "db": DB_NAME,
        "startDate": start_date,
        "endDate": end_date,
        "code": ",".join(codes),
    }
    url = API_ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    data = urllib.request.urlopen(req, timeout=60).read()
    try:
        data = gzip.decompress(data)
    except OSError:
        pass
    payload = json.loads(data.decode("utf-8"))
    if int(payload.get("STATUS", 500)) != 200:
        raise RuntimeError(f"BOJ API request failed: {payload}")
    return payload


def _resultset_to_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metadata: dict[str, dict[str, Any]] = {}
    for result in payload.get("RESULTSET", []):
        code = str(result.get("SERIES_CODE", ""))
        values = result.get("VALUES", {})
        dates = values.get("SURVEY_DATES", [])
        observations = values.get("VALUES", [])
        if not code or not dates:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": [_quarter_end(x) for x in dates],
                    code: pd.to_numeric(pd.Series(observations), errors="coerce"),
                }
            )
        )
        metadata[code] = {
            "name": result.get("NAME_OF_TIME_SERIES", ""),
            "unit": result.get("UNIT", ""),
            "frequency": result.get("FREQUENCY", ""),
            "category": result.get("CATEGORY", ""),
            "last_update": result.get("LAST_UPDATE", ""),
        }
    if not frames:
        return pd.DataFrame(), metadata
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True), metadata


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _complete_sum(parts: Sequence[pd.Series], index: pd.Index) -> pd.Series:
    if not parts:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))


def _taxonomy_measures(
    stocks: Mapping[str, pd.Series],
    *,
    buckets: Sequence[str],
    valid: pd.Series,
    index: pd.Index,
) -> tuple[
    dict[str, pd.Series],
    dict[str, pd.Series],
    pd.Series,
    dict[str, pd.Series],
    dict[str, pd.Series],
]:
    net = {bucket: stocks[bucket].diff().where(valid) for bucket in buckets}
    positive = {bucket: net[bucket].clip(lower=0.0) for bucket in buckets}
    positive_total = _complete_sum(list(positive.values()), index)
    denominator_1q = positive_total.replace(0.0, np.nan)
    composition_1q = {
        bucket: positive[bucket] / denominator_1q
        for bucket in buckets
    }
    rolling = {
        bucket: positive[bucket].rolling(window=4, min_periods=4).sum()
        for bucket in buckets
    }
    rolling_total = _complete_sum(list(rolling.values()), index)
    composition_4q = {
        bucket: rolling[bucket] / rolling_total.replace(0.0, np.nan)
        for bucket in buckets
    }
    return net, positive, positive_total, composition_1q, composition_4q


def _construct_destination_panel(
    raw: pd.DataFrame,
    *,
    sector_map: Sequence[Mapping[str, str]] = SECTOR_MAP,
) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(raw["date"], errors="coerce")})
    sector_audit: dict[str, pd.Series] = {}
    net_parts: dict[str, list[pd.Series]] = {"G": [], "B": [], "E": [], "U": []}
    series_positive_parts: dict[str, list[pd.Series]] = {"G": [], "B": [], "E": [], "U": []}
    fixed_parts: dict[str, list[pd.Series]] = {"G": [], "B": [], "E": []}
    stock_parts: dict[str, list[pd.Series]] = {"G": [], "B": [], "E": []}

    stock_by_sector: dict[str, pd.Series] = {}
    for row in sector_map:
        bucket = row["bucket"]
        stock_code = row["stock_code"]
        fixed_code = _fixed_investment_code(stock_code)
        stock = _numeric_series(raw, stock_code)
        stock_by_sector[row["sector"]] = stock
        net_delta = stock.diff()
        positive_delta = net_delta.clip(lower=0.0)
        fixed_new = _numeric_series(raw, fixed_code).clip(lower=0.0)
        sector_audit[f"stock_{row['sector']}"] = stock
        sector_audit[f"delta_net_{row['sector']}"] = net_delta
        sector_audit[f"delta_positive_{row['sector']}"] = positive_delta
        sector_audit[f"fixed_investment_new_{row['sector']}"] = fixed_new
        if bucket in net_parts:
            net_parts[bucket].append(net_delta)
            series_positive_parts[bucket].append(positive_delta)
        if bucket in fixed_parts:
            fixed_parts[bucket].append(fixed_new)
            stock_parts[bucket].append(stock)

    net_buckets = {
        bucket: _complete_sum(net_parts[bucket], raw.index)
        for bucket in ("G", "B", "E")
    }
    bucket_positive = {
        bucket: net_buckets[bucket].clip(lower=0.0)
        for bucket in ("G", "B", "E")
    }
    series_positive_buckets = {
        bucket: _complete_sum(series_positive_parts[bucket], raw.index)
        for bucket in ("G", "B", "E")
    }
    fixed_buckets = {
        bucket: _complete_sum(fixed_parts[bucket], raw.index)
        for bucket in ("G", "B", "E")
    }
    stock_buckets = {
        bucket: _complete_sum(stock_parts[bucket], raw.index)
        for bucket in ("G", "B", "E")
    }

    unclassified_net = _complete_sum(net_parts["U"], raw.index)
    unclassified_positive = unclassified_net.clip(lower=0.0)
    unclassified_series_positive = _complete_sum(series_positive_parts["U"], raw.index)
    classified_net = sum(net_buckets.values())
    classified_positive = sum(bucket_positive.values())
    classified_series_positive = sum(series_positive_buckets.values())
    all_positive = classified_positive + unclassified_positive
    all_series_positive = classified_series_positive + unclassified_series_positive
    all_net = classified_net + unclassified_net

    dates = pd.to_datetime(raw["date"], errors="coerce")
    common_taxonomy_delta_valid = dates >= FIRST_VALID_COMMON_TAXONOMY_FLOW
    central_flows = {
        "C_G": bucket_positive["G"],
        "C_B": bucket_positive["B"],
        "C_E": bucket_positive["E"],
        "C_G_net": net_buckets["G"],
        "C_B_net": net_buckets["B"],
        "C_E_net": net_buckets["E"],
        "classified_positive_flow": classified_positive,
        "unclassified_positive_flow": unclassified_positive,
        "total_positive_flow": all_positive,
        "classified_net_flow": classified_net,
        "unclassified_net_flow": unclassified_net,
        "total_net_flow_observed": all_net,
        "C_G_series_positive": series_positive_buckets["G"],
        "C_B_series_positive": series_positive_buckets["B"],
        "C_E_series_positive": series_positive_buckets["E"],
        "classified_series_positive_flow": classified_series_positive,
        "unclassified_series_positive_flow": unclassified_series_positive,
        "total_series_positive_flow": all_series_positive,
    }
    central_flows = {
        column: series.where(common_taxonomy_delta_valid)
        for column, series in central_flows.items()
    }

    bucket_positive_total = (
        central_flows["C_G"]
        + central_flows["C_B"]
        + central_flows["C_E"]
    )
    c_t = bucket_positive_total.replace(0.0, np.nan)
    composition_1q = {
        bucket: central_flows[f"C_{bucket}"] / c_t
        for bucket in ("G", "B", "E")
    }
    rolling_components = {
        bucket: central_flows[f"C_{bucket}"].rolling(window=4, min_periods=4).sum()
        for bucket in ("G", "B", "E")
    }
    rolling_total = sum(rolling_components.values())
    composition_4q = {
        bucket: rolling_components[bucket] / rolling_total.replace(0.0, np.nan)
        for bucket in ("G", "B", "E")
    }
    total_stock = _numeric_series(raw, OFFICIAL_TOTAL_STOCK_CODE)
    total_raw_delta = total_stock.diff()
    total_fixed_new = _numeric_series(raw, OFFICIAL_TOTAL_FIXED_CODE)
    legacy_mapped_domestic_stock = stock_buckets["G"] + stock_buckets["B"] + stock_buckets["E"]
    legacy_coverage = np.where(
        central_flows["total_positive_flow"] > 0,
        central_flows["classified_positive_flow"] / central_flows["total_positive_flow"],
        np.nan,
    )

    summary_columns = {
        "C_t": c_t,
        "C_t_raw_delta": total_raw_delta,
        **central_flows,
        **{
            f"borrower_composition_{bucket}_1q": composition_1q[bucket]
            for bucket in ("G", "B", "E")
        },
        **{
            f"borrower_composition_{bucket}_4q": composition_4q[bucket]
            for bucket in ("G", "B", "E")
        },
        "q_t": composition_4q["G"],
        # Legacy aliases retained for downstream consumers of earlier archives.
        "operating_borrower_share_1q": composition_1q["G"],
        "operating_borrower_share_4q": composition_4q["G"],
        "operating_borrower_share": composition_4q["G"],
        "mapped_borrower_coverage_observed": legacy_coverage,
        "destination_coverage_observed": legacy_coverage,
        "common_taxonomy_delta_valid": common_taxonomy_delta_valid,
        "stock_G": stock_buckets["G"],
        "stock_B": stock_buckets["B"],
        "stock_E": stock_buckets["E"],
        "mapped_domestic_stock": legacy_mapped_domestic_stock,
        "stock_total": total_stock,
        "fixed_investment_new_G": fixed_buckets["G"],
        "fixed_investment_new_B": fixed_buckets["B"],
        "fixed_investment_new_E": fixed_buckets["E"],
        "fixed_investment_new_total": total_fixed_new,
    }

    has_published_taxonomy = any(
        row.get("primary_bucket") is not None
        for row in sector_map
    )
    if has_published_taxonomy:
        primary_sector_rows = {
            bucket: [
                row
                for row in sector_map
                if row.get("primary_bucket") == bucket
            ]
            for bucket in ("NFB", "FIN", "PROP")
        }
        mapped_sector_stocks = {
            bucket: _complete_sum(
                [stock_by_sector[row["sector"]] for row in primary_sector_rows[bucket]],
                raw.index,
            )
            for bucket in ("NFB", "FIN", "PROP")
        }
        household_total = _numeric_series(raw, HOUSEHOLD_TOTAL_STOCK_CODE)
        household_housing = _numeric_series(raw, HOUSEHOLD_HOUSING_STOCK_CODE)
        household_consumer = _numeric_series(raw, HOUSEHOLD_CONSUMER_STOCK_CODE)
        household_nonhousing_raw = household_total - household_housing
        household_nonhousing = household_nonhousing_raw.where(household_nonhousing_raw >= 0.0)
        local_government_stock = stock_by_sector.get(
            "local_governments",
            pd.Series(np.nan, index=raw.index, dtype=float),
        )
        overseas_stock = stock_by_sector.get(
            "overseas_yen_and_transferred_loans",
            pd.Series(np.nan, index=raw.index, dtype=float),
        )
        other_organizations_stock = stock_by_sector.get(
            "other_organizations",
            pd.Series(np.nan, index=raw.index, dtype=float),
        )
        published_nfb_stock = (
            total_stock
            - mapped_sector_stocks["FIN"]
            - mapped_sector_stocks["PROP"]
            - local_government_stock
            - household_total
        )
        unresolved_stock = (
            published_nfb_stock
            - mapped_sector_stocks["NFB"]
            - overseas_stock
        )
        primary_stocks = {
            "NFB": published_nfb_stock,
            "FIN": mapped_sector_stocks["FIN"],
            "PROP": mapped_sector_stocks["PROP"] + household_housing,
            "HH_NONHOUSING": household_nonhousing,
        }
        primary_complete = pd.concat(primary_stocks, axis=1).notna().all(axis=1)
        primary_valid = (
            common_taxonomy_delta_valid
            & primary_complete
            & primary_complete.shift(1, fill_value=False)
        )
        (
            primary_net,
            primary_positive,
            primary_positive_total,
            primary_composition_1q,
            primary_composition_4q,
        ) = _taxonomy_measures(
            primary_stocks,
            buckets=PRIMARY_BUCKETS,
            valid=primary_valid,
            index=raw.index,
        )
        primary_included_stock = _complete_sum(list(primary_stocks.values()), raw.index)
        primary_included_net = primary_included_stock.diff().where(primary_valid)

        selected_sector_stock = _complete_sum(list(stock_by_sector.values()), raw.index)
        unresolved_aggregate_gap = total_stock - selected_sector_stock
        disclosed_component_stocks = {
            "local_governments": local_government_stock,
            "overseas": overseas_stock,
            "unresolved": unresolved_stock,
        }
        disclosed_component_net = {
            key: stock.diff().where(common_taxonomy_delta_valid)
            for key, stock in disclosed_component_stocks.items()
        }
        disclosed_component_positive = {
            key: value.clip(lower=0.0)
            for key, value in disclosed_component_net.items()
        }
        explicit_scope_positive_total = _complete_sum(
            [primary_positive_total, disclosed_component_positive["local_governments"]],
            raw.index,
        )
        primary_flow_coverage = primary_positive_total / explicit_scope_positive_total.replace(
            0.0,
            np.nan,
        )
        primary_stock_coverage = primary_included_stock / total_stock.replace(0.0, np.nan)
        explicit_scope_stock = _complete_sum(
            [primary_included_stock, local_government_stock],
            raw.index,
        )

        construction_stock = stock_by_sector["construction"]
        werner_stocks = {
            "FCP": _complete_sum(
                [
                    stock_by_sector[row["sector"]]
                    for row in sector_map
                    if row.get("werner_bucket") == "FCP"
                ],
                raw.index,
            ),
        }
        werner_stocks["COMPLEMENT"] = primary_included_stock - werner_stocks["FCP"]
        (
            werner_net,
            werner_positive,
            werner_positive_total,
            werner_composition_1q,
            werner_composition_4q,
        ) = _taxonomy_measures(
            werner_stocks,
            buckets=WERNER_BUCKETS,
            valid=primary_valid,
            index=raw.index,
        )

        muller_verner_stocks = {
            bucket: _complete_sum(
                [
                    stock_by_sector[row["sector"]]
                    for row in sector_map
                    if row.get("muller_verner_bucket") == bucket
                ],
                raw.index,
            )
            for bucket in ("TRADABLE", "NONTRADABLE", "OTHER_NFB", "FIN")
        }
        muller_verner_stocks["HH"] = household_total
        muller_verner_stocks["UNRESOLVED"] = (
            primary_included_stock
            - _complete_sum(list(muller_verner_stocks.values()), raw.index)
        )
        (
            muller_verner_net,
            muller_verner_positive,
            muller_verner_positive_total,
            muller_verner_composition_1q,
            muller_verner_composition_4q,
        ) = _taxonomy_measures(
            muller_verner_stocks,
            buckets=MULLER_VERNER_BUCKETS,
            valid=primary_valid,
            index=raw.index,
        )

        legacy_c_t = summary_columns["C_t"]
        legacy_q_t = summary_columns["q_t"]
        summary_columns.update(
            {
                "C_t": primary_positive_total.replace(0.0, np.nan),
                "C_t_primary": primary_positive_total.replace(0.0, np.nan),
                "C_t_raw_delta": primary_included_net,
                "primary_positive_flow": primary_positive_total,
                "primary_included_net_flow": primary_included_net,
                **{
                    f"C_{bucket}": primary_positive[bucket]
                    for bucket in PRIMARY_BUCKETS
                },
                **{
                    f"C_{bucket}_net": primary_net[bucket]
                    for bucket in PRIMARY_BUCKETS
                },
                **{
                    f"borrower_composition_{bucket}_1q": primary_composition_1q[bucket]
                    for bucket in PRIMARY_BUCKETS
                },
                **{
                    f"borrower_composition_{bucket}_4q": primary_composition_4q[bucket]
                    for bucket in PRIMARY_BUCKETS
                },
                "q_t": primary_composition_4q["NFB"],
                "q_t_primary": primary_composition_4q["NFB"],
                "primary_taxonomy_delta_valid": primary_valid,
                "primary_included_stock": primary_included_stock,
                "primary_stock_coverage_official": primary_stock_coverage,
                "primary_flow_coverage_observed": primary_flow_coverage,
                "mapped_borrower_coverage_observed": primary_flow_coverage,
                "destination_coverage_observed": primary_flow_coverage,
                **{
                    f"stock_primary_{bucket.lower()}": primary_stocks[bucket]
                    for bucket in PRIMARY_BUCKETS
                },
                "stock_household_total": household_total,
                "stock_household_housing": household_housing,
                "stock_household_nonhousing": household_nonhousing,
                "stock_household_consumer_purpose": household_consumer,
                "household_nonhousing_raw_stock": household_nonhousing_raw,
                "household_split_identity_gap_stock": (
                    household_total - household_housing - household_nonhousing
                ),
                "household_nonhousing_minus_consumer_purpose_stock": (
                    household_nonhousing - household_consumer
                ),
                "household_nonhousing_valid": household_nonhousing.notna(),
                "stock_local_governments_explicit": local_government_stock,
                "stock_overseas_explicit": overseas_stock,
                "stock_other_organizations_unresolved": other_organizations_stock,
                "stock_unresolved_aggregate_gap": unresolved_aggregate_gap,
                "stock_unresolved_residual": unresolved_stock,
                "stock_primary_nfb_mapped_detail": mapped_sector_stocks["NFB"],
                "primary_nfb_residual_identity_gap_stock": (
                    published_nfb_stock
                    - mapped_sector_stocks["NFB"]
                    - overseas_stock
                    - unresolved_stock
                ),
                "primary_nfb_mapped_detail_coverage": (
                    mapped_sector_stocks["NFB"] / published_nfb_stock.replace(0.0, np.nan)
                ),
                "local_governments_net_flow": disclosed_component_net["local_governments"],
                "overseas_net_flow": disclosed_component_net["overseas"],
                "unresolved_residual_net_flow": disclosed_component_net["unresolved"],
                "local_governments_positive_flow": disclosed_component_positive[
                    "local_governments"
                ],
                "overseas_positive_flow": disclosed_component_positive["overseas"],
                "unresolved_residual_positive_flow": disclosed_component_positive[
                    "unresolved"
                ],
                "explicit_scope_positive_flow": explicit_scope_positive_total,
                "explicit_scope_stock": explicit_scope_stock,
                "explicit_scope_gap_to_official_stock": total_stock - explicit_scope_stock,
                "construction_share_of_primary_nfb_stock": (
                    construction_stock / primary_stocks["NFB"].replace(0.0, np.nan)
                ),
                "legacy_C_t": legacy_c_t,
                "legacy_C_t_raw_delta_official": total_raw_delta,
                "legacy_q_t": legacy_q_t,
                "legacy_mapped_domestic_stock": legacy_mapped_domestic_stock,
                "legacy_destination_coverage_observed": legacy_coverage,
                **{
                    f"C_WERNER_{bucket}": werner_positive[bucket]
                    for bucket in WERNER_BUCKETS
                },
                **{
                    f"C_WERNER_{bucket}_net": werner_net[bucket]
                    for bucket in WERNER_BUCKETS
                },
                **{
                    f"werner_composition_{bucket}_1q": werner_composition_1q[bucket]
                    for bucket in WERNER_BUCKETS
                },
                **{
                    f"werner_composition_{bucket}_4q": werner_composition_4q[bucket]
                    for bucket in WERNER_BUCKETS
                },
                "werner_positive_flow": werner_positive_total,
                "werner_population_gap_stock": (
                    primary_included_stock - _complete_sum(list(werner_stocks.values()), raw.index)
                ),
                **{
                    f"C_MV_{bucket}": muller_verner_positive[bucket]
                    for bucket in MULLER_VERNER_BUCKETS
                },
                **{
                    f"C_MV_{bucket}_net": muller_verner_net[bucket]
                    for bucket in MULLER_VERNER_BUCKETS
                },
                **{
                    f"muller_verner_composition_{bucket}_1q": (
                        muller_verner_composition_1q[bucket]
                    )
                    for bucket in MULLER_VERNER_BUCKETS
                },
                **{
                    f"muller_verner_composition_{bucket}_4q": (
                        muller_verner_composition_4q[bucket]
                    )
                    for bucket in MULLER_VERNER_BUCKETS
                },
                "muller_verner_positive_flow": muller_verner_positive_total,
                "muller_verner_population_gap_stock": (
                    primary_included_stock
                    - _complete_sum(list(muller_verner_stocks.values()), raw.index)
                ),
            }
        )

    out = pd.concat([out, pd.DataFrame(summary_columns), pd.DataFrame(sector_audit)], axis=1)
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def build_jp_credit_destination(
    *,
    start_date: str = "197701",
    end_date: str = "202601",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stock_codes = [row["stock_code"] for row in SECTOR_MAP]
    fixed_codes = [_fixed_investment_code(code) for code in stock_codes]
    purpose_stock_codes = [
        HOUSEHOLD_HOUSING_STOCK_CODE,
        HOUSEHOLD_CONSUMER_STOCK_CODE,
    ]
    total_codes = [OFFICIAL_TOTAL_STOCK_CODE, OFFICIAL_TOTAL_FIXED_CODE]
    all_codes = sorted(set(stock_codes + fixed_codes + purpose_stock_codes + total_codes))
    payload = _fetch_codes(all_codes, start_date=start_date, end_date=end_date)
    raw, series_metadata = _resultset_to_frame(payload)
    if raw.empty:
        raise RuntimeError("BOJ API returned no usable observations.")

    out = _construct_destination_panel(raw)

    metadata = {
        "source": "Bank of Japan Time-Series Data Search API",
        "api_endpoint": API_ENDPOINT,
        "db": DB_NAME,
        "unit": UNIT,
        "mapping_version": MAPPING_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "construction": (
            "The primary Bezemer-Samarina-Zhang Japan crosswalk has four buckets: "
            "non-financial business (including construction), financial business, "
            "property/mortgage (real-estate borrowers plus purpose-coded household housing "
            "loans), and household non-housing (household total less housing). C_t is the "
            "sum of positive quarterly net changes after stocks are aggregated within those "
            "four buckets; C_t_raw_delta is the signed change in exactly the same included "
            "stock population. The primary four-quarter composition vector uses the common "
            "denominator sum_4q(C_NFB+C_FIN+C_PROP+C_HH_NONHOUSING), and q_t is its NFB "
            "coordinate. Household non-housing is not labelled consumer credit; the BOJ "
            "consumer-purpose series is an audit comparator. Local governments, overseas-linked "
            "loans, and the unresolved residual are reported explicitly. Werner (1997) and "
            "Muller-Verner (2024) mappings are literature-anchored robustness taxonomies "
            "reported jointly over the same included population; the current-sample declaration "
            "is not an externally time-stamped preregistration. Legacy G/B/E columns remain "
            "for Appendix and archive compatibility. "
            "The 2009Q2 classification-break difference is excluded; the first valid "
            "common-taxonomy flow is 2009Q3."
        ),
        "primary_population_note": (
            "primary_included_stock equals the official total for domestically licensed "
            "banks less local-government loans. Borrower geography is not restricted to "
            "domestic borrowers: the published residual NFB formula absorbs the explicitly "
            "reported overseas-linked series."
        ),
        "primary_taxonomy_id": PRIMARY_TAXONOMY_ID,
        "primary_vector_columns": [
            "borrower_composition_NFB_4q",
            "borrower_composition_FIN_4q",
            "borrower_composition_PROP_4q",
            "borrower_composition_HH_NONHOUSING_4q",
        ],
        "primary_scale_stock_column": "primary_included_stock",
        "primary_scale_flow_column": "C_t",
        "primary_signed_scale_flow_column": "C_t_raw_delta",
        "display_coordinate": "q_t = borrower_composition_NFB_4q",
        "taxonomy_selection": TAXONOMY_SELECTION_RULE,
        "taxonomies": TAXONOMY_METADATA,
        "robustness_taxonomy_ids": [
            WERNER_TAXONOMY_ID,
            MULLER_VERNER_TAXONOMY_ID,
        ],
        "legacy_taxonomy_id": LEGACY_TAXONOMY_ID,
        "legacy_aliases": {
            "legacy_C_t": "sum(C_G, C_B, C_E)",
            "legacy_q_t": "borrower_composition_G_4q",
            "mapped_domestic_stock": (
                "legacy_mapped_domestic_stock; retained for archive compatibility and "
                "not the primary Bezemer population scale"
            ),
            "operating_borrower_share_1q": "borrower_composition_G_1q",
            "operating_borrower_share_4q": "borrower_composition_G_4q",
            "operating_borrower_share": "borrower_composition_G_4q",
        },
        "common_taxonomy_stock_start": COMMON_TAXONOMY_STOCK_START.date().isoformat(),
        "first_valid_common_taxonomy_flow": FIRST_VALID_COMMON_TAXONOMY_FLOW.date().isoformat(),
        "bucket_mapping": SECTOR_MAP,
        "series_metadata": series_metadata,
    }
    return out, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch BOJ JP credit data for the published four-bucket crosswalk."
    )
    parser.add_argument("--start-date", default="197701", help="BOJ quarterly start date, YYYYQQ.")
    parser.add_argument("--end-date", default="202601", help="BOJ quarterly end date, YYYYQQ.")
    parser.add_argument("--output", default="data/credit_destination_jp.csv")
    parser.add_argument("--metadata-output", default="data/credit_destination_jp_metadata.json")
    args = parser.parse_args(argv)

    panel, metadata = build_jp_credit_destination(start_date=args.start_date, end_date=args.end_date)
    out_path = ROOT / args.output
    meta_path = ROOT / args.metadata_output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path} ({len(panel)} rows)")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
