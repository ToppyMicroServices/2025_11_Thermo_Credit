from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.loop_area import loop_area_null_distribution, loop_window_metrics


REGION_PANELS = {
    "jp": ("Japan (JP)", "indicators_realtime.csv"),
    "eu": ("Euro Area (EU)", "indicators_eu_realtime.csv"),
    "us": ("United States (US)", "indicators_us_realtime.csv"),
}

NULL_METHOD_LABELS = {
    "block_shuffle": "block shuffle",
    "phase_randomization": "phase randomization",
    "ar_surrogate": "AR(1) surrogate",
    "event_date_permutation": "event-date permutation",
    "placebo_periods": "placebo periods",
}

SEGMENTATION_WINDOWS = (8, 12, 16)


def _stable_seed(*parts: Any, base_seed: int = 9_241) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(base_seed)) % (2**32 - 1)


def _clean_frame(frame: pd.DataFrame, *, date_col: str = "date", p_col: str = "p_C", v_col: str = "V_C") -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[date_col, p_col, v_col])
    missing = {date_col, p_col, v_col} - set(frame.columns)
    if missing:
        raise ValueError(f"loop null test frame is missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            date_col: pd.to_datetime(frame[date_col], errors="coerce"),
            p_col: pd.to_numeric(frame[p_col], errors="coerce"),
            v_col: pd.to_numeric(frame[v_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=[date_col, p_col, v_col]).sort_values(date_col).reset_index(drop=True)
    return out


def _as_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.date().isoformat()


def _region_tokens(value: Any) -> set[str]:
    return {part.strip().lower() for part in str(value or "").split(",") if part.strip()}


def load_event_windows(events_path: Path, *, region_key: str) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    events = pd.read_csv(events_path)
    out: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        regions = _region_tokens(row.get("regions", ""))
        if regions and "all" not in regions and region_key not in regions:
            continue
        start = pd.to_datetime(row.get("start_date"), errors="coerce")
        end = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        out.append(
            {
                "key": row.get("key", ""),
                "label": row.get("label", row.get("key", "")),
                "category": row.get("category", ""),
                "start": start,
                "end": end,
                "regions": sorted(regions),
            }
        )
    return out


def _overlaps(start: pd.Timestamp, end: pd.Timestamp, intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    for left, right in intervals:
        if start <= right and end >= left:
            return True
    return False


def _contiguous_window_areas(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = int(length)
    if n < 3 or len(frame) < n:
        return pd.DataFrame()
    for start_idx in range(0, len(frame) - n + 1):
        segment = frame.iloc[start_idx:start_idx + n]
        metrics = loop_window_metrics(segment["p_C"].to_numpy(), segment["V_C"].to_numpy())
        rows.append(
            {
                "window_start": segment["date"].iloc[0],
                "window_end": segment["date"].iloc[-1],
                "loop_signed_area": metrics["loop_signed_area"],
                "loop_closed_area": metrics["loop_closed_area"],
            }
        )
    return pd.DataFrame(rows)


def _sample_from_values(values: Sequence[float], *, samples: int, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full(int(samples), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    size = max(0, int(samples))
    return rng.choice(arr, size=size, replace=True)


def _null_summary(observed_area: float, null_distribution: Sequence[float]) -> dict[str, Any]:
    null = np.asarray(null_distribution, dtype=float)
    null_abs = np.abs(null[np.isfinite(null)])
    observed_abs = abs(float(observed_area)) if np.isfinite(observed_area) else float("nan")
    if null_abs.size == 0 or not np.isfinite(observed_abs):
        return {
            "actual_null_percentile": float("nan"),
            "upper_tail_share": float("nan"),
            "null_mean_abs": float("nan"),
            "null_sd_abs": float("nan"),
            "null_q50_abs": float("nan"),
            "null_q90_abs": float("nan"),
            "null_q95_abs": float("nan"),
            "null_n": int(null_abs.size),
            "null_status": "insufficient_null",
        }
    percentile = float((1 + np.sum(null_abs <= observed_abs)) / (null_abs.size + 1))
    upper_tail = float((1 + np.sum(null_abs >= observed_abs)) / (null_abs.size + 1))
    status = "top_5pct" if percentile >= 0.95 else "top_10pct" if percentile >= 0.90 else "not_extreme"
    return {
        "actual_null_percentile": percentile,
        "upper_tail_share": upper_tail,
        "null_mean_abs": float(null_abs.mean()),
        "null_sd_abs": float(null_abs.std(ddof=0)),
        "null_q50_abs": float(np.quantile(null_abs, 0.50)),
        "null_q90_abs": float(np.quantile(null_abs, 0.90)),
        "null_q95_abs": float(np.quantile(null_abs, 0.95)),
        "null_n": int(null_abs.size),
        "null_status": status,
    }


def _row(
    *,
    region_key: str,
    region_label: str,
    panel_source: str,
    window_family: str,
    cycle_key: str,
    cycle_label: str,
    event_category: str,
    cycle_start: Any,
    cycle_end: Any,
    observed_start: Any,
    observed_end: Any,
    segmentation_window: int,
    null_method: str,
    null_distribution: Sequence[float],
    metrics: Mapping[str, float],
    n_obs: int,
    null_samples_requested: int,
    block_size: int | None,
) -> dict[str, Any]:
    signed = float(metrics.get("loop_signed_area", np.nan))
    return {
        "region_key": region_key,
        "region_label": region_label,
        "panel_source": panel_source,
        "window_family": window_family,
        "cycle_key": cycle_key,
        "cycle_label": cycle_label,
        "event_category": event_category,
        "cycle_start": _as_date(cycle_start),
        "cycle_end": _as_date(cycle_end),
        "observed_start": _as_date(observed_start),
        "observed_end": _as_date(observed_end),
        "segmentation_window": int(segmentation_window),
        "n_obs": int(n_obs),
        "null_method": null_method,
        "null_label": NULL_METHOD_LABELS.get(null_method, null_method),
        "null_samples_requested": int(null_samples_requested),
        "block_size": int(block_size) if block_size else "",
        "actual_signed_area": signed,
        "actual_closed_area": float(metrics.get("loop_closed_area", np.nan)),
        "actual_abs_area": float(metrics.get("loop_abs_area", np.nan)),
        "actual_open_integral": float(metrics.get("loop_open_integral", np.nan)),
        **_null_summary(signed, null_distribution),
    }


def _method_distribution(
    *,
    method: str,
    segment: pd.DataFrame,
    full_frame: pd.DataFrame,
    event_intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    actual_start: pd.Timestamp,
    actual_end: pd.Timestamp,
    samples: int,
    seed: int,
    block_size: int | None,
) -> np.ndarray:
    p = segment["p_C"].to_numpy()
    v = segment["V_C"].to_numpy()
    if method == "phase_randomization":
        return loop_area_null_distribution(p, v, samples=samples, method="phase", seed=seed)
    if method == "block_shuffle":
        return loop_area_null_distribution(p, v, samples=samples, method="block_shuffle", seed=seed, block_size=block_size)
    if method == "ar_surrogate":
        return loop_area_null_distribution(p, v, samples=samples, method="ar1", seed=seed)

    windows = _contiguous_window_areas(full_frame, len(segment))
    if windows.empty:
        return np.full(int(samples), np.nan, dtype=float)
    exact_actual = windows["window_start"].eq(actual_start) & windows["window_end"].eq(actual_end)
    candidates = windows.loc[~exact_actual].copy()

    if method == "placebo_periods":
        placebo_mask = [
            not _overlaps(pd.Timestamp(row.window_start), pd.Timestamp(row.window_end), event_intervals)
            and not _overlaps(pd.Timestamp(row.window_start), pd.Timestamp(row.window_end), [(actual_start, actual_end)])
            for row in candidates.itertuples()
        ]
        candidates = candidates.loc[placebo_mask]
        return candidates["loop_signed_area"].to_numpy(dtype=float)

    if method == "event_date_permutation":
        return _sample_from_values(candidates["loop_signed_area"].to_numpy(dtype=float), samples=samples, seed=seed)

    raise ValueError(f"unknown loop null-test method: {method}")


def _evaluate_segment(
    *,
    region_key: str,
    region_label: str,
    panel_source: str,
    window_family: str,
    cycle_key: str,
    cycle_label: str,
    event_category: str,
    cycle_start: Any,
    cycle_end: Any,
    segment: pd.DataFrame,
    full_frame: pd.DataFrame,
    event_intervals: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    segmentation_window: int,
    null_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(segment) < 3:
        return []
    metrics = loop_window_metrics(segment["p_C"].to_numpy(), segment["V_C"].to_numpy())
    actual_start = pd.Timestamp(segment["date"].iloc[0])
    actual_end = pd.Timestamp(segment["date"].iloc[-1])
    block_size = max(2, int(np.sqrt(len(segment))))
    rows: list[dict[str, Any]] = []
    for method in NULL_METHOD_LABELS:
        method_seed = _stable_seed(region_key, window_family, cycle_key, segmentation_window, method, base_seed=seed)
        null = _method_distribution(
            method=method,
            segment=segment,
            full_frame=full_frame,
            event_intervals=event_intervals,
            actual_start=actual_start,
            actual_end=actual_end,
            samples=null_samples,
            seed=method_seed,
            block_size=block_size,
        )
        rows.append(
            _row(
                region_key=region_key,
                region_label=region_label,
                panel_source=panel_source,
                window_family=window_family,
                cycle_key=cycle_key,
                cycle_label=cycle_label,
                event_category=event_category,
                cycle_start=cycle_start,
                cycle_end=cycle_end,
                observed_start=actual_start,
                observed_end=actual_end,
                segmentation_window=segmentation_window,
                null_method=method,
                null_distribution=null,
                metrics=metrics,
                n_obs=len(segment),
                null_samples_requested=null_samples,
                block_size=block_size if method == "block_shuffle" else None,
            )
        )
    return rows


def evaluate_loop_null_tests_region(
    frame: pd.DataFrame,
    *,
    region_key: str,
    region_label: str,
    panel_source: str,
    events: Sequence[Mapping[str, Any]] = (),
    latest_windows: Sequence[int] = SEGMENTATION_WINDOWS,
    null_samples: int = 199,
    seed: int = 9_241,
) -> pd.DataFrame:
    full_frame = _clean_frame(frame)
    if full_frame.empty:
        return pd.DataFrame()
    intervals = [(pd.Timestamp(event["start"]), pd.Timestamp(event["end"])) for event in events]
    rows: list[dict[str, Any]] = []

    for window in latest_windows:
        n = int(window)
        if n < 3 or len(full_frame) < n:
            continue
        segment = full_frame.iloc[-n:].reset_index(drop=True)
        rows.extend(
            _evaluate_segment(
                region_key=region_key,
                region_label=region_label,
                panel_source=panel_source,
                window_family="latest_rolling",
                cycle_key=f"latest_{n}q",
                cycle_label=f"Latest {n}-quarter window",
                event_category="rolling",
                cycle_start=segment["date"].iloc[0],
                cycle_end=segment["date"].iloc[-1],
                segment=segment,
                full_frame=full_frame,
                event_intervals=intervals,
                segmentation_window=n,
                null_samples=null_samples,
                seed=seed,
            )
        )

    for event in events:
        mask = (full_frame["date"] >= event["start"]) & (full_frame["date"] <= event["end"])
        segment = full_frame.loc[mask].reset_index(drop=True)
        if len(segment) < 3:
            continue
        rows.extend(
            _evaluate_segment(
                region_key=region_key,
                region_label=region_label,
                panel_source=panel_source,
                window_family="registered_event",
                cycle_key=str(event.get("key", "")),
                cycle_label=str(event.get("label", event.get("key", ""))),
                event_category=str(event.get("category", "")),
                cycle_start=event["start"],
                cycle_end=event["end"],
                segment=segment,
                full_frame=full_frame,
                event_intervals=intervals,
                segmentation_window=len(segment),
                null_samples=null_samples,
                seed=seed,
            )
        )

    return pd.DataFrame(rows)


def run_loop_area_null_tests(
    site_dir: Path,
    *,
    events_path: Path | None = None,
    latest_windows: Sequence[int] = SEGMENTATION_WINDOWS,
    null_samples: int = 199,
    seed: int = 9_241,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    root = site_dir.parent
    events_path = events_path or root / "data" / "report_events.csv"
    for region_key, (region_label, filename) in REGION_PANELS.items():
        path = site_dir / filename
        if not path.exists() and region_key == "jp":
            path = site_dir / "indicators.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        events = load_event_windows(events_path, region_key=region_key)
        frames.append(
            evaluate_loop_null_tests_region(
                frame,
                region_key=region_key,
                region_label=region_label,
                panel_source=path.relative_to(root).as_posix(),
                events=events,
                latest_windows=latest_windows,
                null_samples=null_samples,
                seed=seed,
            )
        )
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_loop_area_null_tests(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {"regions": {}, "overall": {}, "interpretation": "No loop-area null tests were produced."}
    summary: dict[str, Any] = {}
    for region_key, group in results.groupby("region_key", sort=True):
        latest = group[group["window_family"].eq("latest_rolling")]
        events = group[group["window_family"].eq("registered_event")]
        pct = pd.to_numeric(group["actual_null_percentile"], errors="coerce")
        latest_pct = pd.to_numeric(latest["actual_null_percentile"], errors="coerce")
        event_pct = pd.to_numeric(events["actual_null_percentile"], errors="coerce")
        summary[region_key] = {
            "region_label": str(group["region_label"].iloc[0]),
            "rows": int(len(group)),
            "latest_rows": int(len(latest)),
            "registered_event_rows": int(len(events)),
            "max_percentile": float(pct.max()) if pct.notna().any() else float("nan"),
            "max_latest_percentile": float(latest_pct.max()) if latest_pct.notna().any() else float("nan"),
            "max_event_percentile": float(event_pct.max()) if event_pct.notna().any() else float("nan"),
            "top_5pct_rows": int(group["null_status"].eq("top_5pct").sum()),
            "top_10pct_rows": int(group["null_status"].isin(["top_5pct", "top_10pct"]).sum()),
        }
    latest_12 = results[
        results["window_family"].eq("latest_rolling") & results["segmentation_window"].eq(12)
    ].copy()
    top5 = int(results["null_status"].eq("top_5pct").sum())
    top10 = int(results["null_status"].isin(["top_5pct", "top_10pct"]).sum())
    return {
        "regions": summary,
        "overall": {
            "rows": int(len(results)),
            "latest_12_rows": int(len(latest_12)),
            "top_5pct_rows": top5,
            "top_10pct_rows": top10,
            "null_methods": list(NULL_METHOD_LABELS.keys()),
            "segmentation_windows": list(SEGMENTATION_WINDOWS),
        },
        "interpretation": (
            "Closed loop area is treated as hysteresis-like evidence only when the observed "
            "area is extreme relative to block-shuffle, phase-randomized, AR(1), event-date, "
            "and placebo-window null distributions. Otherwise it remains a monitoring statistic."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    if abs(val) >= 100_000 or (0 < abs(val) < 0.001):
        return f"{val:.{digits}e}"
    return f"{val:.{digits}f}"


def _latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def _segmentation_ranges(results: pd.DataFrame) -> pd.DataFrame:
    latest = results[results["window_family"].eq("latest_rolling")].copy()
    if latest.empty:
        return pd.DataFrame(columns=["region_key", "null_method", "seg_pctile_range"])
    grouped = latest.groupby(["region_key", "null_method"])["actual_null_percentile"]
    ranges = grouped.agg(lambda x: f"{_fmt(pd.to_numeric(x, errors='coerce').min(), 2)}-{_fmt(pd.to_numeric(x, errors='coerce').max(), 2)}")
    return ranges.reset_index().rename(columns={"actual_null_percentile": "seg_pctile_range"})


def render_loop_area_null_tests_tex(results: pd.DataFrame) -> str:
    if results.empty:
        return "% Loop-area null-test table unavailable.\n"
    latest_12 = results[
        results["window_family"].eq("latest_rolling") & results["segmentation_window"].eq(12)
    ].copy()
    if latest_12.empty:
        latest_12 = results[results["window_family"].eq("latest_rolling")].copy()
    ranges = _segmentation_ranges(results)
    table = latest_12.merge(ranges, on=["region_key", "null_method"], how="left")
    table = table.sort_values(["region_label", "null_method"])

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\caption{Loop-area null tests for the latest closed-cycle window.}",
        "  \\label{tab:loop_area_null_tests}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{@{}llllllll@{}}",
        "    \\toprule",
        "    Region & Null method & Window & Closed area & Null pctl. & Upper-tail & Seg. pctl. range & Decision \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        decision = {
            "top_5pct": "top 5%",
            "top_10pct": "top 10%",
            "not_extreme": "not extreme",
            "insufficient_null": "insufficient null",
        }.get(str(row["null_status"]), str(row["null_status"]))
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["region_label"]),
                    _latex_escape(row["null_label"]),
                    f"{int(row['segmentation_window'])}q",
                    _fmt(row["actual_closed_area"]),
                    _fmt(row["actual_null_percentile"]),
                    _fmt(row["upper_tail_share"]),
                    _latex_escape(row.get("seg_pctile_range", "")),
                    _latex_escape(decision),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  }",
            "  \\par\\smallskip\\raggedright\\footnotesize "
            "The statistic is the absolute closed-loop area in the $(V_C,p_C)$ plane. "
            "Null percentile is the share of absolute null areas below the observed area; "
            "upper-tail is the finite-sample tail share above it. The segmentation range "
            "summarises the same percentile across the 8-, 12-, and 16-quarter latest "
            "windows. The companion CSV also reports registered-event windows from the "
            "event registry, event-date permutations, and placebo periods outside the "
            "registered events.",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def write_loop_area_null_test_outputs(results: pd.DataFrame, *, root: Path) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    csv_path = site_dir / "loop_area_null_tests.csv"
    json_path = data_dir / "loop_area_null_summary.json"
    tex_path = tex_dir / "theory_loop_area_null_tests.tex"

    results.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(summarize_loop_area_null_tests(results), indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tex_path.write_text(render_loop_area_null_tests_tex(results), encoding="utf-8")
    return [csv_path, json_path, tex_path]
