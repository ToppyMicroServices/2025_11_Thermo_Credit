from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


REGION_SPECS: Sequence[tuple[str, str, Sequence[str]]] = (
    ("jp", "Japan (JP)", ("indicators_jp.csv", "indicators.csv")),
    ("eu", "Euro Area (EU)", ("indicators_eu.csv",)),
    ("us", "United States (US)", ("indicators_us.csv",)),
)


@dataclass
class RegionFrame:
    key: str
    label: str
    frame: pd.DataFrame
    source_path: str = ""
    panel_mode: str = "dashboard"


def _coerce_indicator_frame(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if frame.empty or "date" not in frame.columns:
        return None
    local = frame.copy().assign(date=pd.to_datetime(frame["date"], errors="coerce"))
    local = local.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return local if not local.empty else None


def _load_indicator_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return _coerce_indicator_frame(frame)


def _load_indicator_csv_from_ref(
    repo_root: Path,
    relative_path: str,
    source_ref: str,
) -> Optional[pd.DataFrame]:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{source_ref}:{relative_path}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        frame = pd.read_csv(io.StringIO(raw))
    except Exception:
        return None
    return _coerce_indicator_frame(frame)


def _realtime_candidate(candidate: str) -> str:
    if candidate == "indicators.csv":
        return "indicators_realtime.csv"
    return candidate.replace(".csv", "_realtime.csv")


def _candidate_sequence(candidates: Sequence[str], mode: str) -> list[str]:
    if mode != "realtime":
        return list(candidates)
    ordered: list[str] = []
    for candidate in candidates:
        realtime = _realtime_candidate(candidate)
        if realtime not in ordered:
            ordered.append(realtime)
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def load_region_frames(
    site_dir: Path,
    source_ref: str | None = None,
    *,
    mode: str = "dashboard",
) -> list[RegionFrame]:
    frames: list[RegionFrame] = []
    repo_root = site_dir.parent
    panel_mode = (mode or "dashboard").strip().lower()
    for key, label, candidates in REGION_SPECS:
        frame: Optional[pd.DataFrame] = None
        source_path = ""
        for candidate in _candidate_sequence(candidates, panel_mode):
            local_frame = _load_indicator_csv(site_dir / candidate)
            ref_frame = None
            if source_ref:
                ref_frame = _load_indicator_csv_from_ref(
                    repo_root,
                    f"{site_dir.name}/{candidate}",
                    source_ref,
                )
            if local_frame is None:
                selected = ref_frame
            elif ref_frame is None:
                selected = local_frame
            else:
                local_end = pd.to_datetime(local_frame["date"].max(), errors="coerce")
                ref_end = pd.to_datetime(ref_frame["date"].max(), errors="coerce")
                selected = (
                    ref_frame
                    if pd.notna(ref_end) and (pd.isna(local_end) or ref_end > local_end)
                    else local_frame
                )
            if selected is not None and not selected.empty:
                frame = selected
                source_path = f"{site_dir.name}/{candidate}"
                break
        if frame is not None:
            selected_mode = "realtime" if "realtime" in source_path else "dashboard"
            frames.append(
                RegionFrame(
                    key=key,
                    label=label,
                    frame=frame,
                    source_path=source_path,
                    panel_mode=selected_mode,
                )
            )
    return frames
