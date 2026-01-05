"""Utilities for loading and normalizing raw input series declared in sources.json.

Normalization rule: scale each enabled series to 100 at its first non-missing observation.
Expected CSV schema: columns 'date' and 'value' (case-insensitive). If a different
filename is needed, provide "path" in sources.json entry.
"""
from __future__ import annotations

import json
import os

import pandas as pd


def load_sources(path: str = "data/sources.json") -> list[dict]:
    base_dir = os.path.dirname(os.path.abspath(path))
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            entries: list[dict] = []
            for e in data:
                if isinstance(e, dict):
                    rec = dict(e)
                    rec.setdefault("_base_dir", base_dir)
                    entries.append(rec)
            return entries
    except Exception:
        pass
    return []


def enabled_sources(sources: list[dict]) -> list[dict]:
    return [s for s in sources if s.get("enabled") is True]


def _series_csv_path(entry: dict) -> str | None:
    sid = entry.get("id")
    if not sid:
        return None
    base_dir = entry.get("_base_dir") or os.path.join(os.getcwd(), "data")
    custom = entry.get("path")
    candidates = []
    if custom:
        custom_path = custom if os.path.isabs(custom) else os.path.join(base_dir, custom)
        candidates.append(custom_path)
    # Look in the same folder as sources.json first
    candidates.append(os.path.join(base_dir, f"{sid}.csv"))
    # Fallback to repo-level data directory if different
    repo_candidate = os.path.join("data", f"{sid}.csv")
    if repo_candidate not in candidates:
        candidates.append(repo_candidate)

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def load_and_normalize(enabled: list[dict]) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    meta: list[tuple[str, str]] = []  # (series_id, country)
    for e in enabled:
        path = _series_csv_path(e)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            date_col = next((c for c in df.columns if str(c).lower() == "date"), None)
            val_col = next((c for c in df.columns if str(c).lower() == "value"), None)
            if not date_col or not val_col:
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[[date_col, val_col]].dropna().sort_values(date_col)
            if df.empty:
                continue
            first = df[val_col].iloc[0]
            if first == 0 or first is None:
                continue
            sid = e.get("id") or e.get("title") or "series"
            df["norm"] = df[val_col] / first * 100.0
            frames.append(df.rename(columns={date_col: "date", "norm": sid})[["date", sid]])
            meta.append((sid, (e.get("country") or e.get("Country") or "").upper()))
        except Exception:
            continue
    if not frames:
        return None
    out = frames[0]
    for fr in frames[1:]:
        out = out.merge(fr, on="date", how="outer")
    # Attach simple metadata map for plotting color grouping by country (if caller wants it)
    out = out.sort_values("date")
    out.attrs["series_country_map"] = {sid: country for sid, country in meta if sid}
    return out


__all__ = [
    "load_sources",
    "enabled_sources",
    "load_and_normalize",
]
