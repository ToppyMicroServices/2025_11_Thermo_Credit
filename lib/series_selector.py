"""Helpers for selecting economic series with preferences and fallbacks."""
import os
from typing import Any, Callable, Dict, Iterable, List, Optional
import pandas as pd
import yaml

DEFAULT_START = "1990-01-01"

DEFAULT_SERIES: Dict[str, List[Dict[str, Any]]] = {
    "money_scale": [
        {"id": "MYAGM2JPM189S", "title": "Japan M2 (Monthly)", "start": DEFAULT_START},
        {"id": "JPNASSETS", "title": "Bank of Japan Total Assets", "start": DEFAULT_START},
    ],
    "base_proxy": [
        {"id": "JPNASSETS", "title": "Bank of Japan Total Assets", "start": DEFAULT_START},
    ],
    "yield_proxy": [
        {"id": "DGS10", "title": "US 10Y Treasury Yield", "start": DEFAULT_START},
    ],
    "money_scale_eu": [
        {"id": "ECBASSETS", "title": "ECB Total Assets", "start": "1999-01-01"},
        {"id": "MABMM301EZM189S", "title": "Euro Area M3 (Broad Money)", "start": "1999-01-01"},
    ],
    "base_proxy_eu": [
        {"id": "ECBASSETS", "title": "ECB Total Assets", "start": "1999-01-01"},
    ],
    "yield_proxy_eu": [
        {"id": "IRLTLT01EZM156N", "title": "Euro Area Long-term Yield", "start": "1999-01-01"},
        {"id": "DE10Y", "title": "Germany 10Y Bund Yield", "start": "1999-01-01"},
    ],
    "money_scale_us": [
        {"id": "WALCL", "title": "Federal Reserve Total Assets", "start": DEFAULT_START},
        {"id": "WM2NS", "title": "US M2 (NSA)", "start": DEFAULT_START},
    ],
    "base_proxy_us": [
        {"id": "WALCL", "title": "Federal Reserve Total Assets", "start": DEFAULT_START},
    ],
    "yield_proxy_us": [
        {"id": "DGS10", "title": "US 10Y Treasury Yield", "start": DEFAULT_START},
        {"id": "GS30", "title": "US 30Y Treasury Yield", "start": DEFAULT_START},
    ],
    # Credit enrichment (JP defaults; region-specific can override via config)
    "asset_proxy": [
        {"id": "JPNASSETS", "title": "BoJ Total Assets", "start": DEFAULT_START},
    ],
    "energy_proxy": [
        {"id": "NY.GDP.MKTP.CN", "title": "Nominal GDP (World Bank)", "start": DEFAULT_START},
    ],
    "depth_proxy": [
        {"id": "CRDQJPAPABIS", "title": "Private NF credit (BIS)", "start": DEFAULT_START},
    ],
    "turnover_proxy": [
        {"id": "MYAGM2JPM189S", "title": "Japan M2", "start": DEFAULT_START},
    ],
}


def _read_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            content = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"Warning: could not read {config_path}: {exc}")
        return {}

    if not isinstance(content, dict):
        return {}
    return content


def load_project_config(config_path: str) -> Dict[str, Any]:
    """Load the full project config as a dictionary (missing/invalid -> empty dict)."""
    return _read_config(config_path)


def load_series_preferences(config_path: str) -> Dict[str, List[Any]]:
    """Load per-role series preferences from a YAML config file if available."""
    content = _read_config(config_path)
    raw_series = content.get("series", {})
    if not isinstance(raw_series, dict):
        return {}

    prefs: Dict[str, List[Any]] = {}
    for role, spec in raw_series.items():
        if isinstance(spec, dict) and "preferred" in spec:
            entries = spec.get("preferred")
        else:
            entries = spec
        if isinstance(entries, list):
            prefs[role] = entries
        elif isinstance(entries, (str, dict)):
            prefs[role] = [entries]
    return prefs


def _normalize_candidate(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, str):
        raw = entry.strip()
        if not raw:
            return None
        if "@" in raw:
            series_id, start = raw.split("@", 1)
            series_id = series_id.strip()
            start = start.strip()
        else:
            series_id = raw
            start = None
        if not series_id:
            return None
        result: Dict[str, Any] = {"id": series_id}
        if start:
            result["start"] = start
        return result

    if isinstance(entry, dict):
        series_id = entry.get("id")
        if not isinstance(series_id, str) or not series_id.strip():
            return None
        result = {"id": series_id.strip()}
        if entry.get("start"):
            result["start"] = str(entry["start"]).strip()
        if entry.get("title"):
            result["title"] = str(entry["title"]).strip()
        if entry.get("note"):
            result["note"] = str(entry["note"]).strip()
        return result
    return None


def candidate_queue(
    role: str,
    env_var: Optional[str],
    preferences: Optional[Dict[str, List[Any]]],
    defaults: Optional[Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    preferences = preferences or {}
    defaults = defaults or DEFAULT_SERIES

    queue: List[Dict[str, Any]] = []
    seen: set = set()

    def _append(candidate: Dict[str, Any], source: str) -> None:
        series_id = candidate.get("id")
        if not series_id:
            return
        key = series_id.upper()
        if key in seen:
            return
        item = dict(candidate)
        item["source"] = source
        queue.append(item)
        seen.add(key)

    if env_var:
        raw = os.getenv(env_var, "").strip()
        if raw:
            candidate = _normalize_candidate(raw)
            if candidate:
                _append(candidate, f"env:{env_var}")

    pref_entries = preferences.get(role, [])
    for entry in pref_entries:
        candidate = _normalize_candidate(entry)
        if candidate:
            _append(candidate, "config")

    default_entries = defaults.get(role, [])
    for entry in default_entries:
        candidate = _normalize_candidate(entry)
        if not candidate:
            continue
        if isinstance(entry, dict):
            if entry.get("title") and not candidate.get("title"):
                candidate["title"] = entry["title"]
            if entry.get("note") and not candidate.get("note"):
                candidate["note"] = entry["note"]
            if entry.get("start") and not candidate.get("start"):
                candidate["start"] = entry["start"]
        _append(candidate, "default")

    return queue


def select_series(
    role: str,
    env_var: Optional[str],
    fetcher: Callable[[str, str], Any],
    *,
    preferences: Optional[Dict[str, List[Any]]] = None,
    defaults: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Pick a series for a given role using environment overrides, config preferences,
    and built-in defaults. Returns metadata plus fetched data.
    """
    queue = candidate_queue(role, env_var, preferences, defaults)
    if not queue:
        raise RuntimeError(f"No candidate series configured for role '{role}'.")

    errors: List[str] = []
    successes: List[Dict[str, Any]] = []

    def _latest_date(data: Any) -> Optional[pd.Timestamp]:
        if not hasattr(data, "__getitem__"):
            return None
        try:
            if "date" not in data.columns:  # type: ignore[attr-defined]
                return None
            dates = pd.to_datetime(data["date"], errors="coerce").dropna()
            if dates.empty:
                return None
            return pd.Timestamp(dates.max())
        except Exception:
            return None

    for candidate in queue:
        series_id = candidate["id"]
        start = candidate.get("start") or DEFAULT_START
        try:
            data = fetcher(series_id, start)
        except Exception as exc:
            errors.append(f"{series_id} ({candidate['source']}): {exc}")
            continue

        is_empty = False
        if hasattr(data, "empty"):
            is_empty = bool(getattr(data, "empty"))
        elif isinstance(data, (list, tuple, set, dict)):
            is_empty = len(data) == 0

        if is_empty:
            errors.append(f"{series_id} ({candidate['source']}): empty result")
            continue

        item = dict(candidate)
        item["data"] = data
        item["start"] = start
        item["_latest_date"] = _latest_date(data)
        successes.append(item)

    if successes:
        first = successes[0]
        first_latest = first.get("_latest_date")
        freshest = max(
            successes,
            key=lambda item: (
                item.get("_latest_date") is not None,
                item.get("_latest_date") or pd.Timestamp.min,
            ),
        )
        freshest_latest = freshest.get("_latest_date")
        if (
            isinstance(first_latest, pd.Timestamp)
            and isinstance(freshest_latest, pd.Timestamp)
            and freshest_latest - first_latest > pd.Timedelta(days=365)
        ):
            freshest["selection_reason"] = "freshness_fallback"
            freshest.pop("_latest_date", None)
            return freshest
        first.pop("_latest_date", None)
        return first

    details = ", ".join(errors) if errors else "no candidates returned data"
    raise RuntimeError(f"No usable series for role '{role}'. Tried: {details}")


__all__ = [
    "DEFAULT_SERIES",
    "DEFAULT_START",
    "load_project_config",
    "load_series_preferences",
    "candidate_queue",
    "select_series",
]
