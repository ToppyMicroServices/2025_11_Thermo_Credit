from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class LoopArea:
    """Streaming open-path estimator of ∫ p_R dV_R with exponential forgetting.

    W_t = λ W_{t-1} + p_{t-1} (V_t - V_{t-1})
    """

    def __init__(self, lam: float = 1.0):
        self.prev_p = None
        self.prev_v = None
        self.W = 0.0
        self.lam = float(lam)

    def update(self, p, v):
        p = float(p)
        v = float(v)
        if self.prev_p is not None:
            self.W = self.lam * self.W + self.prev_p * (v - self.prev_v)
        self.prev_p, self.prev_v = p, v
        return self.W


def _finite_pair(p: Sequence[float], v: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    p_arr = np.asarray(p, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    n = min(p_arr.size, v_arr.size)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    p_arr = p_arr[:n]
    v_arr = v_arr[:n]
    mask = np.isfinite(p_arr) & np.isfinite(v_arr)
    return p_arr[mask], v_arr[mask]


def open_path_integral(p: Sequence[float], v: Sequence[float]) -> float:
    """Signed trapezoid approximation to ∫ p dV along the observed path."""
    p_arr, v_arr = _finite_pair(p, v)
    if p_arr.size < 2:
        return float("nan")
    return float(np.sum(0.5 * (p_arr[1:] + p_arr[:-1]) * np.diff(v_arr)))


def absolute_path_area(p: Sequence[float], v: Sequence[float]) -> float:
    """Unsigned path integral, useful when direction changes cancel."""
    p_arr, v_arr = _finite_pair(p, v)
    if p_arr.size < 2:
        return float("nan")
    increments = 0.5 * (p_arr[1:] + p_arr[:-1]) * np.diff(v_arr)
    return float(np.sum(np.abs(increments)))


def closed_loop_signed_area(p: Sequence[float], v: Sequence[float]) -> float:
    """Signed shoelace area of the closed polygon in the (V, p) plane."""
    p_arr, v_arr = _finite_pair(p, v)
    if p_arr.size < 3:
        return float("nan")
    return float(0.5 * np.sum(v_arr * np.roll(p_arr, -1) - np.roll(v_arr, -1) * p_arr))


def loop_window_metrics(p: Sequence[float], v: Sequence[float]) -> dict[str, float]:
    """Return signed, closed, absolute, and open-path area metrics for one window."""
    signed = closed_loop_signed_area(p, v)
    return {
        "loop_signed_area": signed,
        "loop_closed_area": abs(signed) if np.isfinite(signed) else float("nan"),
        "loop_abs_area": absolute_path_area(p, v),
        "loop_open_integral": open_path_integral(p, v),
    }


def rolling_loop_diagnostics(p: Sequence[float], v: Sequence[float], window: int) -> dict[str, np.ndarray]:
    """Compute loop metrics on rolling windows.

    The rolling window is a descriptive proxy only. Policy-cycle evidence should
    use explicit event windows via ``event_window_loop_diagnostics``.
    """
    p_arr = np.asarray(p, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    n = min(p_arr.size, v_arr.size)
    p_arr = p_arr[:n]
    v_arr = v_arr[:n]
    window = int(window)
    out = {
        "loop_signed_area": np.full(n, np.nan, dtype=float),
        "loop_closed_area": np.full(n, np.nan, dtype=float),
        "loop_abs_area": np.full(n, np.nan, dtype=float),
        "loop_open_integral": np.full(n, np.nan, dtype=float),
    }
    if window < 3 or n < window:
        return out
    for idx in range(window - 1, n):
        metrics = loop_window_metrics(p_arr[idx - window + 1:idx + 1], v_arr[idx - window + 1:idx + 1])
        for key, value in metrics.items():
            out[key][idx] = value
    return out


def _phase_randomized(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    clean = np.asarray(series, dtype=float)
    centered = clean - float(np.mean(clean))
    spectrum = np.fft.rfft(centered)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.size)
    phases[0] = 0.0
    if clean.size % 2 == 0 and phases.size:
        phases[-1] = 0.0
    randomized = np.fft.irfft(spectrum * np.exp(1j * phases), n=clean.size)
    return randomized + float(np.mean(clean))


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(1, n - block_size + 1), size=int(np.ceil(n / block_size)))
    pieces = [np.arange(start, min(start + block_size, n)) for start in starts]
    return np.concatenate(pieces)[:n]


def _block_shuffle_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [np.arange(start, min(start + block_size, n)) for start in range(0, n, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[idx] for idx in order])[:n]


def _ar1_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    clean = np.asarray(series, dtype=float)
    if clean.size < 3:
        return clean.copy()
    mean = float(np.mean(clean))
    centered = clean - mean
    scale = float(np.std(centered, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.full_like(clean, mean, dtype=float)

    lag = centered[:-1]
    lead = centered[1:]
    denom = float(np.dot(lag, lag))
    phi = float(np.dot(lag, lead) / denom) if denom > 0.0 else 0.0
    phi = float(np.clip(phi, -0.98, 0.98))
    resid = lead - phi * lag
    resid_sd = float(np.std(resid, ddof=1)) if resid.size > 1 else scale * np.sqrt(max(1.0 - phi**2, 0.0))
    if not np.isfinite(resid_sd) or resid_sd <= 0.0:
        resid_sd = scale * np.sqrt(max(1.0 - phi**2, 1e-6))

    out = np.empty_like(clean, dtype=float)
    out[0] = rng.normal(0.0, scale)
    for idx in range(1, clean.size):
        out[idx] = phi * out[idx - 1] + rng.normal(0.0, resid_sd)

    out_sd = float(np.std(out, ddof=0))
    if np.isfinite(out_sd) and out_sd > 0.0:
        out = (out - float(np.mean(out))) * (scale / out_sd)
    return out + mean


def loop_area_null_distribution(
    p: Sequence[float],
    v: Sequence[float],
    *,
    samples: int = 199,
    method: str = "phase",
    block_size: int | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Generate a null distribution for closed-loop signed area.

    Supported methods are ``phase`` (phase-randomized marginals), ``block``
    (paired block bootstrap), ``block_shuffle`` (paired non-overlapping block
    order shuffle), ``ar1`` (separate AR(1) surrogates), and
    ``permute``/``permuted_event_windows`` (paired time-order permutation).
    """
    p_arr, v_arr = _finite_pair(p, v)
    if p_arr.size < 3:
        return np.full(int(samples), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    samples = max(0, int(samples))
    method = str(method or "phase").strip().lower()
    out = np.full(samples, np.nan, dtype=float)
    n = p_arr.size
    if block_size is None:
        block_size = max(2, int(np.sqrt(n)))
    block_size = max(1, min(int(block_size), n))

    for idx in range(samples):
        if method == "phase":
            p_null = _phase_randomized(p_arr, rng)
            v_null = _phase_randomized(v_arr, rng)
        elif method in {"block", "block_bootstrap"}:
            order = _block_bootstrap_indices(n, block_size, rng)
            p_null = p_arr[order]
            v_null = v_arr[order]
        elif method in {"block_shuffle", "block-shuffle", "shuffle_blocks"}:
            order = _block_shuffle_indices(n, block_size, rng)
            p_null = p_arr[order]
            v_null = v_arr[order]
        elif method in {"ar", "ar1", "ar_surrogate", "ar1_surrogate"}:
            p_null = _ar1_surrogate(p_arr, rng)
            v_null = _ar1_surrogate(v_arr, rng)
        elif method in {"permute", "permutation", "permuted_event_windows"}:
            order = rng.permutation(n)
            p_null = p_arr[order]
            v_null = v_arr[order]
        else:
            raise ValueError(f"unknown loop-area null method: {method}")
        out[idx] = closed_loop_signed_area(p_null, v_null)
    return out


def loop_area_p_value(observed_area: float, null_distribution: Sequence[float]) -> float:
    null = np.asarray(null_distribution, dtype=float)
    null = np.abs(null[np.isfinite(null)])
    if null.size == 0 or not np.isfinite(observed_area):
        return float("nan")
    return float((1 + np.sum(null >= abs(float(observed_area)))) / (null.size + 1))


def event_window_loop_diagnostics(
    frame: pd.DataFrame,
    windows: Iterable[Mapping[str, Any]],
    *,
    region_key: str | None = None,
    date_col: str = "date",
    p_col: str = "p_C",
    v_col: str = "V_C",
    null_samples: int = 199,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Compute loop metrics for explicit policy or credit-cycle windows."""
    if frame is None or frame.empty or date_col not in frame.columns:
        return pd.DataFrame()
    local = frame.copy(deep=True).assign(**{date_col: pd.to_datetime(frame[date_col], errors="coerce")})
    rows: list[dict[str, Any]] = []
    for window in windows:
        regions = window.get("regions") if isinstance(window, Mapping) else None
        if region_key and regions and region_key not in set(regions):
            continue
        start = pd.to_datetime(window.get("start"), errors="coerce")
        end = pd.to_datetime(window.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        segment = local[(local[date_col] >= start) & (local[date_col] <= end)]
        if segment.empty or p_col not in segment.columns or v_col not in segment.columns:
            continue
        metrics = loop_window_metrics(segment[p_col].to_numpy(), segment[v_col].to_numpy())
        signed = metrics["loop_signed_area"]
        phase_null = loop_area_null_distribution(segment[p_col], segment[v_col], samples=null_samples, method="phase", seed=seed)
        block_null = loop_area_null_distribution(segment[p_col], segment[v_col], samples=null_samples, method="block", seed=seed)
        perm_null = loop_area_null_distribution(segment[p_col], segment[v_col], samples=null_samples, method="permuted_event_windows", seed=seed)
        rows.append({
            "cycle_key": window.get("key", ""),
            "cycle_label": window.get("label", window.get("key", "")),
            "cycle_start": start.date().isoformat(),
            "cycle_end": end.date().isoformat(),
            "n_obs": int(segment[[p_col, v_col]].dropna().shape[0]),
            **metrics,
            "loop_phase_p": loop_area_p_value(signed, phase_null),
            "loop_block_p": loop_area_p_value(signed, block_null),
            "loop_permuted_event_p": loop_area_p_value(signed, perm_null),
        })
    return pd.DataFrame(rows)
