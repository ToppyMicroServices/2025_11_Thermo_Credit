"""Utilities for reading parameter defaults from the project config."""

from typing import Any, Dict, Optional


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def leverage_share(config: Optional[Dict[str, Any]], region: str, default: float = 0.4) -> float:
    """Return the leverage share multiplier for a region, clamped to non-negative values."""
    if not isinstance(config, dict):
        return default
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        return default
    shares = parameters.get("leverage_share")
    if isinstance(shares, dict):
        candidate = shares.get(region, shares.get("default"))
        if candidate is not None:
            value = _coerce_float(candidate, default)
            return max(0.0, min(value, 10.0))
    return default


def allocation_weights(
    config: Optional[Dict[str, Any]],
    region: str,
    defaults: Dict[str, float],
) -> Dict[str, float]:
    """Return allocation weights for a region with defaults applied per key."""
    weights = dict(defaults)
    if not isinstance(config, dict):
        return weights
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        return weights
    allocations = parameters.get("quarterly_allocations")
    if not isinstance(allocations, dict):
        return weights
    region_cfg = allocations.get(region)
    if not isinstance(region_cfg, dict):
        return weights
    for key, value in region_cfg.items():
        weights[key] = _coerce_float(value, weights.get(key, 0.0))
    return weights


__all__ = [
    "allocation_weights",
    "leverage_share",
]
