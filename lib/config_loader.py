"""Configuration loader with per-region overrides.

Usage:
  from lib.config_loader import load_config
  cfg = load_config("us")  # merges base config.yml with config_us.yml if present

Merge rules:
  - Shallow merge for top-level keys (region file overrides base).
  - For mapping value under key 'series', perform key-wise override (deep one level).
  - Missing file -> ignored.
  - Environment variable CONFIG_REGION can override requested region when
    passed region_code is None or empty.

Baseline parameters (p0, V0, U0, S0, T0) can therefore differ by region and allow
distinct exergy (X_C) computations.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as fp:
            data = yaml.safe_load(fp)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to parse YAML in %s: %s, using empty config", path, exc)
        return {}
    except OSError as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to read %s: %s, using empty config", path, exc)
        return {}

def _merge_series(base_series: dict[str, Any], override_series: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(base_series, dict):
        base_series = {}
    if not isinstance(override_series, dict):
        return base_series
    out = dict(base_series)
    for k, v in override_series.items():
        out[k] = v
    return out

def load_config(region_code: str | None) -> dict[str, Any]:
    """Load base config.yml and merge region-specific override if present.

    Region file pattern: config_<region>.yml (e.g., config_us.yml).
    If CONFIG_REGION env var is set, it supersedes region_code.
    """
    env_region = os.getenv('CONFIG_REGION')
    if env_region:
        region_code = env_region.strip().lower()
    region_code = (region_code or '').strip().lower()
    base_path = os.path.join(ROOT, 'config.yml')
    base_cfg = _read_yaml(base_path)
    if not region_code:
        return base_cfg
    region_path = os.path.join(ROOT, f'config_{region_code}.yml')
    region_cfg = _read_yaml(region_path)
    if not region_cfg:
        return base_cfg
    merged = dict(base_cfg)
    for k, v in region_cfg.items():
        if k == 'series' and isinstance(v, dict):
            merged_series = _merge_series(base_cfg.get('series', {}), v)
            merged['series'] = merged_series
        elif k == 'external_coupling' and isinstance(v, dict):
            base_ext = base_cfg.get('external_coupling', {})
            ext = dict(base_ext) if isinstance(base_ext, dict) else {}
            for sub_key, sub_val in v.items():
                ext[sub_key] = sub_val
            merged['external_coupling'] = ext
        else:
            merged[k] = v
    return merged

__all__ = ['load_config']
