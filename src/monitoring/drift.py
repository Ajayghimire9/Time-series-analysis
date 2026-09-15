"""Lightweight distribution-drift checks without heavyweight dependencies."""

from __future__ import annotations

import numpy as np


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Return PSI between two numeric samples using reference quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        raise ValueError("At least 20 finite observations are required per sample")
    if np.all(ref == ref[0]):
        return 0.0 if np.all(cur == cur[0] == ref[0]) else float("inf")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_status(psi: float, warning: float = 0.10, critical: float = 0.25) -> str:
    """Map PSI to an operational status."""
    if psi >= critical:
        return "critical"
    if psi >= warning:
        return "warning"
    return "stable"
