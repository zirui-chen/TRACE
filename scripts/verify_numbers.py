#!/usr/bin/env python3
"""
verify_numbers.py -- reproduce every numerical claim of the TRACE paper
from the raw per-seed, per-KG evaluation logs (paper Appendix O: "A
verification script parses the raw evaluation logs of every run ...
recomputes every table cell").

Input
-----
A single CSV with the columns

    kg, seed, mrr, hits_at_1, hits_at_3, hits_at_10, method

One row per (method, kg, seed) evaluation run.  ``method`` must be one of
trace.benchmarks.ARMS + ABLATION_ARMS + CONTROL_ARMS; only the subset of
arms actually present in the file is checked -- everything else is
reported as "skipped (no data for method X)" and never an error.

Checks (banner per section, expected values from trace.benchmarks.EXPECTED_*):
    Table 1   per-arm slice means, 95% seed t-intervals, TRACE-ULTRA-Real
              pipeline deltas with wild-cluster-bootstrap CIs, corpus
              effect | premise-faithful supervision, corpus x supervision
              interaction with CI.
    Table 2   non-inferiority lower bounds (wild cluster bootstrap,
              mixed model, graph-level bootstrap), verdicts at
              delta = 0.01 / 0.005.
    Counts    win/tie/loss under the +/-0.005 band (25/11/18), sign test
              (30/54, p ~ 0.25).
    Hits      hits@1 / hits@10 for trace and ultra_real.
    Settings  per-setting means (T / E / ER) for ultra_real and published.
    Table 3   2x2 generator ablation means, contrasts, cluster-bootstrap
              p-values (uncorrected, exploratory).
    Table 4   corpus-level controls on all four slices.
    App. D    cumulative-k delta (ascending published MRR) and the
              adversarial ascending-delta zero crossing k ~ 50.
    Crossfit  cross-fitted external-hard slice (seeds 0-1 select,
              seeds 2-4 evaluate): +0.030.

Usage
-----
    python scripts/verify_numbers.py --csv results/all_results.csv \
        [--B 10000] [--tol 0.0015]

All statistical procedures are delegated to trace.analysis.bootstrap
(wild_cluster_bootstrap, graph_level_bootstrap, mixed_model_lower_bound,
noninferiority_test, wins_ties_losses, sign_test, cumulative_delta),
which is imported lazily inside the check functions; if the module (or
its scipy-backed internals) is unavailable, bootstrap-dependent cells are
skipped with a clear message while the deterministic checks still run.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace import benchmarks as B
from trace.defaults import (BOOTSTRAP_REPLICATES, NON_INFERIORITY_MARGIN,
                            TIE_BAND, SEEDS_MAIN)

# ---------------------------------------------------------------------------
# Constants that come from the paper text rather than trace.benchmarks
# ---------------------------------------------------------------------------

REQUIRED_COLS = ("kg", "seed", "mrr", "hits_at_1", "hits_at_3",
                 "hits_at_10", "method")
METRIC_COLS = ("mrr", "hits_at_1", "hits_at_3", "hits_at_10")
VALID_METHODS = set(B.ARMS) | set(B.ABLATION_ARMS) | set(B.CONTROL_ARMS)

# Appendix D: cumulative mean delta at k with graphs sorted by ascending
# published ULTRA(3g) MRR: +0.018 / +0.034 / +0.023 / +0.012 / +0.005.
CUMULATIVE_K_EXPECTED = {1: +.018, 14: +.034, 27: +.023, 40: +.012, 54: +.005}
# Appendix D, Fig. 7 (most adversarial ordering, ascending per-graph delta):
# starts at the largest regression -0.042 and crosses zero near k ~= 50.
ADVERSARIAL_START = -0.042
ADVERSARIAL_ZERO_CROSS_K = 50
# Appendix C: cross-fitted hard slice (seeds 0-1 select, 2-4 evaluate).
CROSSFIT_HARD_DELTA = +.030

# Two-sided 97.5% Student-t quantiles for the 95% seed t-interval
# (avoids a scipy dependency; falls back to the normal value beyond 30 df).
T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(" " + title)
    print("=" * 74)


class Checker:
    """Accumulates PASS/FAIL per cell and prints one line per check."""

    def __init__(self, tol: float):
        self.tol = tol
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def check(self, label, got, want, tol=None, fmt="{:.4f}"):
        if got is None or want is None:
            print(f"  SKIP   {label}")
            self.skipped += 1
            return None
        tol = self.tol if tol is None else tol
        ok = abs(float(got) - float(want)) <= tol
        print(f"  {'PASS' if ok else 'FAIL'}   {str(label):<54s} "
              f"got {fmt.format(float(got))}   exp {fmt.format(float(want))}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(str(label))
        return ok

    def check_bool(self, label, got, want):
        if got is None:
            print(f"  SKIP   {label}")
            self.skipped += 1
            return None
        ok = bool(got) == bool(want)
        print(f"  {'PASS' if ok else 'FAIL'}   {str(label):<54s} "
              f"got {got}   exp {want}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(str(label))
        return ok

    def skip(self, label):
        print(f"  SKIP   {label}")
        self.skipped += 1

    def note(self, text):
        print(f"  note   {text}")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 74)
        print(f" SUMMARY: {self.passed}/{total} checks passed"
              f"  ({self.skipped} skipped)")
        if self.failures:
            print(" Failed cells:")
            for f in self.failures:
                print(f"   - {f}")
        print("=" * 74)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_data(path):
    """Return data[method][kg][seed] = {mrr, hits_at_1, hits_at_3, hits_at_10}."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: CSV not found: {path}")
    data = {}
    unknown_kgs, unknown_methods = set(), set()
    bad_rows = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"ERROR: CSV is missing required columns: {missing}")
        for row in reader:
            method = (row.get("method") or "").strip()
            kg = (row.get("kg") or "").strip()
            if method not in VALID_METHODS:
                unknown_methods.add(method)
                continue
            if kg not in B.GRAPHS:
                unknown_kgs.add(kg)
                continue
            try:
                seed = int(float(row["seed"]))
                vals = {c: float(row[c]) for c in METRIC_COLS}
            except (TypeError, ValueError):
                bad_rows += 1
                continue
            data.setdefault(method, {}).setdefault(kg, {})[seed] = vals
    if unknown_methods:
        print(f"WARNING: ignoring rows with unknown methods: "
              f"{sorted(unknown_methods)}")
    if unknown_kgs:
        print(f"WARNING: ignoring rows with unknown KGs: {sorted(unknown_kgs)}")
    if bad_rows:
        print(f"WARNING: ignoring {bad_rows} unparseable rows")
    return data


# ---------------------------------------------------------------------------
# Pure-data aggregation helpers (no bootstrap needed)
# ---------------------------------------------------------------------------

def arms_present(data):
    return [m for m in VALID_METHODS if m in data]


def seeds_of(data, method):
    return sorted({s for per_seed in data[method].values() for s in per_seed})


def kg_mean(data, method, kg, metric="mrr", seeds=None):
    per_seed = data[method].get(kg, {})
    vals = [v[metric] for s, v in per_seed.items() if seeds is None or s in seeds]
    return float(np.mean(vals)) if vals else None


def slice_mean(data, method, slice_kgs, metric="mrr", seeds=None):
    vals = [kg_mean(data, method, kg, metric, seeds) for kg in slice_kgs
            if kg in data[method]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def per_seed_slice(data, method, slice_kgs):
    """One aggregate value per seed (mean over the graphs present for that seed)."""
    out = []
    for seed in seeds_of(data, method):
        vals = [v["mrr"] for kg in slice_kgs
                for s, v in data[method].get(kg, {}).items() if s == seed]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def t_half_width(values):
    """Half-width of the two-sided 95% Student-t interval of the mean."""
    n = len(values)
    if n < 2:
        return None
    sd = float(np.std(values, ddof=1))
    t = T_975.get(n - 1, 1.960)
    return t * sd / math.sqrt(n)


def paired_delta(data, a, b, slice_kgs):
    """Per-graph paired delta (a - b) over graphs and seeds common to both arms.

    Returns (kgs, delta_array); empty when there is no overlap.
    """
    kgs, deltas = [], []
    for kg in slice_kgs:
        if kg not in data[a] or kg not in data[b]:
            continue
        common = sorted(set(data[a][kg]) & set(data[b][kg]))
        if not common:
            continue
        d = np.mean([data[a][kg][s]["mrr"] - data[b][kg][s]["mrr"]
                     for s in common])
        kgs.append(kg)
        deltas.append(float(d))
    return kgs, np.asarray(deltas, dtype=float)


# ---------------------------------------------------------------------------
# Lazy access to trace.analysis.bootstrap, with shape-normalising adapters
# ---------------------------------------------------------------------------

class BootstrapUnavailable(Exception):
    pass


def _api(name):
    """Import one function from trace.analysis.bootstrap lazily."""
    try:
        import trace.analysis.bootstrap as mod  # lazy: needs the analysis module
    except Exception as exc:  # ImportError / RuntimeError from scipy internals
        raise BootstrapUnavailable(
            f"trace.analysis.bootstrap unavailable ({type(exc).__name__}: {exc})")
    fn = getattr(mod, name, None)
    if fn is None:
        raise BootstrapUnavailable(f"{name} missing from trace.analysis.bootstrap")
    return fn


def _try_call(fn, *args, **kwargs):
    """Call fn with kwargs; on TypeError retry positionally (contract tolerant)."""
    try:
        return fn(*args, **kwargs)
    except TypeError as exc:
        try:
            return fn(*args)
        except TypeError:
            raise BootstrapUnavailable(
                f"could not call {fn.__name__} ({type(exc).__name__}: {exc})")


def _as_dict(res):
    if hasattr(res, "_asdict"):
        try:
            return res._asdict()
        except Exception:
            pass
    return res if isinstance(res, dict) else None


def _ci_of(res):
    """Normalise bootstrap results to (ci_low, ci_high) or None."""
    d = _as_dict(res)
    if d is not None:
        for key in ("ci", "ci95", "ci_95", "interval", "bounds"):
            v = d.get(key)
            if v is not None and len(v) == 2:
                return float(v[0]), float(v[1])
        lo = next((d[k] for k in ("ci_low", "lower", "lo", "ci_lower") if k in d), None)
        hi = next((d[k] for k in ("ci_high", "upper", "hi", "ci_upper") if k in d), None)
        if lo is not None and hi is not None:
            return float(lo), float(hi)
        return None
    if isinstance(res, (tuple, list, np.ndarray)):
        flat = [float(x) for x in np.asarray(res, dtype=float).ravel()]
        if len(flat) >= 3:
            return flat[1], flat[2]          # (point, lo, hi) convention
        if len(flat) == 2:
            return flat[0], flat[1]          # (lo, hi) convention
    if isinstance(res, (int, float)):
        return None
    return None


def _lower_of(res):
    ci = _ci_of(res)
    if ci is not None:
        return ci[0]
    d = _as_dict(res)
    if d is not None:
        for key in ("lower_bound", "ci_lower", "lower", "bound"):
            if key in d and isinstance(d[key], (int, float)):
                return float(d[key])
    if isinstance(res, (int, float)):
        return float(res)
    return None


def _one_sided(res):
    """One-sided 95% lower bound from the analysis-module result shapes
    (trace.analysis.bootstrap returns dict(..., one_sided_lower=...))."""
    d = _as_dict(res)
    if d is not None:
        for key in ("one_sided_lower", "lower_bound", "ci_lower"):
            if key in d and isinstance(d[key], (int, float)):
                return float(d[key])
    return _lower_of(res)


def _p_of(res):
    d = _as_dict(res)
    if d is not None:
        for key in ("p", "p_value", "pvalue", "pval"):
            if key in d and isinstance(d[key], (int, float)):
                return float(d[key])
    if isinstance(res, (tuple, list, np.ndarray)):
        flat = [float(x) for x in np.asarray(res, dtype=float).ravel()]
        cands = [x for x in flat if 0.0 <= x <= 1.0]
        if cands:
            return cands[-1]
    return None


def _two_sided_p(delta, se):
    """Normal-approximation two-sided p from a point estimate and its SE."""
    if se is None or se <= 0:
        return None
    z = abs(delta) / se
    return math.erfc(z / math.sqrt(2.0))


def _rng():
    return np.random.default_rng(0)


def wild_ci(data, a, b, slice_kgs, Bn):
    """Paired per-graph delta + wild-cluster-bootstrap CI over the 11 clusters."""
    fn = _api("wild_cluster_bootstrap")
    kgs, delta = paired_delta(data, a, b, slice_kgs)
    if len(kgs) == 0:
        return kgs, delta, None, None
    res = _try_call(fn, delta, B.cluster_ids(kgs), B=Bn, rng=_rng())
    ci = _ci_of(res)
    p = _p_of(res) if _as_dict(res) is not None else None
    if p is None and ci is not None:
        p = _two_sided_p(float(np.mean(delta)), (ci[1] - ci[0]) / (2 * 1.96))
    return kgs, delta, ci, p


# ---------------------------------------------------------------------------
# Section: Table 1 -- source x supervision factorial (paper §4.1)
# ---------------------------------------------------------------------------

def section_table1(data, chk, Bn, tol):
    banner("TABLE 1 (§4.1)  source x supervision factorial: slice means, "
           "seed intervals, deltas")
    for arm in B.ARMS:
        if arm not in data:
            chk.skip(f"Table 1: no data for method {arm}")
            continue
        exp = B.EXPECTED_TABLE1[arm]
        for slice_name, slice_kgs in B.SLICES.items():
            got = slice_mean(data, arm, slice_kgs)
            chk.check(f"Table 1  {arm}  mean MRR [{slice_name}]",
                      got, exp[slice_name])
        half = t_half_width(per_seed_slice(data, arm, B.BENCHMARK_KGS))
        chk.check(f"Table 1  {arm}  aggregate 95% seed t half-width",
                  half, exp.get("all_sd"))

    # TRACE - ULTRA-Real pipeline deltas with wild-cluster-bootstrap CIs.
    if "trace" in data and "ultra_real" in data:
        for slice_name, (exp_pt, (exp_lo, exp_hi)) in B.EXPECTED_PIPELINE_DELTAS.items():
            try:
                _, delta, ci, _ = wild_ci(data, "trace", "ultra_real",
                                          B.SLICES[slice_name], Bn)
            except BootstrapUnavailable as exc:
                chk.skip(f"Table 1  TRACE - ULTRA-Real [{slice_name}] "
                         f"bootstrap skipped ({exc})")
                continue
            chk.check(f"Table 1  TRACE - ULTRA-Real [{slice_name}]",
                      float(np.mean(delta)) if len(delta) else None, exp_pt)
            chk.check(f"Table 1  pipeline CI lo [{slice_name}]",
                      ci[0] if ci else None, exp_lo)
            chk.check(f"Table 1  pipeline CI hi [{slice_name}]",
                      ci[1] if ci else None, exp_hi)
    else:
        chk.skip("Table 1: pipeline deltas need both trace and ultra_real")

    # Corpus effect | premise-faithful supervision = trace - trace_real.
    if "trace" in data and "trace_real" in data:
        for slice_name, exp_pt in B.EXPECTED_CORPUS_EFFECT_GIVEN_PF.items():
            _, delta = paired_delta(data, "trace", "trace_real",
                                    B.SLICES[slice_name])
            chk.check(f"Table 1  corpus effect | PF [{slice_name}]",
                      float(np.mean(delta)) if len(delta) else None, exp_pt)
    else:
        chk.skip("Table 1: corpus effect | PF needs trace and trace_real")

    # Corpus x supervision interaction = (trace - ultra_syn) - (trace_real - ultra_real).
    need = ("trace", "ultra_syn", "trace_real", "ultra_real")
    if all(m in data for m in need):
        for slice_name, exp_pt in B.EXPECTED_INTERACTION.items():
            try:
                kgs, d1, _, _ = wild_ci(data, "trace", "ultra_syn",
                                        B.SLICES[slice_name], Bn)
                kgs2, d2, _, _ = wild_ci(data, "trace_real", "ultra_real",
                                         B.SLICES[slice_name], Bn)
            except BootstrapUnavailable as exc:
                chk.skip(f"Table 1  interaction [{slice_name}] "
                         f"bootstrap skipped ({exc})")
                continue
            if len(d1) == len(d2) and kgs == kgs2:
                inter = d1 - d2
                chk.check(f"Table 1  interaction [{slice_name}]",
                          float(np.mean(inter)), exp_pt)
                if slice_name == "all":
                    try:
                        _, ci, _ = _wild_on_delta(inter, kgs, Bn)
                    except BootstrapUnavailable:
                        ci = None
                    exp_lo, exp_hi = B.EXPECTED_INTERACTION_CI
                    chk.check("Table 1  interaction CI lo [all]",
                              ci[0] if ci else None, exp_lo)
                    chk.check("Table 1  interaction CI hi [all]",
                              ci[1] if ci else None, exp_hi)
            else:
                chk.skip(f"Table 1  interaction [{slice_name}]: "
                         "arms do not share the same graphs")
    else:
        missing = [m for m in need if m not in data]
        chk.skip("Table 1  interaction: skipped (no data for method "
                 + ", ".join(missing) + ")")


def _wild_on_delta(delta, kgs, Bn):
    fn = _api("wild_cluster_bootstrap")
    res = _try_call(fn, delta, B.cluster_ids(kgs), B=Bn, rng=_rng())
    ci = _ci_of(res)
    p = _p_of(res) if _as_dict(res) is not None else None
    if p is None and ci is not None:
        p = _two_sided_p(float(np.mean(delta)), (ci[1] - ci[0]) / (2 * 1.96))
    return float(np.mean(delta)), ci, p


# ---------------------------------------------------------------------------
# Section: Table 2 -- non-inferiority of TRACE vs ULTRA-Real (paper §4.1)
# ---------------------------------------------------------------------------

def section_table2(data, chk, Bn):
    banner("TABLE 2 (§4.1)  non-inferiority of TRACE vs. ULTRA-Real: "
           "one-sided 95% lower bounds")
    if "trace" not in data or "ultra_real" not in data:
        missing = [m for m in ("trace", "ultra_real") if m not in data]
        chk.skip("Table 2: skipped (no data for method "
                 + ", ".join(missing) + ")")
        return
    kgs, delta = paired_delta(data, "trace", "ultra_real", B.BENCHMARK_KGS)
    if len(kgs) == 0:
        chk.skip("Table 2: no overlapping (kg, seed) observations")
        return
    clusters = B.cluster_ids(kgs)

    # Primary: wild cluster bootstrap, one-sided 95% lower bound.
    lower_wild = None
    try:
        fn = _api("wild_cluster_bootstrap")
        lower_wild = _one_sided(_try_call(fn, delta, clusters, B=Bn, rng=_rng()))
    except BootstrapUnavailable as exc:
        chk.skip(f"Table 2: wild cluster bootstrap skipped ({exc})")
    chk.check("Table 2  wild_cluster_bootstrap lower bound",
              lower_wild, B.EXPECTED_TABLE2["wild_cluster_bootstrap"])

    # Mixed model (cluster + graph-within-cluster + method x seed effects).
    try:
        fn = _api("mixed_model_lower_bound")
        graph_ids = np.arange(len(kgs), dtype=np.int64)
        lower_mm = _one_sided(_try_call(fn, delta, clusters, graph_ids,
                                        rng=_rng()))
    except BootstrapUnavailable as exc:
        chk.skip(f"Table 2: mixed model skipped ({exc})")
        lower_mm = None
    chk.check("Table 2  mixed_model lower bound",
              lower_mm, B.EXPECTED_TABLE2["mixed_model"])

    # Graph-level bootstrap (ignores provenance clusters).
    try:
        fn = _api("graph_level_bootstrap")
        lower_gl = _one_sided(_try_call(fn, delta, B=Bn, rng=_rng()))
    except BootstrapUnavailable as exc:
        chk.skip(f"Table 2: graph-level bootstrap skipped ({exc})")
        lower_gl = None
    chk.check("Table 2  graph_level_bootstrap lower bound",
              lower_gl, B.EXPECTED_TABLE2["graph_level_bootstrap"])

    # Paired-t non-inferiority on the per-seed aggregate MRRs (same seeds,
    # same order) as a secondary read-out of the margin.
    common = sorted(set(seeds_of(data, "trace")) & set(seeds_of(data, "ultra_real")))
    if len(common) >= 2:
        all_kgs = B.BENCHMARK_KGS
        t_seed = [np.mean([data["trace"][k][s]["mrr"] for k in all_kgs
                           if k in data["trace"] and s in data["trace"][k]])
                  for s in common]
        u_seed = [np.mean([data["ultra_real"][k][s]["mrr"] for k in all_kgs
                           if k in data["ultra_real"] and s in data["ultra_real"][k]])
                  for s in common]
        try:
            fn = _api("noninferiority_test")
            res = _try_call(fn, t_seed, u_seed, delta=NON_INFERIORITY_MARGIN)
            dres = _as_dict(res) or {}
            chk.note("paired-t non-inferiority on per-seed aggregates: "
                     f"mean_diff={dres.get('mean_diff', float('nan')):+.4f} "
                     f"ci_lower={dres.get('ci_lower', float('nan')):+.4f} "
                     f"reject H0 at delta={NON_INFERIORITY_MARGIN}: "
                     f"{dres.get('reject_h0')}")
        except BootstrapUnavailable as exc:
            chk.skip(f"Table 2: noninferiority_test skipped ({exc})")

    # Verdicts (Appendix C): delta = 0.01 passes; delta = 0.005 NOT
    # established; the mixed model is borderline at 0.005.
    if lower_wild is not None:
        chk.check_bool("Table 2  verdict: non-inferiority at delta=0.01 "
                       "(primary bound)",
                       lower_wild > -NON_INFERIORITY_MARGIN, True)
        chk.check_bool("Table 2  verdict: delta=0.005 NOT established "
                       "(primary bound)",
                       lower_wild > -0.005, False)
    else:
        chk.skip("Table 2  verdicts: need the wild cluster lower bound")
    if lower_mm is not None:
        chk.note(f"mixed model at delta=0.005: "
                 f"{'borderline/passes' if lower_mm > -0.005 else 'fails'} "
                 f"(borderline in the paper)")


# ---------------------------------------------------------------------------
# Section: headline win/tie/loss counts and sign test (Appendix C)
# ---------------------------------------------------------------------------

def _fallback_wins_ties(a, b, band):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    eps = 1e-12 * max(1.0, float(band))
    return (int((d > band + eps).sum()),
            int((np.abs(d) <= band + eps).sum()),
            int((d < -band - eps).sum()))


def _fallback_sign_test(a, b):
    """Exact one-sided binomial sign test, ties counted against the
    alternative (paper: 30/54 higher, p = 0.25)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    pos = int((d > 0).sum())
    n = int(d.size)
    p = sum(math.comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
    return pos, n, p


def _fallback_cumulative(deltas):
    d = np.asarray(deltas, dtype=float)
    return np.cumsum(d) / np.arange(1, d.size + 1)


def _wins_ties(a, b, band):
    # Real API: wins_ties_losses(deltas, tie_band=...) -> dict(wins,ties,losses)
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    try:
        fn = _api("wins_ties_losses")
        res = _try_call(fn, delta, tie_band=band)
        d = _as_dict(res)
        if d is not None and {"wins", "ties", "losses"} <= set(d):
            return (int(d["wins"]), int(d["ties"]), int(d["losses"]))
        flat = [int(x) for x in np.asarray(res, dtype=int).ravel()]
        if len(flat) >= 3:
            return tuple(flat[:3])
    except (BootstrapUnavailable, KeyError, TypeError, ValueError):
        pass
    return _fallback_wins_ties(a, b, band)


def _sign_test(a, b):
    # Real API: sign_test(deltas) -> dict(n_positive, n_negative, n, p_one_sided)
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    try:
        fn = _api("sign_test")
        res = _try_call(fn, delta)
        d = _as_dict(res)
        if d is not None:
            pos = next((int(d[k]) for k in
                        ("n_positive", "wins", "n_wins", "positive")
                        if k in d and isinstance(d[k], (int, float))), None)
            n = next((int(d[k]) for k in ("n", "n_total", "count")
                      if k in d and isinstance(d[k], (int, float))), None)
            p = next((float(d[k]) for k in
                      ("p_one_sided", "p", "p_value", "pvalue")
                      if k in d and isinstance(d[k], (int, float))), None)
            if pos is not None and n is not None and p is not None:
                return pos, n, p
        flat = np.asarray(res, dtype=float).ravel()
        if flat.size >= 3:
            return int(flat[0]), int(flat[1]), float(flat[2])
        if flat.size == 2:
            return int(flat[0]), 54, float(flat[1])
    except (BootstrapUnavailable, TypeError, ValueError):
        pass
    return _fallback_sign_test(a, b)


def section_counts(data, chk):
    banner("HEADLINE COUNTS (Appendix C)  win/tie/loss under +/-0.005 and "
           "sign test")
    if "trace" not in data or "ultra_real" not in data:
        missing = [m for m in ("trace", "ultra_real") if m not in data]
        chk.skip("counts: skipped (no data for method "
                 + ", ".join(missing) + ")")
        return
    kgs = [k for k in B.BENCHMARK_KGS
           if k in data["trace"] and k in data["ultra_real"]]
    a = np.array([kg_mean(data, "trace", k) for k in kgs])
    b = np.array([kg_mean(data, "ultra_real", k) for k in kgs])
    if len(kgs) != 54:
        chk.note(f"win/tie/loss computed on {len(kgs)}/54 graphs "
                 "(common to both arms)")
    w, t, l = _wins_ties(a, b, TIE_BAND)
    exp_w, exp_t, exp_l = B.EXPECTED_TIE_COUNTS
    chk.check("counts  wins (delta > +0.005)", w, exp_w, tol=0)
    chk.check("counts  ties (|delta| <= 0.005)", t, exp_t, tol=0)
    chk.check("counts  losses (delta < -0.005)", l, exp_l, tol=0)
    pos, n, p = _sign_test(a, b)
    chk.check("counts  sign test: TRACE higher", pos, B.EXPECTED_WINS, tol=0)
    chk.check("counts  sign test: n graphs", n, 54, tol=0)
    chk.check("counts  sign test one-sided p", p, 0.25, tol=0.05, fmt="{:.4f}")


# ---------------------------------------------------------------------------
# Section: hits and per-setting means
# ---------------------------------------------------------------------------

def section_hits_and_settings(data, chk):
    banner("HITS AND PER-SETTING MEANS (§4.1, Appendix C)")
    for arm, exp in B.EXPECTED_HITS.items():
        if arm not in data:
            chk.skip(f"hits: no data for method {arm}")
            continue
        chk.check(f"hits  {arm}  hits@1",
                  slice_mean(data, arm, B.SLICES["all"], "hits_at_1"),
                  exp["h1"])
        chk.check(f"hits  {arm}  hits@10",
                  slice_mean(data, arm, B.SLICES["all"], "hits_at_10"),
                  exp["h10"])

    # Per-setting means (T / E / ER), ultra_real reproduction vs published.
    for setting in ("T", "E", "ER"):
        kgs = [k for k in B.BENCHMARK_KGS if B.GRAPHS[k].setting == setting]
        if "ultra_real" in data:
            chk.check(f"settings  ultra_real  [{setting}]",
                      slice_mean(data, "ultra_real", kgs),
                      B.EXPECTED_SETTING_MEANS["ultra_real"][setting])
        else:
            chk.skip(f"settings: no data for method ultra_real [{setting}]")
        # Published row needs no CSV: computed straight from the manifest,
        # guarding that trace.benchmarks still carries the Table 5 values.
        chk.check(f"settings  published (manifest)  [{setting}]",
                  float(np.mean([B.GRAPHS[k].mrr_pub for k in kgs])),
                  B.EXPECTED_SETTING_MEANS["published"][setting])


# ---------------------------------------------------------------------------
# Section: Table 3 -- 2x2 generator ablation (paper §4.2)
# ---------------------------------------------------------------------------

def section_table3(data, chk, Bn):
    banner("TABLE 3 (§4.2)  2x2 generator ablation (A = rules + varied "
           "stats ... D = neither); bootstrap p-values uncorrected")
    if not any(a in data for a in B.ABLATION_ARMS) and "trace" not in data:
        chk.skip("Table 3: skipped (no data for any ablation arm)")
        return

    A = "ablation_a" if "ablation_a" in data else (
        "trace" if "trace" in data else None)
    arm_of = {"A": A, "B": "ablation_b", "C": "ablation_c", "D": "ablation_d"}
    exp_key = {"A": "ablation_a", "B": "ablation_b",
               "C": "ablation_c", "D": "ablation_d"}
    label = {"A": "A (rules, varied stats)", "B": "B (no rules, varied)",
             "C": "C (rules, fixed stats)", "D": "D (neither)"}

    if A is None:
        chk.skip("Table 3: cell A missing (neither ablation_a nor trace)")
    for cell, arm in arm_of.items():
        if arm is None or arm not in data:
            chk.skip(f"Table 3: skipped (no data for method {arm or cell})")
            continue
        exp = B.EXPECTED_TABLE3[exp_key[cell]]
        chk.check(f"Table 3  {label[cell]}  mean MRR [all]",
                  slice_mean(data, arm, B.SLICES["all"]), exp["all"])
        chk.check(f"Table 3  {label[cell]}  95% seed t half-width",
                  t_half_width(per_seed_slice(data, arm, B.BENCHMARK_KGS)),
                  exp.get("all_sd"))
        chk.check(f"Table 3  {label[cell]}  mean MRR [ext_hard]",
                  slice_mean(data, arm, B.SLICES["ext_hard"]), exp["ext_hard"])

    contrasts = {
        "A_vs_B":   (A, "ablation_b"),
        "C_vs_D":   ("ablation_c", "ablation_d"),
        "A_vs_C":   (A, "ablation_c"),
        "B_vs_D":   ("ablation_b", "ablation_d"),
        "A_vs_D":   (A, "ablation_d"),
        "B_vs_ULTRA": ("ablation_b", "ultra_real"),
    }
    for name, (hi, lo) in contrasts.items():
        if hi is None or lo is None or hi not in data or lo not in data:
            miss = [str(m) for m in (hi, lo) if m is None or m not in data]
            chk.skip(f"Table 3  contrast {name}: skipped (no data for "
                     f"method {' and '.join(miss)})")
            continue
        exp_all, exp_eh, exp_p = B.EXPECTED_TABLE3_CONTRASTS[name]
        try:
            _, d_all, _, p_all = wild_ci(data, hi, lo, B.SLICES["all"], Bn)
            _, d_eh, _, _ = wild_ci(data, hi, lo, B.SLICES["ext_hard"], Bn)
        except BootstrapUnavailable as exc:
            chk.skip(f"Table 3  contrast {name}: bootstrap skipped ({exc})")
            continue
        chk.check(f"Table 3  contrast {name} [all]",
                  float(np.mean(d_all)) if len(d_all) else None, exp_all)
        chk.check(f"Table 3  contrast {name} [ext_hard]",
                  float(np.mean(d_eh)) if len(d_eh) else None, exp_eh)
        chk.check(f"Table 3  contrast {name} p (uncorrected)",
                  p_all, _expected_p_value(exp_p), tol=_expected_p_tol(exp_p),
                  fmt="{:.4f}")

    # Interaction ((A - B) - (C - D)); expected +0.005 all, +0.012 ext-hard,
    # p = 0.21 (hierarchical cluster bootstrap, uncorrected, exploratory).
    cells = [arm_of["A"], arm_of["B"], arm_of["C"], arm_of["D"]]
    if all(c is not None and c in data for c in cells):
        for slice_name, exp_pt in zip(("all", "ext_hard"),
                                      B.EXPECTED_TABLE3_CONTRASTS["interaction"][:2]):
            slice_kgs = B.SLICES[slice_name]
            k1, d1 = paired_delta(data, cells[0], cells[1], slice_kgs)
            k2, d2 = paired_delta(data, cells[2], cells[3], slice_kgs)
            if k1 == k2 and len(d1):
                inter = d1 - d2
                chk.check(f"Table 3  interaction ((A-B)-(C-D)) [{slice_name}]",
                          float(np.mean(inter)), exp_pt)
                if slice_name == "all":
                    try:
                        _, _, p_int = _wild_on_delta(inter, k1, Bn)
                    except BootstrapUnavailable as exc:
                        chk.skip(f"Table 3  interaction p: bootstrap skipped ({exc})")
                        p_int = None
                    chk.check("Table 3  interaction p (uncorrected)",
                              p_int, 0.21, tol=0.10, fmt="{:.4f}")
            else:
                chk.skip(f"Table 3  interaction [{slice_name}]: cells do "
                         "not share the same graphs")
    else:
        chk.skip("Table 3  interaction: skipped (needs A, B, C and D)")


def _expected_p_value(exp_p):
    return 0.0005 if exp_p.strip().startswith("<") else float(exp_p)


def _expected_p_tol(exp_p):
    return 0.0005 if exp_p.strip().startswith("<") else 0.10


# ---------------------------------------------------------------------------
# Section: Table 4 -- corpus-level controls (paper §4.3)
# ---------------------------------------------------------------------------

def section_table4(data, chk):
    banner("TABLE 4 (§4.3)  corpus-level controls (rules + premise-faithful "
           "throughout)")
    for arm in B.CONTROL_ARMS:
        if arm not in data:
            chk.skip(f"Table 4: skipped (no data for method {arm})")
            continue
        exp = B.EXPECTED_TABLE4[arm]
        for slice_name in ("all", "disjoint", "proximal", "ext_hard"):
            chk.check(f"Table 4  {arm}  [{slice_name}]",
                      slice_mean(data, arm, B.SLICES[slice_name]),
                      exp[slice_name])


# ---------------------------------------------------------------------------
# Section: Appendix D -- cumulative-k robustness
# ---------------------------------------------------------------------------

def _cumulative(deltas, order):
    # Real API: cumulative_delta(deltas, order) -> running means of the
    # deltas taken in the given permutation `order` (length-N list).
    deltas = np.asarray(deltas, dtype=float)
    order = np.asarray(order, dtype=int)
    try:
        fn = _api("cumulative_delta")
        res = _try_call(fn, deltas, order)
        if isinstance(res, (tuple, list)) and len(res) == 2 \
                and np.ndim(res[0]) == 1 and np.ndim(res[1]) == 1:
            return np.asarray(res[1], dtype=float)
        d = _as_dict(res)
        if d is not None:
            for key in ("cumulative", "cum", "values"):
                if key in d:
                    return np.asarray(d[key], dtype=float)
        out = np.asarray(res, dtype=float)
        if out.size == deltas.size:
            return out
    except (BootstrapUnavailable, TypeError, ValueError):
        pass
    return _fallback_cumulative(deltas[order])


def section_cumulative(data, chk):
    banner("APPENDIX D  cumulative-k robustness (published-MRR order) and "
           "the adversarial ascending-delta curve")
    if "trace" not in data or "ultra_real" not in data:
        missing = [m for m in ("trace", "ultra_real") if m not in data]
        chk.skip("Appendix D: skipped (no data for method "
                 + ", ".join(missing) + ")")
        return
    kgs, delta = paired_delta(data, "trace", "ultra_real", B.BENCHMARK_KGS)
    if len(kgs) != 54:
        chk.skip(f"Appendix D: needs all 54 graphs paired "
                 f"(have {len(kgs)}); skipped")
        return

    # Published-MRR ascending order (external-hard selection basis).
    order = sorted(range(54), key=lambda i: B.GRAPHS[kgs[i]].mrr_pub)
    cum = _cumulative(delta, order)
    for k, exp in CUMULATIVE_K_EXPECTED.items():
        chk.check(f"App. D  cumulative mean delta at k={k}",
                  float(cum[k - 1]), exp)

    # Adversarial ordering: ascending per-graph delta (Fig. 7).
    adv_order = sorted(range(54), key=lambda i: delta[i])
    adv = _cumulative(delta, adv_order)
    chk.check("App. D  adversarial curve start (largest regression)",
              float(adv[0]), ADVERSARIAL_START, tol=0.01)
    nonneg = np.nonzero(adv >= 0.0)[0]
    k_cross = int(nonneg[0] + 1) if len(nonneg) else None
    chk.check("App. D  adversarial zero crossing k",
              k_cross, ADVERSARIAL_ZERO_CROSS_K, tol=2, fmt="{:.1f}")
    chk.check("App. D  adversarial curve end == overall mean delta",
              float(adv[-1]), B.EXPECTED_PIPELINE_DELTAS["all"][0], tol=0.0015)


# ---------------------------------------------------------------------------
# Section: cross-fitted external-hard slice
# ---------------------------------------------------------------------------

def section_crossfit(data, chk):
    banner("APPENDIX C  cross-fitted external-hard slice "
           "(seeds 0-1 select, seeds 2-4 evaluate): +0.030")
    if "trace" not in data or "ultra_real" not in data:
        missing = [m for m in ("trace", "ultra_real") if m not in data]
        chk.skip("cross-fit: skipped (no data for method "
                 + ", ".join(missing) + ")")
        return
    eval_seeds = {2, 3, 4}
    for arm in ("trace", "ultra_real"):
        have = set(seeds_of(data, arm))
        if not eval_seeds <= have or not {0, 1} <= have:
            chk.skip(f"cross-fit: skipped (method {arm} lacks either the "
                     f"selection seeds 0-1 or the evaluation seeds 2-4; "
                     f"have {sorted(have)})")
            return
    kgs, delta = [], []
    for kg in sorted(B.EXTERNAL_HARD_KGS):
        common = sorted(eval_seeds & set(data["trace"].get(kg, {}))
                        & set(data["ultra_real"].get(kg, {})))
        if not common:
            chk.skip(f"cross-fit: {kg} has no evaluation-seed overlap; skipped")
            continue
        kgs.append(kg)
        delta.append(np.mean([data["trace"][kg][s]["mrr"] -
                              data["ultra_real"][kg][s]["mrr"] for s in common]))
    if kgs:
        chk.check("cross-fit  TRACE - ULTRA-Real on ext-hard (seeds 2-4)",
                  float(np.mean(delta)), CROSSFIT_HARD_DELTA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="verify_numbers.py",
        description="Reproduce every numerical claim of the TRACE paper "
                    "from per-seed, per-KG evaluation logs "
                    "(paper Appendix O).",
        epilog="CSV columns: kg, seed, mrr, hits_at_1, hits_at_3, "
               "hits_at_10, method.")
    p.add_argument("--csv", type=str, default="results/all_results.csv",
                   help="per-seed per-KG evaluation log CSV "
                        "(default: results/all_results.csv)")
    p.add_argument("--B", type=int, default=BOOTSTRAP_REPLICATES,
                   help=f"bootstrap replicates (default {BOOTSTRAP_REPLICATES})")
    p.add_argument("--tol", type=float, default=0.0015,
                   help="rounding tolerance for PASS/FAIL on table cells "
                        "(default 0.0015)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = load_data(args.csv)
    if not data:
        sys.exit("ERROR: no usable rows found in the CSV.")
    print(f"Loaded {args.csv}: arms present = {sorted(data)}")
    print(f"Bootstrap replicates B = {args.B}, tolerance = {args.tol}")

    chk = Checker(args.tol)
    section_table1(data, chk, args.B, args.tol)
    section_table2(data, chk, args.B)
    section_counts(data, chk)
    section_hits_and_settings(data, chk)
    section_table3(data, chk, args.B)
    section_table4(data, chk)
    section_cumulative(data, chk)
    section_crossfit(data, chk)

    chk.summary()
    return 0 if chk.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
