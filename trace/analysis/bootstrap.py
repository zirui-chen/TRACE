"""
Cluster bootstrap and non-inferiority machinery (Appendix C, Table 2).

Primary procedure (§4.1, Appendix C): a hierarchical wild cluster bootstrap
(Cameron-Gelbach-Miller 2008; Webb 2014 six-point weights) over the 11
provenance clusters of trace.benchmarks, with B = 10,000 replicates
(defaults.BOOTSTRAP_REPLICATES). Within each replicate, seeds are resampled
independently within each method -- seeds are NOT paired across arms because
the arms consume different data streams (Appendix O, "Seeds and intervals").
The wild perturbation below operates on residuals around the grand mean of
the per-graph paired ΔMRR (TRACE - ULTRA-Real), which is the cluster-level
half of the hierarchical scheme.

Table 2 reports three procedures on the same data:
    wild_cluster_bootstrap   one-sided 95% lower bound  -0.006  (primary)
    mixed_model              one-sided 95% lower bound  -0.005  (borderline)
    graph_level_bootstrap    one-sided 95% lower bound  -0.002  (ignores clusters)

The non-inferiority margin delta = 0.01 (defaults.NON_INFERIORITY_MARGIN) was
fixed before any TRACE evaluation; the delta = 0.005 verdict is read as
"unestablished", not as evidence of a practical loss.

scipy is not imported at module scope: the Student-t survival function and
its quantile are computed from the regularized incomplete beta function
(Lentz continued fraction), so this module imports with only numpy installed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from trace.defaults import BOOTSTRAP_REPLICATES, NON_INFERIORITY_MARGIN, TIE_BAND

__all__ = [
    "webb_weights",
    "wild_cluster_bootstrap",
    "graph_level_bootstrap",
    "mixed_model_lower_bound",
    "noninferiority_test",
    "wins_ties_losses",
    "sign_test",
    "cumulative_delta",
]

# Webb (2014) six-point support: {+-1, +-sqrt(3/5), +-sqrt(1/5)}
_WEBB_MAGNITUDES = (1.0, math.sqrt(3.0 / 5.0), math.sqrt(1.0 / 5.0))


# ---------------------------------------------------------------------------
# Webb wild weights
# ---------------------------------------------------------------------------

def webb_weights(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw `size` wild-bootstrap weights uniformly from the Webb (2014)
    six-point support {+-1, +-sqrt(3/5), +-sqrt(1/5)}.

    Parameters
    ----------
    rng  : numpy Generator (rng.integers required).
    size : number of weights.

    Returns
    -------
    np.ndarray of shape (size,) with values in the six-point support.
    """
    idx = rng.integers(0, 3, size=size)
    sign = 2 * rng.integers(0, 2, size=size) - 1
    mags = np.asarray(_WEBB_MAGNITUDES, dtype=np.float64)
    return sign * mags[idx]


# ---------------------------------------------------------------------------
# Student-t tail / quantile from the incomplete beta (no scipy at import)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    MAXIT, EPS, FPMIN = 200, 3.0e-9, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log(1.0 - x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: int) -> float:
    """Upper-tail probability P(T_df > t) for the Student-t distribution."""
    if df <= 0:
        return float("nan")
    p = 0.5 * _betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return p if t >= 0 else 1.0 - p


def _t_isf(q: float, df: int) -> float:
    """Inverse survival function: t with P(T_df > t) = q (bisection)."""
    if not (0.0 < q < 1.0):
        return float("nan")
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_sf(mid, df) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _norm_sf(z: float) -> float:
    """Upper-tail probability of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_isf(q: float) -> float:
    """Inverse survival of the standard normal (bisection)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_sf(mid) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Bootstrap procedures (Table 2)
# ---------------------------------------------------------------------------

def _as_arrays(deltas, cluster_ids=None):
    d = np.asarray(deltas, dtype=np.float64).ravel()
    if d.size == 0:
        raise ValueError("deltas must be non-empty")
    if cluster_ids is None:
        return d, None
    c = np.asarray(cluster_ids).ravel()
    if c.size != d.size:
        raise ValueError("deltas and cluster_ids must have equal length")
    return d, c


def _interval_result(reps: np.ndarray, mean: float, alpha: float,
                     B: int, n_clusters: int) -> Dict[str, float]:
    return {
        "mean": float(mean),
        "lower": float(np.quantile(reps, alpha / 2.0)),
        "upper": float(np.quantile(reps, 1.0 - alpha / 2.0)),
        "one_sided_lower": float(np.quantile(reps, alpha)),
        "B": int(B),
        "n_clusters": int(n_clusters),
    }


def wild_cluster_bootstrap(deltas, cluster_ids,
                           B: int = BOOTSTRAP_REPLICATES,
                           alpha: float = 0.05,
                           rng: Optional[np.random.Generator] = None
                           ) -> Dict[str, float]:
    """Primary inference procedure (Appendix C): wild cluster bootstrap with
    Webb six-point weights over provenance clusters.

    For per-graph paired deltas Δ_i = MRR_trace(i) - MRR_ultra_real(i) with
    provenance cluster c(i):

        Δ̄   = mean_i Δ_i
        e_i = Δ_i - Δ̄                                  (residuals, sum to 0)
        Δ̄*ᵇ = Δ̄ + Σ_c w_cᵇ Σ_{i ∈ c} e_i / N           (replicate b)

    with an independent Webb weight w_cᵇ per provenance cluster per replicate.
    The reported interval is the percentile interval of {Δ̄*ᵇ}.

    Note on seeds: seeds are resampled within methods (each arm's seed
    replicates are drawn independently), because seeds are not paired across
    arms -- the arms are trained on different data streams (Appendix O). The
    cluster-level wild perturbation here is the hierarchical component; the
    within-arm seed resampling happens upstream when per-graph Δ_i are formed.

    Parameters
    ----------
    deltas      : (N,) per-graph paired ΔMRR (TRACE - ULTRA-Real).
    cluster_ids : (N,) provenance cluster id per graph (trace.benchmarks.cluster_ids).
    B           : number of replicates (default defaults.BOOTSTRAP_REPLICATES = 10,000).
    alpha       : significance level; interval is the (alpha, 1-alpha) percentile band.
    rng         : numpy Generator; default_rng() when None.

    Returns
    -------
    dict(mean, lower, upper, one_sided_lower, B, n_clusters)
    """
    d, c = _as_arrays(deltas, cluster_ids)
    rng = np.random.default_rng() if rng is None else rng
    N = d.size
    _, c_inv = np.unique(c, return_inverse=True)
    n_clusters = int(c_inv.max()) + 1

    grand_mean = float(d.mean())
    resid = d - grand_mean
    cluster_sums = np.bincount(c_inv, weights=resid, minlength=n_clusters)

    W = webb_weights(rng, B * n_clusters).reshape(B, n_clusters)
    reps = grand_mean + (W @ cluster_sums) / N
    return _interval_result(reps, grand_mean, alpha, B, n_clusters)


def graph_level_bootstrap(deltas,
                          B: int = BOOTSTRAP_REPLICATES,
                          alpha: float = 0.05,
                          rng: Optional[np.random.Generator] = None
                          ) -> Dict[str, float]:
    """Plain nonparametric bootstrap resampling graphs with replacement
    (Table 2 row 3; ignores provenance clustering). Reported only to show how
    much apparent precision comes from treating correlated graphs as
    independent; the paper quotes the cluster bound throughout.

    Returns the same dict shape as wild_cluster_bootstrap; `n_clusters` here
    equals the number of resampling units (graphs).
    """
    d, _ = _as_arrays(deltas)
    rng = np.random.default_rng() if rng is None else rng
    N = d.size
    idx = rng.integers(0, N, size=(B, N))
    reps = d[idx].mean(axis=1)
    return _interval_result(reps, float(d.mean()), alpha, B, N)


def mixed_model_lower_bound(deltas, cluster_ids, graph_ids,
                            n_draws: int = 2000,
                            alpha: float = 0.05,
                            rng: Optional[np.random.Generator] = None
                            ) -> Dict[str, float]:
    """Variance-components approximation to the linear mixed model of
    Appendix C / Table 2 row 2: random effects for provenance cluster,
    graph-within-cluster, and a residual covering seed-level variation
    (seeds resampled within methods, as in the primary procedure).

    The exact REML fit (cluster, graph-in-cluster, method x seed random
    effects) reported in the paper gives a one-sided 95% lower bound of
    -0.005 -- borderline against the delta = 0.005 margin (Table 2); this
    parametric-bootstrap approximation reproduces that bound's ballpark.

    Variance components are estimated by method of moments on the per-graph
    means (cluster spread, graph-within-cluster spread, pooled within-graph
    seed spread), then `n_draws` parametric bootstrap grand means are drawn
    around the observed grand mean.

    Parameters
    ----------
    deltas      : (M,) observations -- per-graph Δ, or per-(graph, seed) Δ with
                  `graph_ids` mapping rows to graphs.
    cluster_ids : (M,) provenance cluster id per observation.
    graph_ids   : (M,) graph id per observation.
    n_draws     : number of parametric bootstrap draws.
    alpha       : one-sided level for `one_sided_lower`.
    rng         : numpy Generator.

    Returns
    -------
    dict(one_sided_lower, mean, lower, upper, B, n_clusters, sigma2 = dict of
         variance-component estimates).
    """
    d, c = _as_arrays(deltas, cluster_ids)
    g = np.asarray(graph_ids).ravel()
    if g.size != d.size:
        raise ValueError("deltas and graph_ids must have equal length")
    rng = np.random.default_rng() if rng is None else rng

    _, g_inv = np.unique(g, return_inverse=True)
    _, c_inv = np.unique(c, return_inverse=True)
    n_graphs = g_inv.max() + 1
    n_clusters = c_inv.max() + 1

    grand = float(d.mean())
    n_g = np.bincount(g_inv, minlength=n_graphs)
    sum_g = np.bincount(g_inv, weights=d, minlength=n_graphs)
    mean_g = sum_g / n_g
    # cluster of each graph (graphs nest in clusters by construction)
    c_of_g = np.bincount(g_inv, weights=c_inv, minlength=n_graphs) / n_g
    c_of_g = np.rint(c_of_g).astype(int)

    # pooled within-graph (seed) variance
    ss_within = float(np.sum(d * d) - np.sum(sum_g * sum_g / n_g))
    dof_within = max(int(d.size - n_graphs), 1)
    sigma2_e = max(ss_within / dof_within, 0.0)

    # graph-within-cluster variance from graph means around cluster means
    mean_c = np.bincount(c_of_g, weights=mean_g * n_g, minlength=n_clusters) / \
        np.maximum(np.bincount(c_of_g, weights=n_g, minlength=n_clusters), 1)
    dev_g = mean_g - mean_c[c_of_g]
    n_bar_g = float(n_g.mean())
    sigma2_g = max(float(np.mean(dev_g ** 2)) - sigma2_e / max(n_bar_g, 1.0), 0.0)

    # cluster variance from spread of cluster means, deflated by graph noise
    n_bar_c = float((np.bincount(c_of_g, minlength=n_clusters)).mean())
    dev_c = mean_c - grand
    sigma2_c = max(float(np.mean(dev_c ** 2)) - sigma2_g / max(n_bar_c, 1.0), 0.0)

    # parametric bootstrap of the grand mean around the observed grand mean
    u = rng.normal(0.0, math.sqrt(sigma2_c), size=(n_draws, n_clusters))
    v = rng.normal(0.0, math.sqrt(sigma2_g), size=(n_draws, n_graphs))
    e = rng.normal(0.0, math.sqrt(sigma2_e), size=(n_draws, d.size))
    reps = grand + (u[:, c_inv].sum(axis=1) + v[:, g_inv].sum(axis=1)
                    + e.sum(axis=1)) / d.size
    out = _interval_result(reps, grand, alpha, n_draws, n_clusters)
    out["sigma2"] = {"cluster": sigma2_c, "graph_within_cluster": sigma2_g,
                     "residual": sigma2_e}
    return out


# ---------------------------------------------------------------------------
# Non-inferiority (Table 2, Appendix C)
# ---------------------------------------------------------------------------

def noninferiority_test(trace_seed_mrrs, ultra_seed_mrrs,
                        delta: float = NON_INFERIORITY_MARGIN,
                        alpha: float = 0.05) -> Dict[str, float]:
    """One-sided paired t-test of H0: mu_diff <= -delta vs. H1: mu_diff > -delta
    on per-seed aggregate MRRs (Table 2's non-inferiority margins).

    Parameters
    ----------
    trace_seed_mrrs : per-seed aggregate MRR of TRACE (one value per seed).
    ultra_seed_mrrs : per-seed aggregate MRR of ULTRA-Real (same seed order).
    delta           : non-inferiority margin; default 0.01
                      (defaults.NON_INFERIORITY_MARGIN, fixed a priori).
    alpha           : one-sided level; `ci_lower` is the (1 - alpha) bound.

    Returns
    -------
    dict(mean_diff, t_stat, df, p_value, reject_h0, margin, ci_lower)
    """
    t_arr = np.asarray(trace_seed_mrrs, dtype=np.float64).ravel()
    u_arr = np.asarray(ultra_seed_mrrs, dtype=np.float64).ravel()
    if t_arr.size != u_arr.size or t_arr.size < 2:
        raise ValueError("seed arrays must be paired and contain >= 2 seeds")
    d = t_arr - u_arr
    n = d.size
    mean_diff = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    df = n - 1
    if se == 0.0:
        t_stat = math.copysign(float("inf"), mean_diff + delta)
        p_value = 0.0 if t_stat > 0 else 1.0
        ci_lower = mean_diff
    else:
        t_stat = (mean_diff + delta) / se           # H0 boundary at -delta
        p_value = _t_sf(t_stat, df)
        ci_lower = mean_diff - _t_isf(alpha, df) * se
    return {
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "df": int(df),
        "p_value": float(p_value),
        "reject_h0": bool(p_value < alpha),
        "margin": float(delta),
        "ci_lower": float(ci_lower),
    }


# ---------------------------------------------------------------------------
# Per-graph consistency (Appendix C)
# ---------------------------------------------------------------------------

def wins_ties_losses(deltas, tie_band: float = TIE_BAND) -> Dict[str, int]:
    """Win/tie/loss counts of per-graph Δ under the +/-0.005 tie band
    (defaults.TIE_BAND): paper reports 25 wins, 11 ties, 18 losses."""
    d = np.asarray(deltas, dtype=np.float64).ravel()
    band = float(tie_band)
    # epsilon guard: 3-decimal published MRRs put two graphs exactly on
    # the band (e.g. 0.290 - 0.285 == 0.005 + 4e-18 in binary), which
    # strict > would misclassify as a win/loss instead of a tie
    eps = 1e-12 * max(1.0, band)
    return {
        "wins": int(np.sum(d > band + eps)),
        "ties": int(np.sum(np.abs(d) <= band + eps)),
        "losses": int(np.sum(d < -band - eps)),
    }


def sign_test(deltas) -> Dict[str, float]:
    """Exact two-sided sign test reported with its one-sided p-value
    (Appendix C: 30/54 higher for TRACE, one-sided p = 0.25, two-sided 0.50).

    The binomial denominator is all N graphs -- exact ties count as non-wins
    (30/54 with n = 54 gives p = 0.2483, the paper's 0.25; dropping the tie
    would give the conventional-sign-test 0.205 instead). p_one_sided is the
    tail in the evidence direction, P(Bin(n, 1/2) >= max(n_positive,
    n_negative)); the two-sided value is twice the smaller tail.
    """
    d = np.asarray(deltas, dtype=np.float64).ravel()
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n = d.size
    if n == 0:
        return {"n_positive": 0, "n_negative": 0, "n": 0, "p_one_sided": 1.0}
    k = max(n_pos, n_neg)
    # exact binomial upper tail with p = 1/2, computed stably in log space
    tail = 0.0
    for i in range(k, n + 1):
        log_c = sum(math.log(n - j) - math.log(j + 1) for j in range(i)) if i > 0 else 0.0
        tail += math.exp(log_c - n * math.log(2.0))
    return {"n_positive": n_pos, "n_negative": n_neg, "n": n,
            "p_one_sided": float(min(1.0, tail))}


def cumulative_delta(deltas, order: Sequence[int]) -> List[float]:
    """Cumulative mean Δ for k = 1..N under a given graph `order`
    (Appendix D / Figure 7).

    The paper uses two orderings: ascending published ULTRA(3g) MRR (the
    robustness curve: +0.018 at k=1, +0.034 at k=14, ... +0.005 at k=54) and
    the adversarial ascending-Δ ordering of Figure 7 (starts at -0.042,
    crosses zero near k ≈ 50, ends at the overall mean +0.005).

    Returns a list of length N whose k-th entry is mean(deltas[order[:k]]).
    """
    d = np.asarray(deltas, dtype=np.float64).ravel()
    idx = np.asarray(order, dtype=np.int64).ravel()
    if idx.size != d.size or set(idx.tolist()) != set(range(d.size)):
        raise ValueError("order must be a permutation of range(len(deltas))")
    sorted_d = d[idx]
    cum = np.cumsum(sorted_d) / np.arange(1, d.size + 1)
    return [float(x) for x in cum]
