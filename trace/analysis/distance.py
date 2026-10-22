"""
Structural-distance constructions (Appendix E) and the Eq. 5 interaction fit
used by §5.1 / Figures 4 and 6.

Degree distance: for each graph, take the inference graph's canonical triples
and count each triple once at each endpoint (total degree, no inverses).
Degrees are binned into log2 bins on the union support of the two compared
graphs, +1/2 smoothing is applied, and the JS divergence (base 2) is
computed. d_g is the minimum over the three pretraining graphs, z-scored
across the 54 benchmark graphs.

Relation-graph distance: mean JS divergence between the degree distributions
of the four interaction types (0 head-tail, 1 head-head, 2 tail-head,
3 tail-tail) of the binary relation graph G_r (a coarsening of the KG,
structurally different from the degree distance rather than a refinement).

Eq. 5 (§5.1):
    MRR[g, m, s] = beta0 + beta1*m + beta2*d_g + beta3*(m*d_g)
                   + gamma^T x_g + u_{c(g)} + eps
with m = 1 for TRACE, d_g the z-scored distance, x_g the covariates
(log|V|, log|R|, evaluation setting) and u_{c(g)} provenance-cluster fixed
effects (one-hot). Interaction estimates (paper): beta3 = +0.011
[+0.004, +0.019] for min-of-3, +0.012 fb-only, +0.011 mean-of-3, +0.010 /
+0.011 in-/out-degree (benchmarks.EXPECTED_BETA3); slopes -0.021 (ULTRA-Real)
vs -0.010 (TRACE) per SD (benchmarks.EXPECTED_SLOPES).

scipy is not imported at module scope (cluster_spearman is pure numpy).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from trace.defaults import BOOTSTRAP_REPLICATES
from trace.analysis.bootstrap import _norm_isf, _norm_sf, wild_cluster_bootstrap

__all__ = [
    "degree_histogram",
    "js_divergence_degree",
    "relation_graph_distance",
    "structural_distances",
    "fit_interaction",
    "cluster_spearman",
]

# interaction types of the binary lifted relation graph (trace.generator.lift)
INTERACTION_TYPES = ("head_tail", "head_head", "tail_head", "tail_tail")
N_INTERACTION_TYPES = 4

_PRETRAIN_KEYS = ("fb15k237", "wn18rr", "codex_m")


# ---------------------------------------------------------------------------
# Degree histograms and JS divergence (Appendix E, "Degree distance")
# ---------------------------------------------------------------------------

def _endpoint_degrees(triples, n_entities: int, direction: str) -> np.ndarray:
    """Total / in / out degree per entity from canonical triples (E, 3)
    (h, r, t). 'total' counts each canonical triple once at each endpoint
    (no inverses)."""
    t = np.asarray(triples, dtype=np.int64)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("triples must have shape (E, 3)")
    h, tail = t[:, 0], t[:, 2]
    n = int(n_entities)
    if direction == "total":
        return np.bincount(np.concatenate([h, tail]), minlength=n)
    if direction == "out_degree":
        return np.bincount(h, minlength=n)
    if direction == "in_degree":
        return np.bincount(tail, minlength=n)
    raise ValueError(f"unknown direction {direction!r} "
                     "(expected 'total', 'in_degree', 'out_degree')")


def _log2_bins(lo: int, hi: int) -> np.ndarray:
    """Log2-spaced bin edges covering degrees in [lo, hi] (lo >= 1); the last
    edge strictly exceeds hi so every observed degree lands in some bin."""
    lo = max(int(lo), 1)
    hi = max(int(hi), lo)
    edges = np.unique(np.rint(2.0 ** np.arange(
        np.floor(np.log2(lo)), np.floor(np.log2(hi)) + 2.0)).astype(np.int64))
    return np.unique(np.maximum(edges, 1))


def degree_histogram(triples, n_entities: int, direction: str = "total",
                     bins=None) -> Tuple[np.ndarray, np.ndarray]:
    """Probability distribution of positive entity degrees, log2-binned with
    +1/2 smoothing (Appendix E).

    Parameters
    ----------
    triples     : (E, 3) canonical triples (h, r, t); each triple is counted
                  once at each endpoint for direction='total' (no inverses).
    n_entities  : |V| of the graph.
    direction   : 'total' | 'in_degree' | 'out_degree'.
    bins        : bin edges; when None they are log2 bins on this graph's own
                  support. For a two-graph comparison pass edges built on the
                  UNION support (see js_divergence_degree).

    Returns
    -------
    (probs, edges) with probs summing to 1 after +1/2 smoothing.
    """
    deg = _endpoint_degrees(triples, n_entities, direction)
    pos = deg[deg > 0]
    if pos.size == 0:
        return np.ones(1, dtype=np.float64), np.array([1, 2], dtype=np.int64)
    if bins is None:
        edges = _log2_bins(pos.min(), pos.max())
    else:
        edges = np.asarray(bins, dtype=np.int64)
    counts, edges = np.histogram(pos, bins=edges)
    smoothed = counts.astype(np.float64) + 0.5          # +1/2 smoothing
    return smoothed / smoothed.sum(), edges


def _union_bins(deg_a: np.ndarray, deg_b: np.ndarray) -> np.ndarray:
    pos_a, pos_b = deg_a[deg_a > 0], deg_b[deg_b > 0]
    if pos_a.size == 0 and pos_b.size == 0:
        return np.array([1, 2], dtype=np.int64)
    lo = min(int(pos_a.min()) if pos_a.size else 1,
             int(pos_b.min()) if pos_b.size else 1)
    hi = max(int(pos_a.max()) if pos_a.size else 1,
             int(pos_b.max()) if pos_b.size else 1)
    return _log2_bins(lo, hi)


def _js_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0)
    return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())


def js_divergence_degree(triples_a, n_a: int, triples_b, n_b: int,
                         direction: str = "total") -> float:
    """Base-2 Jensen-Shannon divergence between the degree distributions of
    two graphs, binned into log2 bins on the union support of the two graphs
    with +1/2 smoothing (Appendix E)."""
    deg_a = _endpoint_degrees(triples_a, n_a, direction)
    deg_b = _endpoint_degrees(triples_b, n_b, direction)
    edges = _union_bins(deg_a, deg_b)
    p, _ = degree_histogram(triples_a, n_a, direction=direction, bins=edges)
    q, _ = degree_histogram(triples_b, n_b, direction=direction, bins=edges)
    return _js_divergence_base2(p, q)


# ---------------------------------------------------------------------------
# Relation-graph distance (Appendix E, "Relation-graph distance")
# ---------------------------------------------------------------------------

def _type_pairs_and_degrees(gr: Dict, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unique unordered relation-node pairs and node degrees for interaction
    type t of a binary lifted relation graph dict
    {"edge_index": (2, E), "edge_type": (E,), "num_nodes": R}.

    The binary G_r is undirected per interaction type, so duplicated pairs
    are collapsed; self-pairs are excluded."""
    ei = np.asarray(gr["edge_index"], dtype=np.int64)
    et = np.asarray(gr["edge_type"], dtype=np.int64)
    n = int(gr["num_nodes"])
    mask = et == t
    u, v = ei[0][mask], ei[1][mask]
    keep = u != v
    u, v = u[keep], v[keep]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    pairs = np.unique(lo * n + hi)
    pu, pv = pairs // n, pairs % n
    deg = np.bincount(pu, minlength=n) + np.bincount(pv, minlength=n)
    return pairs, deg


def relation_graph_distance(gr_a: Dict, gr_b: Dict) -> float:
    """Mean base-2 JS divergence between the per-interaction-type degree
    distributions of two binary relation graphs (Appendix E): for each of the
    four types (0 head-tail, 1 head-head, 2 tail-head, 3 tail-tail) of the
    Lift(E) dicts produced by trace.generator.lift, the JS divergence of the
    relation-node degree distributions is computed; the distance is the mean
    over the four types.

    Sensitive to relation-graph density; comparatively insensitive to
    higher-order motifs -- a structurally different second view of distance.
    """
    dists = []
    for t in range(N_INTERACTION_TYPES):
        _, deg_a = _type_pairs_and_degrees(gr_a, t)
        _, deg_b = _type_pairs_and_degrees(gr_b, t)
        edges = _union_bins(deg_a, deg_b)
        pa = np.histogram(deg_a[deg_a > 0], bins=edges)[0].astype(np.float64) + 0.5
        pb = np.histogram(deg_b[deg_b > 0], bins=edges)[0].astype(np.float64) + 0.5
        dists.append(_js_divergence_base2(pa / pa.sum(), pb / pb.sum()))
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Benchmark-vs-pretraining distances (§5.1, Figure 6)
# ---------------------------------------------------------------------------

def structural_distances(benchmark_triples: Dict[str, Tuple[np.ndarray, int]],
                          pretraining_triples: Dict[str, Tuple[np.ndarray, int]],
                          aggregation: str = "min",
                          direction: str = "total"
                          ) -> Dict[str, float]:
    """Structural distance d_g for every benchmark graph, z-scored across the
    benchmark set (Appendix E; the robustness sweep of Figure 6).

    Parameters
    ----------
    benchmark_triples   : {kg: (triples, n_entities)}.
    pretraining_triples : {kg: (triples, n_entities)} with keys
                          'fb15k237', 'wn18rr', 'codex_m'.
    aggregation         : 'min'   -- paper default, min over the three
                          pretraining graphs;
                          'mean'  -- mean of the three;
                          'fb_only' -- distance to FB15k-237 only;
                          'in_degree' / 'out_degree' -- direction variants of
                          the min-of-3 distance (aggregation stays 'min').
    direction           : 'total' | 'in_degree' | 'out_degree'.

    Returns
    -------
    {kg: z-scored distance} (z-scored across the benchmark set).
    """
    if aggregation in ("in_degree", "out_degree"):
        direction, agg = aggregation, "min"
    elif aggregation in ("min", "mean", "fb_only"):
        agg = aggregation
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    for key in (_PRETRAIN_KEYS if agg != "fb_only" else (_PRETRAIN_KEYS[0],)):
        if key not in pretraining_triples:
            raise KeyError(f"pretraining graph {key!r} missing")

    raw: Dict[str, float] = {}
    for kg, (triples, n_ent) in benchmark_triples.items():
        if agg == "fb_only":
            keys = (_PRETRAIN_KEYS[0],)
        else:
            keys = _PRETRAIN_KEYS
        per_kg = [js_divergence_degree(triples, n_ent,
                                       pretraining_triples[k][0],
                                       pretraining_triples[k][1],
                                       direction=direction) for k in keys]
        raw[kg] = min(per_kg) if agg == "min" else float(np.mean(per_kg))

    vals = np.array(list(raw.values()), dtype=np.float64)
    if vals.size == 0:
        return {}
    mu, sd = vals.mean(), vals.std(ddof=0)
    sd = sd if sd > 0 else 1.0
    return {kg: float((v - mu) / sd) for kg, v in raw.items()}


# ---------------------------------------------------------------------------
# Eq. 5 interaction fit (§5.1)
# ---------------------------------------------------------------------------

def fit_interaction(mrr, method_ind, dist_z, covariates, cluster_ids,
                    *, graph_ids=None,
                    B: int = BOOTSTRAP_REPLICATES,
                    alpha: float = 0.05,
                    rng: Optional[np.random.Generator] = None) -> Dict:
    """OLS fit of Eq. 5 on per-seed observations via numpy lstsq, with
    provenance-cluster fixed effects (one-hot) in place of the random effect
    u_c(g), and a wild-cluster-bootstrap CI for beta3.

        MRR[g,m,s] = b0 + b1*m + b2*d_g + b3*(m*d_g) + gamma^T x_g
                     + u_c(g) + eps

    Parameters
    ----------
    mrr         : (N,) per-seed MRR observations.
    method_ind  : (N,) method indicator m (0 = ULTRA-Real, 1 = TRACE).
    dist_z      : (N,) z-scored structural distance of the row's graph.
    covariates  : (N, k) design array x_g (log|V|, log|R|, setting dummies),
                  or None.
    cluster_ids : (N,) provenance cluster id per row.
    graph_ids   : optional (N,) graph id per row, used to aggregate residuals
                  per graph for the CI. When None, rows are grouped by their
                  (cluster, dist_z, covariates) signature -- rows of the same
                  graph across seeds/methods share that signature.
    B, alpha    : wild-cluster-bootstrap replicates / level for the beta3 CI.
    rng         : numpy Generator.

    Returns
    -------
    dict with beta0..beta3, slopes={0: b2, 1: b2+b3} (per-method MRR loss per
    SD of distance; paper: -0.021 vs -0.010), beta3_ci=(lo, hi), p_value,
    se_beta3, ci_method, n_obs, n_clusters.
    """
    y = np.asarray(mrr, dtype=np.float64).ravel()
    m = np.asarray(method_ind, dtype=np.float64).ravel()
    d = np.asarray(dist_z, dtype=np.float64).ravel()
    c = np.asarray(cluster_ids).ravel()
    N = y.size
    if not (m.size == d.size == c.size == N):
        raise ValueError("mrr, method_ind, dist_z and cluster_ids must align")

    # design: [1, m, d, m*d, covariates..., cluster dummies (K-1)]
    cols = [np.ones(N), m, d, m * d]
    n_cov = 0
    if covariates is not None:
        X_cov = np.asarray(covariates, dtype=np.float64)
        if X_cov.ndim == 1:
            X_cov = X_cov[:, None]
        if X_cov.shape[0] != N:
            raise ValueError("covariates must have one row per observation")
        cols.append(X_cov)
        n_cov = X_cov.shape[1]
    _, c_inv = np.unique(c, return_inverse=True)
    n_clusters = int(c_inv.max()) + 1
    if n_clusters > 1:                       # one-hot dummies, first cluster
        # dropped (absorbed by the intercept) to avoid perfect collinearity
        cols.append(np.eye(n_clusters, dtype=np.float64)[c_inv][:, 1:])
    X = np.column_stack(cols)

    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    slopes = {0: float(beta[2]), 1: float(beta[2] + beta[3])}

    # ---- beta3 CI: wild cluster bootstrap on per-graph aggregated residuals.
    # OLS linearity: perturbing residuals by cluster wild weights shifts
    # beta3 by sum_c w_c * sum_{g in c} u_g with u_g = sum_{i in g} a3_i e_i,
    # a3 = row 3 of (X'X)^-1 X'; sum_g u_g = a' e = 0 by orthogonality.
    ci_lo, ci_hi, se3, ci_method = float("nan"), float("nan"), float("nan"), "wild_cluster_bootstrap"
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        a3 = (XtX_inv @ X.T)[3]                       # beta3 influence row
        if graph_ids is None:
            sig = [(int(c_inv[i]), round(float(d[i]), 6))
                   for i in range(N)]
            if n_cov:
                sig = [(s[0], s[1], tuple(np.round(X_cov[i], 6).tolist()))
                       for i, s in enumerate(sig)]
            uniq = {s: j for j, s in enumerate(dict.fromkeys(sig))}
            g_idx = np.array([uniq[s] for s in sig], dtype=np.int64)
        else:
            _, g_idx = np.unique(np.asarray(graph_ids).ravel(),
                                 return_inverse=True)
        n_graphs = int(g_idx.max()) + 1
        # per-graph aggregated residual influence u_g, passed to
        # wild_cluster_bootstrap scaled so its /N equals the OLS shift
        u_g = np.bincount(g_idx, weights=a3 * resid, minlength=n_graphs)
        g_cluster = np.bincount(g_idx, weights=c_inv,
                                minlength=n_graphs).astype(np.int64) // \
            np.maximum(np.bincount(g_idx, minlength=n_graphs), 1)
        boot = wild_cluster_bootstrap(u_g * n_graphs, g_cluster,
                                      B=B, alpha=alpha, rng=rng)
        # mean(u_g * G) = 0 by OLS orthogonality, so the bootstrap band is
        # centered on zero and shifts directly onto beta3
        ci_lo = float(beta[3] + boot["lower"])
        ci_hi = float(beta[3] + boot["upper"])
        se3 = (ci_hi - ci_lo) / (2.0 * _norm_isf(alpha / 2.0))
    except (np.linalg.LinAlgError, ValueError, IndexError):
        # singular design or degenerate grouping -> cluster-robust SEs below
        ci_method = "cluster_sandwich_fallback"

    if ci_method != "cluster_sandwich_fallback" and not np.isfinite(ci_lo):
        ci_method = "cluster_sandwich_fallback"

    if ci_method == "cluster_sandwich_fallback":
        # cluster-robust sandwich SEs: V = (X'X)^-1 (sum_c X_c' e_c e_c' X_c) (X'X)^-1
        XtX_inv = np.linalg.pinv(X.T @ X)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for cl in range(n_clusters):
            mask = c_inv == cl
            Xc = X[mask]
            score = Xc.T @ resid[mask]
            meat += np.outer(score, score)
        V = XtX_inv @ meat @ XtX_inv
        se3 = float(np.sqrt(max(V[3, 3], 0.0)))
        z = beta[3] / se3 if se3 > 0 else 0.0
        ci_lo, ci_hi = float(beta[3] - _norm_isf(alpha / 2.0) * se3), \
                       float(beta[3] + _norm_isf(alpha / 2.0) * se3)

    if not np.isfinite(se3) or se3 <= 0:
        p_value = 1.0 if beta[3] == 0 else 0.0
    else:
        p_value = float(min(2.0 * _norm_sf(abs(beta[3]) / se3), 1.0))

    return {
        "beta0": float(beta[0]), "beta1": float(beta[1]),
        "beta2": float(beta[2]), "beta3": float(beta[3]),
        "slopes": slopes,
        "beta3_ci": (ci_lo, ci_hi),
        "p_value": p_value,
        "se_beta3": float(se3),
        "ci_method": ci_method,
        "n_obs": int(N),
        "n_clusters": int(n_clusters),
    }


# ---------------------------------------------------------------------------
# Cluster-level Spearman (§5.1, last row of the robustness list)
# ---------------------------------------------------------------------------

def _ranks_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged), numpy-only Spearman preprocessing."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # average ties
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def cluster_spearman(deltas, dist_z, cluster_ids,
                     n_permutations: int = 10_000,
                     rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Cluster-level Spearman rho between the mean per-graph Δ and the mean
    z-scored distance across provenance clusters, with a two-sided permutation
    test (§5.1: rho = 0.55, permutation p = 0.09 -- positive but not
    conventionally significant with 11 units, hence 'consistent with
    entanglement rather than a decisive test').

    Spearman is ranks + Pearson in pure numpy; scipy is not required.

    Returns
    -------
    dict(rho, p_value, n_clusters)
    """
    d = np.asarray(deltas, dtype=np.float64).ravel()
    z = np.asarray(dist_z, dtype=np.float64).ravel()
    c = np.asarray(cluster_ids).ravel()
    if not (d.size == z.size == c.size):
        raise ValueError("deltas, dist_z and cluster_ids must align")
    _, c_inv = np.unique(c, return_inverse=True)
    K = int(c_inv.max()) + 1
    n_c = np.bincount(c_inv, minlength=K).astype(np.float64)
    mean_d = np.bincount(c_inv, weights=d, minlength=K) / n_c
    mean_z = np.bincount(c_inv, weights=z, minlength=K) / n_c

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 0 else 0.0

    rho = _pearson(_ranks_avg(mean_d), _ranks_avg(mean_z))
    rng = np.random.default_rng() if rng is None else rng
    r_d, r_z = _ranks_avg(mean_d), _ranks_avg(mean_z)
    count = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(K)
        if abs(_pearson(r_d, r_z[perm])) >= abs(rho):
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return {"rho": float(rho), "p_value": float(p), "n_clusters": int(K)}
