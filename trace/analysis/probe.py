"""
Representation probes (§5.2, Appendix F) and graph statistics (Figure 3).

Ridge probes on graph-level means of R_q predict four query-irrelevant
statistics: degree exponent, density, relation-frequency entropy and
relation-graph density. Probes use provenance-grouped 5-fold CV with nested
regularization selection -- a provenance cluster is never split across folds
(the 54 benchmark graphs plus 300 real subgraphs and 300 out-of-prior
synthetic graphs form the probe set, Appendix F).

Reference values (Table 6, cross-validated R^2, mean over 5 seeds, SD <= 0.03):
    Target                    ULTRA-Real  TRACE  Random
    Degree exponent               0.61     0.34    0.22
    Density                       0.58     0.31    0.25
    Rel.-frequency entropy        0.49     0.29    0.18
    Rel.-graph density            0.55     0.33    0.24
    Heads a composition rule (AUROC) 0.66  0.74    0.55
Aggregate decodability: mean R^2 0.56 (ULTRA-Real) vs 0.32 (TRACE), 0.22 for
a randomly initialized model -- query-irrelevant statistics are more
decodable from ULTRA-Real, rule-role information more decodable from TRACE.
The probes remain correlational: lower decodability does not prove lower
reliance (§5.2).

graph_statistics implements the Figure 3 axes: degree exponent via the
Clauset-Shalizi-Newman discrete MLE with k_min = 1, density, normalized
relation-frequency entropy, and binary relation-graph density (Lift(E) per
the four interaction types).

Pure numpy + closed-form ridge; torch and trace.generator.lift are imported
lazily inside functions only.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "ridge_probe_cv",
    "auroc",
    "graph_statistics",
    "run_probe_suite",
    "extract_relation_embeddings",
]

STAT_TARGETS = ("degree_exponent", "density", "rel_freq_entropy",
                "rel_graph_density")


# ---------------------------------------------------------------------------
# Ridge probe with provenance-grouped K-fold CV (Appendix F)
# ---------------------------------------------------------------------------

def _group_folds(groups: np.ndarray, n_folds: int,
                 rng: np.random.Generator):
    """Split unique groups into <= n_folds folds; a group is never split."""
    uniq = np.unique(groups)
    order = rng.permutation(uniq)
    folds = [np.zeros(0, dtype=groups.dtype) for _ in range(min(n_folds, uniq.size))]
    for i, grp in enumerate(order):
        folds[i % len(folds)] = np.append(folds[i % len(folds)], grp)
    return [np.isin(groups, f) for f in folds if f.size]


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge with intercept, solved on centered columns."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc = X - xm
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ (y - ym))
    return w, ym - xm @ w


def _ridge_cv_predict(X, y, groups, alphas, n_folds, rng):
    """Nested-regularization group CV: outer folds estimate held-out
    predictions; the penalty is re-selected on inner group folds of each
    outer training set. Returns pooled held-out predictions and the
    most-selected alpha."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    if not (X.shape[0] == y.size == groups.size):
        raise ValueError("X, y and groups must share the leading dimension")
    alphas = tuple(alphas)
    outer = _group_folds(groups, n_folds, rng)
    preds = np.zeros_like(y)
    chosen = []
    for val_mask in outer:
        tr_idx = np.where(~val_mask)[0]
        Xtr, ytr, gtr = X[tr_idx], y[tr_idx], groups[tr_idx]
        n_inner = min(n_folds, len(np.unique(gtr)))
        best_alpha, best_sse = alphas[0], np.inf
        if n_inner >= 2:
            inner = _group_folds(gtr, n_inner, rng)
            for a in alphas:
                sse = 0.0
                for iv in inner:
                    itr = np.where(~iv)[0]
                    if itr.size == 0 or iv.sum() == 0:
                        continue
                    w, b = _fit_ridge(Xtr[itr], ytr[itr], a)
                    pred = Xtr[iv] @ w + b
                    sse += float(((ytr[iv] - pred) ** 2).sum())
                if sse < best_sse - 1e-15:
                    best_sse, best_alpha = sse, a
        chosen.append(best_alpha)
        w, b = _fit_ridge(Xtr, ytr, best_alpha)
        preds[val_mask] = X[val_mask] @ w + b
    best = max(set(chosen), key=chosen.count) if chosen else alphas[0]
    return preds, best


def ridge_probe_cv(X, y, groups,
                   alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1, 10, 100),
                   n_folds: int = 5,
                   rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Provenance-grouped ridge probe with nested regularization selection
    (Appendix F).

    Closed-form ridge (pure numpy). K-fold CV never splits a provenance group
    across folds; the penalty alpha is selected by an inner group CV on each
    outer training set. R^2 = 1 - SS_res / SS_tot on the pooled held-out
    folds (SS_tot around the held-out mean).

    Parameters
    ----------
    X       : (G, d) features -- graph-level mean R_q vectors stacked per graph.
    y       : (G,) target statistic per graph.
    groups  : (G,) provenance group id per graph (trace.benchmarks.cluster_ids).
    alphas  : ridge penalty grid.
    n_folds : number of group folds.
    rng     : numpy Generator (deterministic default_rng(0) when None).

    Returns
    -------
    dict(r2, best_alpha)
    """
    rng = np.random.default_rng(0) if rng is None else rng
    y = np.asarray(y, dtype=np.float64).ravel()
    preds, best_alpha = _ridge_cv_predict(X, y, groups, alphas, n_folds, rng)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"r2": float(r2), "best_alpha": float(best_alpha)}


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def auroc(scores, labels) -> float:
    """Rank-based AUROC (Mann-Whitney U / trapezoid over the ROC curve),
    pure numpy with average ranks for ties. Returns nan for degenerate
    label sets."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(np.int64)
    if s.size != y.size:
        raise ValueError("scores and labels must align")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:                              # average tied ranks
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Graph statistics (Figure 3 axes; Table 5 conventions)
# ---------------------------------------------------------------------------

def _unique_canonical_edges(triples) -> np.ndarray:
    t = np.asarray(triples, dtype=np.int64)
    if t.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(t, axis=0)


def graph_statistics(triples, n_entities: int,
                     n_relations: int) -> Dict[str, float]:
    """Figure 3 statistic vector of one graph.

    - degree_exponent: Clauset-Shalizi-Newman discrete MLE with k_min = 1,
      alpha_hat = 1 + n / Σ ln(d_i) on total degrees >= 1 (each canonical
      triple counted once at each endpoint, no inverses).
    - density: |E| / (|V|(|V|-1)) over unique canonical edges.
    - rel_freq_entropy: H(π) / log|R| of the relation-frequency distribution.
    - rel_graph_density: edge density of Lift(E) per the four interaction
      types of the binary relation graph, averaged (trace.generator.lift is
      imported lazily inside this function).

    Returns dict(degree_exponent, density, rel_freq_entropy, rel_graph_density).
    """
    from trace.analysis.distance import _type_pairs_and_degrees, _endpoint_degrees

    t = _unique_canonical_edges(triples)
    n_ent = int(n_entities)

    deg = _endpoint_degrees(t, n_ent, "total")
    pos = deg[deg >= 1].astype(np.float64)
    if pos.size > 0:
        # CSN MLE with x_min = 1: 1 + n / sum(ln d_i)  (ln 1 = 0 drops out)
        log_sum = float(np.log(pos[pos > 1]).sum())
        degree_exponent = float(1.0 + pos.size / log_sum) if log_sum > 0 else float("nan")
    else:
        degree_exponent = float("nan")

    n_edges = t.shape[0]
    density = float(n_edges / (n_ent * (n_ent - 1))) if n_ent > 1 else 0.0

    if t.shape[0] > 0 and n_relations > 1:
        counts = np.bincount(t[:, 1]).astype(np.float64)
        pi = counts[counts > 0] / counts.sum()
        rel_freq_entropy = float(-(pi * np.log(pi)).sum() / np.log(n_relations))
    else:
        rel_freq_entropy = 0.0

    rel_graph_density = 0.0
    if n_edges > 0:
        from trace.generator import lift          # lazy: heavy module
        gr = lift(t, n_relations)
        n_r = int(gr["num_nodes"])
        if n_r > 1:
            dens = []
            for typ in range(4):
                pairs, _ = _type_pairs_and_degrees(gr, typ)
                dens.append(pairs.size / (n_r * (n_r - 1) / 2.0))
            rel_graph_density = float(np.mean(dens))

    return {
        "degree_exponent": degree_exponent,
        "density": density,
        "rel_freq_entropy": rel_freq_entropy,
        "rel_graph_density": rel_graph_density,
    }


# ---------------------------------------------------------------------------
# Table 6 probe suite
# ---------------------------------------------------------------------------

def run_probe_suite(embeddings: Dict[str, Dict[str, np.ndarray]],
                    stats_table: Dict[str, Dict[str, float]],
                    rule_role_labels: Optional[Dict[str, int]] = None,
                    alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1, 10, 100),
                    n_folds: int = 5,
                    rng: Optional[np.random.Generator] = None
                    ) -> Dict[str, Dict[str, Optional[float]]]:
    """Probe every model on the four Table 6 statistic targets (plus the
    rule-role AUROC when labels are given).

    Parameters
    ----------
    embeddings      : {model_name: {graph: 1-D mean R_q vector}}; model names
                      typically 'ultra_real', 'trace', 'random'.
    stats_table     : {graph: graph_statistics(...) row} keyed by the same
                      graph names; provenance groups are taken from
                      trace.benchmarks.cluster_ids.
    rule_role_labels: optional {graph: 0/1} -- whether the graph contains a
                      real relation heading a high-confidence composition
                      rule; predicted with the same group-CV ridge and scored
                      by AUROC on pooled held-out predictions.
    alphas, n_folds : ridge grid / fold count.
    rng             : numpy Generator.

    Returns
    -------
    Table 6-shaped dict:
        {"degree_exponent": {model: r2}, ..., "rel_graph_density": {...},
         "rule_role_auroc": {model: auroc or None}}
    Paper reference values (Table 6): ULTRA-Real ≈ .61/.58/.49/.55,
    TRACE ≈ .34/.31/.29/.33, random ≈ .22 mean R^2; rule-role AUROC .66
    (ULTRA-Real) vs .74 (TRACE).
    """
    from trace.benchmarks import cluster_ids

    rng = np.random.default_rng(0) if rng is None else rng
    out: Dict[str, Dict[str, Optional[float]]] = {t: {} for t in STAT_TARGETS}
    out["rule_role_auroc"] = {}

    for model, emb in embeddings.items():
        graphs = [g for g in emb if g in stats_table]
        if not graphs:
            for t in STAT_TARGETS:
                out[t][model] = None
            out["rule_role_auroc"][model] = None
            continue
        groups = cluster_ids(graphs)
        X = np.vstack([np.asarray(emb[g], dtype=np.float64).ravel()
                       for g in graphs])
        for t in STAT_TARGETS:
            vals = np.array([stats_table[g][t] for g in graphs], dtype=np.float64)
            keep = np.isfinite(vals)
            if keep.sum() >= 3 and np.unique(groups[keep]).size >= 2:
                out[t][model] = ridge_probe_cv(
                    X[keep], vals[keep], groups[keep],
                    alphas=alphas, n_folds=n_folds, rng=rng)["r2"]
            else:
                out[t][model] = None
        if rule_role_labels is not None:
            labeled = [g for g in graphs if g in rule_role_labels]
            if len(labeled) >= 3:
                y = np.array([rule_role_labels[g] for g in labeled],
                             dtype=np.float64)
                g_lab = cluster_ids(labeled)
                if 0 < y.sum() < y.size:
                    preds, _ = _ridge_cv_predict(X[[graphs.index(g) for g in labeled]],
                                                 y, g_lab, alphas, n_folds, rng)
                    out["rule_role_auroc"][model] = auroc(preds, y)
                else:
                    out["rule_role_auroc"][model] = None
            else:
                out["rule_role_auroc"][model] = None
        else:
            out["rule_role_auroc"][model] = None
    return out


# ---------------------------------------------------------------------------
# R_q extraction
# ---------------------------------------------------------------------------

def extract_relation_embeddings(model, kg_batch,
                                device: str = "cpu") -> Dict[str, np.ndarray]:
    """Per-KG mean R_q vectors, the probe features of §5.2.

    Accepts either of two sources:

    1. A model object exposing the ULTRA relation-parameterizer adapter
       ``R_q(gr_edge_index, gr_edge_type, num_relation_nodes,
       query_relations)`` (GNN_r(G_r, q) of §3.1). ``kg_batch`` is then
       {kg: {"gr_edge_index": (2, E), "gr_edge_type": (E,),
             "num_relation_nodes": int, "query_relations": (Q,)}} and the
       R_q outputs are averaged over the Q query relations.
    2. A path to a precomputed .npz (key = KG name, value = the stored 1-D
       mean R_q vector), for cached checkpoints. ``kg_batch`` is ignored.

    Parameters
    ----------
    model    : adapter object or .npz path.
    kg_batch : per-KG relation-graph / query payload (source 1 only).
    device   : torch device string for the model path.

    Returns
    -------
    {kg: 1-D mean R_q vector (np.float64)}.
    """
    if isinstance(model, (str, os.PathLike)):
        out = {}
        with np.load(os.fspath(model), allow_pickle=False) as npz:
            for key in npz.files:
                vec = np.asarray(npz[key], dtype=np.float64)
                if vec.ndim == 2:                    # stored per-query
                    vec = vec.mean(axis=0)
                out[str(key)] = vec.ravel()
        return out

    import torch                                    # lazy heavy import

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    out: Dict[str, np.ndarray] = {}
    for kg, spec in kg_batch.items():
        ei = torch.as_tensor(np.asarray(spec["gr_edge_index"]),
                             dtype=torch.long, device=device)
        et = torch.as_tensor(np.asarray(spec["gr_edge_type"]),
                             dtype=torch.long, device=device)
        qr = torch.as_tensor(np.asarray(spec["query_relations"]),
                             dtype=torch.long, device=device)
        n_r = int(spec["num_relation_nodes"])
        with torch.no_grad():
            R_q = model.R_q(ei, et, n_r, qr)
        if torch.is_tensor(R_q):
            arr = R_q.detach().float().cpu().numpy()
        else:
            arr = np.asarray(R_q, dtype=np.float64)
        if arr.ndim == 1:                            # already pooled
            out[kg] = arr
        else:                                        # (Q, d) -> mean over queries
            out[kg] = arr.mean(axis=0)
    return out
