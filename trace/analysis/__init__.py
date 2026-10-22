"""
Statistical analysis for TRACE (paper §5.1, §5.2, Appendices C-F).

Public API
----------
bootstrap (Appendix C, Table 2):
    webb_weights, wild_cluster_bootstrap (primary procedure),
    graph_level_bootstrap, mixed_model_lower_bound, noninferiority_test,
    wins_ties_losses, sign_test, cumulative_delta (Appendix D, Figure 7).

distance (Appendix E, Eq. 5, Figures 4 / 6):
    degree_histogram, js_divergence_degree, relation_graph_distance,
    structural_distances, fit_interaction, cluster_spearman.

probe (§5.2, Appendix F, Table 6, Figure 3):
    ridge_probe_cv, auroc, graph_statistics, run_probe_suite,
    extract_relation_embeddings.

scipy is listed in requirements.txt but is never imported at module scope,
so all submodules import with numpy alone.
"""

from __future__ import annotations

from trace.analysis.bootstrap import (
    cumulative_delta,
    graph_level_bootstrap,
    mixed_model_lower_bound,
    noninferiority_test,
    sign_test,
    webb_weights,
    wild_cluster_bootstrap,
    wins_ties_losses,
)
from trace.analysis.distance import (
    cluster_spearman,
    degree_histogram,
    fit_interaction,
    js_divergence_degree,
    relation_graph_distance,
    structural_distances,
)
from trace.analysis.probe import (
    auroc,
    extract_relation_embeddings,
    graph_statistics,
    ridge_probe_cv,
    run_probe_suite,
)

__all__ = [
    # bootstrap
    "webb_weights",
    "wild_cluster_bootstrap",
    "graph_level_bootstrap",
    "mixed_model_lower_bound",
    "noninferiority_test",
    "wins_ties_losses",
    "sign_test",
    "cumulative_delta",
    # distance
    "degree_histogram",
    "js_divergence_degree",
    "relation_graph_distance",
    "structural_distances",
    "fit_interaction",
    "cluster_spearman",
    # probe
    "ridge_probe_cv",
    "auroc",
    "graph_statistics",
    "run_probe_suite",
    "extract_relation_embeddings",
]
