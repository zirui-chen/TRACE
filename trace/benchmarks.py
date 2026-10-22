"""
Benchmark manifest for the 54 zero-shot graphs of the official ULTRA
suite (Table 5 of the paper), loaded with the official loaders.

FB15k-237, WN18RR and CoDEx-Medium are the pretraining graphs of
ULTRA-Real and are excluded from evaluation.

Pre-specified slices (§4.1, fixed before training):
    source-proximal (22): derived from FB15k-237, WordNet, or CoDEx,
        including FBNELL;
    source-disjoint (32): the other graphs;
    external-hard  (14): lowest quartile by published ULTRA(3g) MRR.

Primary inference uses a hierarchical wild cluster bootstrap over the
11 provenance clusters (six multi-graph families plus five singleton
graphs), with seeds resampled within methods (Appendix C).

Statistics follow the official ULTRA dataset tables: training-graph
statistics for transductive (T) settings and test/inference-graph
statistics for inductive settings; |R| counts canonical relations.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

import numpy as np

# ---------------------------------------------------------------------------
# Per-graph manifest: Table 5.
#   setting:  'T' transductive | 'E' inductive-e | 'ER' inductive-e,r
#   family:   provenance family (singletons carry their own id)
#   proximal: source-proximal flag (derived from FB15k-237/WordNet/CoDEx)
#   ext_hard: external-hard flag (lowest quartile by published 3g MRR)
#   mrr_pub / mrr_ultra_real / mrr_trace: 5-seed means (repro/TRACE),
#   published column is the official ULTRA(3g) MRR.
#   sd_delta: seed SD of delta.
# ---------------------------------------------------------------------------

_GRAPHS_RAW = [
    # name,          set, fam,    prox, eh,   |V|,   |R|,  triples,  pub,   repro, trace,  sd
    ("CoDEx-S",       "T", "WD",   True,  False,   2034,   42,    32888, .472, .469, .443, .005),
    ("CoDEx-L",       "T", "WD",   True,  False,  77951,   69,   551193, .338, .336, .331, .004),
    ("NELL-995",      "T", "NELL", False, False,  74536,  200,   149678, .406, .409, .397, .005),
    ("YAGO3-10",      "T", "YAGO", False, False, 123182,   37,  1079040, .451, .455, .448, .004),
    ("WDsinger",      "T", "WD",   False, False,  10282,  135,    16142, .382, .386, .389, .006),
    ("NELL23k",       "T", "NELL", False, True,   22925,  200,    25445, .239, .240, .265, .006),
    ("FB237_10",      "T", "FB",   True,  True,   11512,  237,    27211, .248, .247, .253, .004),
    ("FB237_20",      "T", "FB",   True,  False,  13166,  237,    54423, .272, .270, .267, .004),
    ("FB237_50",      "T", "FB",   True,  False,  14149,  237,   136057, .324, .323, .319, .003),
    ("DBpedia100k",   "T", "DBp",  False, False,  99604,  470,   597572, .398, .400, .396, .005),
    ("AristoV4",      "T", "Aristo", False, True,  44949, 1605,   242567, .182, .181, .209, .007),
    ("ConceptNet100k","T", "CN",   False, True,   78334,   34,   100000, .082, .082, .118, .008),
    ("Hetionet",      "T", "Hetio", False, True,  45158,   24,  2025177, .257, .258, .291, .007),
    ("FB v1",         "E", "FB",   True,  False,   1093,  180,     1993, .498, .494, .456, .007),
    ("FB v2",         "E", "FB",   True,  False,   1660,  200,     4145, .512, .510, .479, .006),
    ("FB v3",         "E", "FB",   True,  False,   2501,  215,     7406, .491, .489, .455, .006),
    ("FB v4",         "E", "FB",   True,  False,   3051,  219,    11714, .486, .480, .458, .005),
    ("WN v1",         "E", "WN",   True,  False,    922,    9,     1618, .648, .652, .610, .009),
    ("WN v2",         "E", "WN",   True,  False,   2757,   10,     4011, .663, .661, .620, .008),
    ("WN v3",         "E", "WN",   True,  False,   5084,   11,     6327, .376, .374, .346, .007),
    ("WN v4",         "E", "WN",   True,  False,   7084,    9,    12334, .611, .612, .573, .007),
    ("NELL v1",       "E", "NELL", False, False,    225,   14,      833, .712, .706, .685, .012),
    ("NELL v2",       "E", "NELL", False, False,   2086,   88,     4586, .567, .561, .575, .006),
    ("NELL v3",       "E", "NELL", False, False,   3566,  142,     8048, .563, .558, .560, .005),
    ("NELL v4",       "E", "NELL", False, False,   2795,   76,     7073, .469, .465, .483, .007),
    ("ILPC-small",    "E", "WD",   False, False,   6653,   48,    20960, .302, .296, .311, .005),
    ("ILPC-large",    "E", "WD",   False, False,  29246,   65,    77044, .290, .285, .290, .004),
    ("HM 1k",         "E", "WN",   True,  True,    9899,   11,    18638, .059, .052, .067, .006),
    ("HM 3k",         "E", "WN",   True,  True,   19218,   11,    38285, .037, .038, .050, .005),
    ("HM 5k",         "E", "WN",   True,  True,   23792,   11,    48425, .034, .032, .050, .005),
    ("Indigo-BM",     "E", "WN",   True,  False,  14775,  229,   250195, .440, .439, .439, .004),
    ("FB-25",         "ER","FB",   True,  False,   4097,  216,    17147, .388, .384, .354, .006),
    ("FB-50",         "ER","FB",   True,  False,   4445,  205,    11636, .338, .335, .318, .005),
    ("FB-75",         "ER","FB",   True,  False,   2792,  186,     9316, .403, .398, .362, .007),
    ("FB-100",        "ER","FB",   True,  False,   2624,   77,     6987, .449, .439, .406, .007),
    ("WK-25",         "ER","WD",   False, False,   3228,   74,     3391, .316, .313, .337, .007),
    ("WK-50",         "ER","WD",   False, True,    9328,   93,     9672, .166, .160, .207, .010),
    ("WK-75",         "ER","WD",   False, False,   2722,   65,     3430, .365, .365, .368, .006),
    ("WK-100",        "ER","WD",   False, True,   12136,   37,    13487, .164, .158, .210, .011),
    ("NL-0",          "ER","NELL", False, False,   2026,  112,     2287, .342, .341, .369, .008),
    ("NL-25",         "ER","NELL", False, False,   2146,  120,     2230, .395, .392, .396, .006),
    ("NL-50",         "ER","NELL", False, False,   2335,  119,     2576, .407, .406, .404, .006),
    ("NL-75",         "ER","NELL", False, False,   1578,  116,     1818, .368, .369, .385, .007),
    ("NL-100",        "ER","NELL", False, False,   1709,   53,     2378, .471, .471, .462, .006),
    ("MT1 tax",       "ER","MTDEA", False, True,  10000,    9,    16526, .224, .223, .267, .009),
    ("MT1 health",    "ER","MTDEA", False, False, 10000,    7,    14110, .298, .300, .331, .008),
    ("MT2 org",       "ER","MTDEA", False, True,  10000,   11,    21976, .095, .094, .155, .012),
    ("MT2 sci",       "ER","MTDEA", False, False, 10000,   16,    14852, .258, .255, .283, .007),
    ("MT3 art",       "ER","MTDEA", False, False, 10000,   45,    28023, .259, .259, .286, .007),
    ("MT3 infra",     "ER","MTDEA", False, False, 10000,   27,    21646, .619, .616, .647, .008),
    ("MT4 sci",       "ER","MTDEA", False, True,  10000,   42,    12516, .274, .272, .301, .007),
    ("MT4 health",    "ER","MTDEA", False, False, 10000,   20,    15337, .624, .632, .665, .008),
    ("Metafam",       "ER","ISDEA", False, True,    656,   28,     7257, .238, .234, .305, .013),
    ("FBNELL",        "ER","ISDEA", True,  False,  4752,  183,    10685, .485, .473, .459, .006),
]


class GraphSpec:
    __slots__ = ("name", "setting", "family", "proximal", "ext_hard",
                 "n_ent", "n_rel", "triples",
                 "mrr_pub", "mrr_ultra_real", "mrr_trace", "sd_delta")

    def __init__(self, row):
        (self.name, self.setting, self.family, self.proximal, self.ext_hard,
         self.n_ent, self.n_rel, self.triples,
         self.mrr_pub, self.mrr_ultra_real, self.mrr_trace,
         self.sd_delta) = row

    @property
    def delta(self) -> float:
        return round(self.mrr_trace - self.mrr_ultra_real, 3)

    def __repr__(self):
        return (f"GraphSpec({self.name!r}, set={self.setting}, "
                f"fam={self.family}, prox={self.proximal}, eh={self.ext_hard})")


# name -> GraphSpec, in Table 5 order
GRAPHS: Dict[str, GraphSpec] = {r[0]: GraphSpec(r) for r in _GRAPHS_RAW}

BENCHMARK_KGS: List[str] = list(GRAPHS.keys())

def setting_of(kg: str) -> str:        return GRAPHS[kg].setting
def family_of(kg: str) -> str:         return GRAPHS[kg].family
def is_proximal(kg: str) -> bool:      return GRAPHS[kg].proximal
def is_ext_hard(kg: str) -> bool:      return GRAPHS[kg].ext_hard

KGS_BY_SETTING: Dict[str, List[str]] = {
    s: [k for k in BENCHMARK_KGS if GRAPHS[k].setting == s]
    for s in ("T", "E", "ER")
}

# ---------------------------------------------------------------------------
# Pre-specified slices (§4.1)
# ---------------------------------------------------------------------------

PROXIMAL_KGS: FrozenSet[str]   = frozenset(k for k in BENCHMARK_KGS if GRAPHS[k].proximal)
DISJOINT_KGS: FrozenSet[str]   = frozenset(k for k in BENCHMARK_KGS if not GRAPHS[k].proximal)
EXTERNAL_HARD_KGS: FrozenSet[str] = frozenset(k for k in BENCHMARK_KGS if GRAPHS[k].ext_hard)

assert len(PROXIMAL_KGS) == 22
assert len(DISJOINT_KGS) == 32
assert len(EXTERNAL_HARD_KGS) == 14
# 10 of the 14 external-hard graphs are source-disjoint; the other four
# (HM 1k/3k/5k, FB237_10) are source-proximal (Appendix C).
assert len(EXTERNAL_HARD_KGS & PROXIMAL_KGS) == 4

SLICES: Dict[str, FrozenSet[str]] = {
    "all":      frozenset(BENCHMARK_KGS),
    "disjoint": DISJOINT_KGS,
    "proximal": PROXIMAL_KGS,
    "ext_hard": EXTERNAL_HARD_KGS,
}

# ---------------------------------------------------------------------------
# Provenance clusters for the wild cluster bootstrap (Appendix C)
#   six multi-graph families: FB (11), WN (8), NELL (11), WD (9),
#   MTDEA (8), ISDEA (2); five singleton graphs.
# HM and Indigo-BM are built from WordNet11 and belong to the WN family.
# FBNELL mixes FB15k-237 and NELL-995 but is ISDEA-derived; families are
# defined strictly by source provenance, not content overlap.
# ---------------------------------------------------------------------------

_MULTI_FAMILIES = ("FB", "WN", "NELL", "WD", "MTDEA", "ISDEA")

PROVENANCE_CLUSTERS: List[List[str]] = [
    [k for k in BENCHMARK_KGS if GRAPHS[k].family == fam]
    for fam in _MULTI_FAMILIES
]
PROVENANCE_CLUSTERS += [
    [k for k in BENCHMARK_KGS if GRAPHS[k].family not in _MULTI_FAMILIES]
]
# flatten singletons into their own clusters
_singleton_rows = PROVENANCE_CLUSTERS.pop()
PROVENANCE_CLUSTERS += [[k] for k in _singleton_rows]

assert len(PROVENANCE_CLUSTERS) == 11
assert sum(len(c) for c in PROVENANCE_CLUSTERS) == 54

def cluster_of(kg: str) -> int:
    for i, cluster in enumerate(PROVENANCE_CLUSTERS):
        if kg in cluster:
            return i
    raise KeyError(kg)

def cluster_ids(kgs) -> np.ndarray:
    return np.array([cluster_of(k) for k in kgs], dtype=np.int64)

# ---------------------------------------------------------------------------
# Arms of the corpus x supervision factorial (§4.1, Fig. 2d)
# ---------------------------------------------------------------------------

ARMS = ("ultra_real", "trace_real", "ultra_syn", "trace")
ARM_LABELS = {
    "ultra_real": "ULTRA-Real (3 real, standard)",
    "trace_real": "TRACE-Real (3 real, premise-faithful)",
    "ultra_syn":  "ULTRA-Syn (1,800 syn., standard)",
    "trace":      "TRACE (1,800 syn., premise-faithful)",
}

ABLATION_ARMS = ("ablation_a", "ablation_b", "ablation_c", "ablation_d")
CONTROL_ARMS = (
    "control_3cycled", "control_edge_matched", "control_many_small",
    "control_real_subgraphs", "control_mixture", "control_compute_matched",
)

# ---------------------------------------------------------------------------
# Expected values for scripts/verify_numbers.py (paper tables)
# ---------------------------------------------------------------------------

# Table 1: Source x supervision factorial. Means over unrounded per-graph
# values; ± are 95% seed intervals of the aggregate; brackets are wild
# cluster bootstrap 95% CIs over the 11 provenance clusters.
EXPECTED_TABLE1 = {
    "ultra_real": {"all": .364, "all_sd": .004, "disjoint": .348, "proximal": .387, "ext_hard": .161},
    "trace_real": {"all": .366, "all_sd": .004, "disjoint": .352, "proximal": .387, "ext_hard": .165},
    "ultra_syn":  {"all": .355, "all_sd": .005, "disjoint": .354, "proximal": .356, "ext_hard": .177},
    "trace":      {"all": .369, "all_sd": .004, "disjoint": .369, "proximal": .369, "ext_hard": .195},
}
EXPECTED_PIPELINE_DELTAS = {          # TRACE - ULTRA-Real (Table 1 row 5)
    "all":       (+.005, (-.008, +.017)),
    "disjoint":  (+.020, (+.009, +.032)),
    "proximal":  (-.018, (-.027, -.010)),
    "ext_hard":  (+.034, (+.019, +.050)),
}
EXPECTED_CORPUS_EFFECT_GIVEN_PF = {   # corpus effect | premise-faithful sup.
    "all": +.003, "disjoint": +.017, "proximal": -.018, "ext_hard": +.030,
}
EXPECTED_INTERACTION = {              # corpus x supervision interaction
    "all": +.012, "disjoint": +.011, "proximal": +.013, "ext_hard": +.014,
}
EXPECTED_INTERACTION_CI = (+.004, +.020)   # on the aggregate

# Table 2: non-inferiority of TRACE vs. ULTRA-Real, one-sided 95% lower bounds
EXPECTED_TABLE2 = {
    "wild_cluster_bootstrap": -.006,    # primary; passes delta=0.01, not 0.005
    "mixed_model":              -.005,  # borderline at delta=0.005
    "graph_level_bootstrap":    -.002,  # ignores clusters; passes both
}

# Table 3: 2x2 generator ablation (mean MRR over the 54 graphs, 5 seeds)
EXPECTED_TABLE3 = {
    "ablation_a": {"all": .369, "all_sd": .004, "ext_hard": .195},
    "ablation_b": {"all": .316, "all_sd": .006, "ext_hard": .157},
    "ablation_c": {"all": .338, "all_sd": .005, "ext_hard": .174},
    "ablation_d": {"all": .290, "all_sd": .006, "ext_hard": .148},
}
EXPECTED_TABLE3_CONTRASTS = {
    # contrast: (all, ext_hard, p_all)  -- hierarchical cluster bootstrap, uncorrected
    "A_vs_B": (+.053, +.038, "<0.001"),
    "C_vs_D": (+.048, +.026, "<0.001"),
    "A_vs_C": (+.031, +.021, "<0.001"),
    "B_vs_D": (+.026, +.009, "<0.001"),
    "interaction": (+.005, +.012, "0.21"),
    "A_vs_D": (+.079, +.047, "<0.001"),
    "B_vs_ULTRA": (-.048, -.004, "<0.001"),
}

# Table 4: corpus-level controls
EXPECTED_TABLE4 = {
    "ultra_real":            {"n_kg": 3,    "edges": "545K",  "all": .364, "disjoint": .348, "proximal": .387, "ext_hard": .161},
    "trace":                 {"n_kg": 1800, "edges": "128M",  "all": .369, "disjoint": .369, "proximal": .369, "ext_hard": .195},
    "control_3cycled":       {"n_kg": 3,    "edges": "213K",  "all": .339, "disjoint": .357, "proximal": .312, "ext_hard": .179},
    "control_edge_matched":  {"n_kg": 8,    "edges": "545K",  "all": .349, "disjoint": .361, "proximal": .331, "ext_hard": .183},
    "control_many_small":    {"n_kg": 1800, "edges": "545K",  "all": .356, "disjoint": .364, "proximal": .344, "ext_hard": .187},
    "control_real_subgraphs":{"n_kg": 1800, "edges": "<=545K","all": .361, "disjoint": .351, "proximal": .375, "ext_hard": .169},
    "control_compute_matched":{"n_kg": 1800,"edges": "128M",  "all": .367, "disjoint": .367, "proximal": .367, "ext_hard": .192},
    "control_mixture":       {"n_kg": 1803, "edges": "128M",  "all": .376, "disjoint": .370, "proximal": .384, "ext_hard": .193},
}

# Headline counts (Appendix C)
EXPECTED_WINS = 30          # TRACE higher on 30/54 (sign test p = 0.25)
EXPECTED_TIE_COUNTS = (25, 11, 18)   # wins/ties/losses under the +/-0.005 band
EXPECTED_HITS = {"trace": {"h1": .276, "h10": .523},
                 "ultra_real": {"h1": .271, "h10": .517}}

# Per-setting means, T / E / ER: reproduction vs. published
EXPECTED_SETTING_MEANS = {
    "ultra_real": {"T": .312, "E": .428, "ER": .343},
    "published":  {"T": .312, "E": .431, "ER": .345},
}

# Structural-distance interaction (Eq. 5, §5.1)
EXPECTED_BETA3 = {
    "min_of_3":  (+.011, (+.004, +.019)),
    "fb_only":   (+.012, (+.004, +.020)),
    "mean_of_3": (+.011, (+.003, +.019)),
    "in_degree": (+.010, None),
    "out_degree": (+.011, None),
}
EXPECTED_SLOPES = {"ultra_real": -.021, "trace": -.010}   # MRR per SD
EXPECTED_CLUSTER_SPEARMAN = (0.55, 0.09)                  # rho, permutation p
