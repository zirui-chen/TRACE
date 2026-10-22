"""
Factorized synthetic KG generator (Algorithm 1, §3.3, Appendix N).

The generator factorises the latent specification of a synthetic KG
into a rule axis T (which templates are present) and a statistics axis
S (topology parameters), drawn INDEPENDENTLY so that the corpus
disentangles rule content from graph statistics:

    p(T, S) = p(T) p(S),   (G, Q_train, Q_val, Q_test) ~ p(. | T, S),
    kept iff Acc(G) = 1.                                    (Eq. 4)

* Rule axis. Seven templates at three levels (Table 8); each template
  is included independently with probability 0.70 / 0.62 / 0.44 per
  level (expected 4.2 per KG).  Composition (T4) is the fallback if no
  template is drawn.  Each template gets its own fresh head relation
  on top of the n_rel non-head pool R0; body relations are drawn
  uniformly from R0 and may be shared across templates; heads are
  never bodies (non-recursive), so one derivation pass yields the full
  closure.  Symmetry uses its own head as its body.

* Statistics axis. n_ent ~ LogU(100, 30K), n_rel ~ LogU(6, 192),
  kappa ~ LogU(0.1, 10), gamma ~ U(1.5, 3.0) with a 15% point mass on
  uniform degree, mu ~ LogU(3, 50), eta ~ U(0, 0.5), nu ~ U(0, 0.10).
  Priors were frozen before real-KG evaluation (§3.3).

Premise-faithful splits (Eqs. 1-3): body edges B are permanently
observable and never targets; derived triples split 60/20/20 into
d_train / d_val / d_test, background triples 80/10/10 into
b_train / b_val / b_test; noise N is observable and may enter Q_train
but never Q_val / Q_test.  Leakage is asserted before returning.

Arm B (corpus with rules destroyed) keeps the task construction but
rewires each head relation's derived edges with 10 |E_r|
degree-preserving directed double-edge swaps (Appendix N, Corpus B),
leaving 0.4% of rewired edges rule-supported by chance.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .defaults import (
    DERIVED_CAP_PER_TEMPLATE,
    DERIVED_TRAIN_FRAC,
    DERIVED_VAL_FRAC,
    BG_TRAIN_FRAC,
    BG_VAL_FRAC,
    FALLBACK_TEMPLATE,
    LEVEL_PROB,
    MAX_OBS_EDGES,
    MIN_DERIVED_TRAIN,
    MIN_LCC_FRACTION,
    MIN_Q_VAL,
    NO_RULES_SWAP_FACTOR,
)

__all__ = [
    "TEMPLATES",
    "KGSplit",
    "SyntheticKGGenerator",
    "lift",
    "sample_real_subgraph",
]

# --------------------------------------------------------------------- #
# Rule catalog (Table 8)                                                 #
# --------------------------------------------------------------------- #

#: template id -> specification.  ``bodies`` is the number of body
#: relations drawn from the non-head pool; symmetry (T1) uses its own
#: head as its body and therefore draws zero.
TEMPLATES: Dict[str, Dict] = {
    "T1": dict(level=1, name="symmetry",    bodies=0,
               schema="r(x,y) -> r(y,x)"),
    "T2": dict(level=1, name="inversion",   bodies=1,
               schema="r1(x,y) -> r2(y,x)"),
    "T3": dict(level=1, name="subsumption", bodies=1,
               schema="r1(x,y) -> r2(x,y)"),
    "T4": dict(level=2, name="composition", bodies=2,
               schema="r1(x,z) ^ r2(z,y) -> r3(x,y)"),
    "T5": dict(level=2, name="intersection", bodies=2,
               schema="r1(x,y) ^ r2(x,y) -> r3(x,y)"),
    "T6": dict(level=3, name="long_chain",  bodies=3,
               schema="r1(x,w) ^ r2(w,z) ^ r3(z,y) -> r4(x,y)"),
    "T7": dict(level=3, name="branching",   bodies=2,
               schema="r1(x,z) ^ r2(y,z) -> r3(x,y)"),
}
TEMPLATE_IDS: List[str] = sorted(TEMPLATES)


# --------------------------------------------------------------------- #
# Small numeric helpers                                                  #
# --------------------------------------------------------------------- #

def _triple_keys(triples: np.ndarray, n_rel: int, n_ent: int) -> np.ndarray:
    """Injective int64 key per canonical (h, r, t) triple."""
    return (triples[:, 0].astype(np.int64) * n_rel
            + triples[:, 1].astype(np.int64)) * n_ent + triples[:, 2]


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw from LogU(lo, hi) (log-uniform)."""
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def _dedupe(triples: np.ndarray) -> np.ndarray:
    if len(triples) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(triples.astype(np.int64), axis=0)


def _empty3() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.int64)


def _concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [a for a in arrays if len(a)]
    if not arrays:
        return _empty3()
    return np.concatenate(arrays, axis=0)


def _lcc_fraction(triples: np.ndarray, n_ent: int) -> float:
    """Fraction of entities in the largest weakly-connected component."""
    if n_ent == 0:
        return 0.0
    if len(triples) == 0:
        return 0.0
    parent = np.arange(n_ent, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    h = triples[:, 0].astype(np.int64)
    t = triples[:, 2].astype(np.int64)
    for a, b in zip(h.tolist(), t.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.fromiter((find(int(v)) for v in range(n_ent)),
                        dtype=np.int64, count=n_ent)
    return float(np.bincount(roots).max() / n_ent)


def _hash_join(a: np.ndarray, b: np.ndarray, key_a: int, key_b: int,
               chunk: int = 4_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Inner join of edge blocks a and b on a[:, key_a] == b[:, key_b].

    Returns (a_row_indices, b_row_indices) grouped by the join key,
    materialised in chunks so a pathological fan-out cannot exhaust
    memory before the acceptance filter rejects the draw.
    """
    if len(a) == 0 or len(b) == 0:
        return _empty_idx(), _empty_idx()
    order_b = np.argsort(b[:, key_b], kind="stable")
    bkey = b[order_b, key_b]
    akey = a[:, key_a]
    lo = np.searchsorted(bkey, akey, side="left")
    hi = np.searchsorted(bkey, akey, side="right")
    counts = hi - lo
    keep = counts > 0
    a_rows = np.repeat(np.nonzero(keep)[0], counts[keep])
    starts = lo[keep]
    n_total = int(counts.sum())
    a_out = np.empty(n_total, dtype=np.int64)
    b_out = np.empty(n_total, dtype=np.int64)
    pos = 0
    for start in range(0, len(a_rows), chunk):
        stop = min(start + chunk, len(a_rows))
        seg_a = a_rows[start:stop]
        seg_s = starts[start:stop]
        seg_n = counts[keep][start:stop]
        offs = np.concatenate(([0], np.cumsum(seg_n)[:-1]))
        rep_n = np.repeat(seg_n, seg_n)
        rep_off = np.repeat(offs, seg_n)
        rep_s = np.repeat(seg_s, seg_n)
        a_out[pos + start:pos + stop] = seg_a
        b_out[pos + start:pos + stop] = order_b[rep_s + (np.arange(rep_n.size) - rep_off)]
        pos += stop - start
    return a_out, b_out


def _empty_idx() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


# --------------------------------------------------------------------- #
# Directed stub matching (Molloy-Reed, Appendix N step 3)                #
# --------------------------------------------------------------------- #

def _powerlaw_activities(rng: np.random.Generator, n_ent: int, gamma: float,
                         M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Shared entity in/out-activity sequences.

    a_out, a_in drawn i.i.d. from the discrete power law
    P(k) proportional to k^-gamma on {1, ..., ceil(sqrt(M))}
    (Appendix N step 3); the caller sets them to 1 under the
    15% uniform-degree option.
    """
    kmax = max(1, int(math.ceil(math.sqrt(max(M, 1)))))
    ks = np.arange(1, kmax + 1, dtype=np.float64)
    pmf = ks ** (-gamma)
    pmf /= pmf.sum()
    a_out = rng.choice(ks, size=n_ent, p=pmf)
    a_in = rng.choice(ks, size=n_ent, p=pmf)
    return a_out.astype(np.float64), a_in.astype(np.float64)


def _stub_matching_relation(rng: np.random.Generator, rel: int, m_r: int,
                            n_ent: int, a_out: np.ndarray,
                            a_in: np.ndarray) -> np.ndarray:
    """Directed stub matching for one relation: assign M_r head/tail
    stubs to entities proportional to the activity sequences, pair the
    two lists uniformly at random, then DROP (not resample) self-loops
    and duplicate triples -- on average 6.1% of stubs (§3.3, App. N).
    """
    if m_r <= 0:
        return _empty3()
    p_out = a_out / a_out.sum()
    p_in = a_in / a_in.sum()
    heads = rng.choice(n_ent, size=m_r, p=p_out)
    tails = rng.choice(n_ent, size=m_r, p=p_in)
    tails = rng.permutation(tails)
    rels = np.full(m_r, rel, dtype=np.int64)
    edges = np.stack([heads, rels, tails], axis=1).astype(np.int64)
    edges = edges[edges[:, 0] != edges[:, 2]]          # drop self-loops
    return _dedupe(edges)                               # drop duplicates


# --------------------------------------------------------------------- #
# Relation graph (Lift), shared with the reasoner and the probes         #
# --------------------------------------------------------------------- #

def lift(edges: np.ndarray, n_relations: int) -> Dict:
    """Binary relation graph G_r = Lift(E) as in ULTRA's get_rel_graph.

    Nodes are relations; four interaction types between relations i, j:
    0 = tail_i == head_j, 1 = head_i == head_j,
    2 = tail_i == tail_j, 3 = head_i == tail_j; plus a self-loop per
    relation.  Returned as ``{"edge_index", "edge_type", "num_nodes"}``.
    """
    edge_index, edge_type = [], []
    if len(edges):
        h = edges[:, 0].astype(np.int64)
        r = edges[:, 1].astype(np.int64)
        t = edges[:, 2].astype(np.int64)
        # group incident relations per entity, then emit all pairs
        order_h = np.lexsort((r, h))
        ent_h, rel_h = h[order_h], r[order_h]
        order_t = np.lexsort((r, t))
        ent_t, rel_t = t[order_t], r[order_t]
        bounds_h = np.searchsorted(ent_h, np.arange(0, _max_ent(edges) + 2))
        bounds_t = np.searchsorted(ent_t, np.arange(0, _max_ent(edges) + 2))
        for e in range(_max_ent(edges) + 1):
            hs = rel_h[bounds_h[e]:bounds_h[e + 1]]
            ts = rel_t[bounds_t[e]:bounds_t[e + 1]]
            if hs.size and ts.size:
                # tail_i == head_j  (type 0)
                ei = _cart(ts, hs)
                edge_index.append(ei)
                edge_type.append(np.full(ei.shape[1], 0, dtype=np.int64))
                # head_i == tail_j  (type 3)
                ei = _cart(hs, ts)
                edge_index.append(ei)
                edge_type.append(np.full(ei.shape[1], 3, dtype=np.int64))
            if hs.size:
                ei = _cart(hs, hs)
                edge_index.append(ei)
                edge_type.append(np.full(ei.shape[1], 1, dtype=np.int64))
            if ts.size:
                ei = _cart(ts, ts)
                edge_index.append(ei)
                edge_type.append(np.full(ei.shape[1], 2, dtype=np.int64))
    if edge_index:
        ei = np.concatenate(edge_index, axis=1)
        et = np.concatenate(edge_type)
        # uniqueness is per (type, i, j): the same ordered relation pair
        # may legitimately carry several interaction types.
        rows = np.stack([et, ei[0], ei[1]], axis=1)
        rows = np.unique(rows, axis=0)
        et = rows[:, 0]
        ei = rows[:, 1:3].T.copy()
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
        et = np.zeros(0, dtype=np.int64)
    # self-loop per relation (type 0), as in the released code
    loop = np.tile(np.arange(n_relations, dtype=np.int64), (2, 1))
    ei = np.concatenate([ei, loop], axis=1)
    et = np.concatenate([et, np.zeros(n_relations, dtype=np.int64)])
    return {"edge_index": ei, "edge_type": et, "num_nodes": int(n_relations)}


def _max_ent(edges: np.ndarray) -> int:
    return int(max(edges[:, 0].max(), edges[:, 2].max()))


def _cart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([np.repeat(a, b.size), np.tile(b, a.size)], axis=0)


# --------------------------------------------------------------------- #
# Accepted-graph container                                               #
# --------------------------------------------------------------------- #

@dataclass
class KGSplit:
    """One accepted synthetic KG with premise-faithful splits (Eqs. 1-3).

    All triple sets are int64 [N, 3] arrays of canonical (h, r, t).
    Raw categories (body / background / noise / derived / dropped) are
    stored alongside the premise-faithful sets so the *standard* arm
    (ULTRA-Syn) can re-pool them into the random 80/10/10 split without
    re-running the generator.
    """

    kg_id: int
    seed: int
    n_entities: int
    n_relations: int
    stats: Dict[str, float]
    templates: List[str]

    body: np.ndarray            # B: premises, always observable, never targets
    background: np.ndarray      # edges of Rbg
    noise: np.ndarray           # N: observable, train queries only
    derived: np.ndarray         # kept rule conclusions (post cap / eta-drop)
    dropped_cap: np.ndarray     # 50K-cap excess -> filter set F
    dropped_eta: np.ndarray     # eta-dropped conclusions -> filter set F

    d_train: np.ndarray
    d_val: np.ndarray
    d_test: np.ndarray
    b_train: np.ndarray
    b_val: np.ndarray
    b_test: np.ndarray

    e_obs_train: np.ndarray
    e_obs_val: np.ndarray
    e_obs_test: np.ndarray
    q_train: np.ndarray
    q_val: np.ndarray
    q_test: np.ndarray

    gr_train: Dict = field(default_factory=dict, repr=False)
    gr_val: Dict = field(default_factory=dict, repr=False)
    gr_test: Dict = field(default_factory=dict, repr=False)

    # -------------------------------------------------------------- #
    @property
    def dropped(self) -> np.ndarray:
        """Filter set F: cap excess + eta-dropped conclusions."""
        return _concat([self.dropped_cap, self.dropped_eta])

    @property
    def all_true(self) -> np.ndarray:
        """Every true triple the generator knows (filtered-ranking set)."""
        return _dedupe(_concat([self.body, self.background, self.noise,
                                self.derived, self.dropped,
                                self.d_train, self.d_val, self.d_test,
                                self.b_train, self.b_val, self.b_test]))

    @property
    def n_edges_train(self) -> int:
        return int(len(self.e_obs_train))

    @property
    def n_train_queries(self) -> int:
        return int(len(self.q_train))

    # -------------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "KGSplit":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __post_init__(self):
        self._assert_no_leakage()

    def _assert_no_leakage(self):
        """Algorithm 1 step 25: val/test queries must not be observable."""
        for q, e_obs, name in (
            (self.q_val, self.e_obs_val, "val"),
            (self.q_test, self.e_obs_test, "test"),
        ):
            if len(q) == 0:
                continue
            qk = _triple_keys(q, self.n_relations, self.n_entities)
            ek = _triple_keys(e_obs, self.n_relations, self.n_entities)
            leak = np.intersect1d(qk, ek)
            assert leak.size == 0, f"supervision leakage in {name} split: " \
                                   f"{leak.size} queries are observable"


# --------------------------------------------------------------------- #
# The generator                                                          #
# --------------------------------------------------------------------- #

class SyntheticKGGenerator:
    """Factorised generator of Algorithm 1.

    Parameters
    ----------
    seed : int
        Master seed; KG ``i`` uses ``spawnentropy(seed, i)``, so a corpus
        is reproducible regardless of acceptance order interleaving.
    use_rules : bool
        False = corpus B/D (§4.2): derived edges are rewired with
        ``10 |E_r|`` degree-preserving double-edge swaps per head.
    fixed_stats : dict or None
        Statistics axis clamped to constants (ablation C/D, Table 3):
        keys n_ent, n_rel, kappa, gamma, mu, eta, nu.
    template_pool : list of str or None
        Rule catalog restriction (Table 9).  The per-KG template-COUNT
        distribution of the full catalog is preserved (Bernoulli draw,
        capped at |pool|); members are drawn uniformly from the pool.
    n_ent_range, mu_range, gamma_range : (lo, hi) or None
        Prior overrides for the many-small control (Appendix H).
    min_derived_train, min_q_val : int or None
        Acceptance overrides; the many-small control relaxes them to
        10 / 5 (Appendix H).
    target_edges_per_kg : int or list of int or None
        Edge budget for the edge-matched / 3-cycled controls: background
        edges are uniformly subsampled until |E_obs_train| <= target.
    uniform_degree_prob : float
        Point mass of the uniform-degree option on gamma (0.15, §3.3).
    """

    def __init__(self, seed: int = 0,
                 use_rules: bool = True,
                 fixed_stats: Optional[Dict[str, float]] = None,
                 template_pool: Optional[Sequence[str]] = None,
                 n_ent_range: Optional[Tuple[float, float]] = None,
                 mu_range: Optional[Tuple[float, float]] = None,
                 gamma_range: Optional[Tuple[float, float]] = None,
                 min_derived_train: Optional[int] = None,
                 min_q_val: Optional[int] = None,
                 uniform_degree_prob: float = 0.15,
                 target_edges_per_kg=None,
                 total_edge_budget: Optional[int] = None,
                 n_kg: Optional[int] = None,
                 no_rules_swap_factor: int = NO_RULES_SWAP_FACTOR,
                 rule_form: str = "hard",
                 epsilon_range: Tuple[float, float] = (0.0, 0.3)):
        if rule_form not in ("hard", "soft"):
            raise ValueError("rule_form must be 'hard' or 'soft'")
        self.rule_form = rule_form
        self.epsilon_range = tuple(epsilon_range)
        self.seed = int(seed)
        self.use_rules = bool(use_rules)
        self.fixed_stats = dict(fixed_stats) if fixed_stats else {}
        if template_pool is not None:
            pool = [t for t in template_pool]
            bad = set(pool) - set(TEMPLATES)
            if bad:
                raise ValueError(f"unknown template(s): {sorted(bad)}")
            self.template_pool = pool
        else:
            self.template_pool = list(TEMPLATE_IDS)
        self.n_ent_range = tuple(n_ent_range) if n_ent_range else (100.0, 30_000.0)
        self.mu_range = tuple(mu_range) if mu_range else (3.0, 50.0)
        self.gamma_range = tuple(gamma_range) if gamma_range else (1.5, 3.0)
        self.min_derived_train = (MIN_DERIVED_TRAIN if min_derived_train is None
                                  else int(min_derived_train))
        self.min_q_val = MIN_Q_VAL if min_q_val is None else int(min_q_val)
        self.uniform_degree_prob = float(uniform_degree_prob)
        self.no_rules_swap_factor = int(no_rules_swap_factor)
        if total_edge_budget is not None and n_kg:
            per = int(total_edge_budget) // int(n_kg)
            self.target_edges_per_kg = [per] * int(n_kg)
        elif target_edges_per_kg is None:
            self.target_edges_per_kg = None
        elif isinstance(target_edges_per_kg, (list, tuple)):
            self.target_edges_per_kg = [int(x) for x in target_edges_per_kg]
        else:
            self.target_edges_per_kg = [int(target_edges_per_kg)]
        self.n_rejections = 0

    # -------------------------------------------------------------- #
    # public                                                          #
    # -------------------------------------------------------------- #
    def generate(self, kg_id: int, max_attempts: int = 100) -> KGSplit:
        """Sample (T, S) from scratch until Acc(G) holds (step 8).

        Overall rejection rate on the released corpus is 7.2%
        (Appendix N), so max_attempts is never approached in practice.
        """
        for attempt in range(max_attempts):
            rng = self._kg_rng(kg_id, attempt)
            split = self._try_generate(kg_id, rng)
            if split is not None:
                return split
            self.n_rejections += 1
        raise RuntimeError(
            f"KG {kg_id} rejected {max_attempts} times in a row; "
            "check the priors / acceptance thresholds.")

    def _kg_rng(self, kg_id: int, attempt: int) -> np.random.Generator:
        master = np.random.SeedSequence([self.seed, kg_id, attempt])
        return np.random.default_rng(master)

    # -------------------------------------------------------------- #
    # Step 1: statistics S (independent of T)                          #
    # -------------------------------------------------------------- #
    def _draw_stats(self, rng: np.random.Generator) -> Dict[str, float]:
        s = {}
        s["n_ent"] = int(round(_log_uniform(rng, *self.n_ent_range)))
        s["n_rel"] = int(round(_log_uniform(rng, 6, 192)))
        s["kappa"] = _log_uniform(rng, 0.1, 10)
        if rng.random() < self.uniform_degree_prob:
            s["gamma"] = None                      # uniform-degree option
        else:
            s["gamma"] = float(rng.uniform(*self.gamma_range))
        s["mu"] = _log_uniform(rng, *self.mu_range)
        s["eta"] = float(rng.uniform(0.0, 0.5))
        s["nu"] = float(rng.uniform(0.0, 0.10))
        for k, v in self.fixed_stats.items():      # ablation C/D clamp
            if k not in s:
                raise KeyError(f"unknown statistic {k!r} in fixed_stats")
            s[k] = float(v) if k not in ("n_ent", "n_rel") else int(round(v))
        s["n_ent"] = max(s["n_ent"], 4)
        s["n_rel"] = max(s["n_rel"], 3)
        return s

    # -------------------------------------------------------------- #
    # Step 2: templates T (independent of S)                           #
    # -------------------------------------------------------------- #
    def _draw_templates(self, rng: np.random.Generator) -> List[str]:
        count = sum(1 for tid in TEMPLATE_IDS
                    if rng.random() < LEVEL_PROB[TEMPLATES[tid]["level"]])
        if count == 0:                             # composition fallback
            fb = FALLBACK_TEMPLATE
            return [fb] if fb in self.template_pool else [self.template_pool[0]]
        # count-preserving ablation (App. J.2): keep the full-catalog
        # count distribution, cap at |pool|, members uniform in pool.
        k = min(count, len(self.template_pool))
        idx = rng.choice(len(self.template_pool), size=k, replace=False)
        return [self.template_pool[i] for i in sorted(idx.tolist())]

    # -------------------------------------------------------------- #
    # Steps 3-8                                                        #
    # -------------------------------------------------------------- #
    def _try_generate(self, kg_id: int, rng: np.random.Generator
                      ) -> Optional[KGSplit]:
        stats = self._draw_stats(rng)
        templates = self._draw_templates(rng)
        n_ent = stats["n_ent"]

        # ---- Step 3: relation slots (always feasible) -------------- #
        n_head = len(templates)
        r0 = list(range(stats["n_rel"]))
        heads = {t: stats["n_rel"] + i for i, t in enumerate(templates)}
        sym_head = heads.get("T1")
        bodies: Dict[str, List[int]] = {}
        for t in templates:
            if t == "T1":
                bodies[t] = [heads["T1"]]          # head is its own body
            else:
                k = TEMPLATES[t]["bodies"]
                bodies[t] = [int(x) for x in rng.choice(r0, size=k,
                                                        replace=False)]
        body_rels = sorted({r for t in templates for r in bodies[t]})
        rbg = [r for r in r0 if r not in set(body_rels)]
        n_relations = stats["n_rel"] + n_head
        assert all(r not in set(heads.values()) or r == sym_head
                   for r in body_rels), "heads are never bodies"

        # ---- Step 4: non-derived edges ----------------------------- #
        M = int(round(n_ent * stats["mu"]))
        freq_rels = rbg + [r for r in body_rels if r != sym_head] + \
                    ([sym_head] if sym_head is not None else [])
        pi = rng.dirichlet(np.full(len(freq_rels), stats["kappa"]))
        m_r = rng.multinomial(M, pi)
        if stats["gamma"] is None:
            a_out = np.ones(n_ent)
            a_in = np.ones(n_ent)
        else:
            a_out, a_in = _powerlaw_activities(rng, n_ent, stats["gamma"], M)
        edges = []
        for rel, m in zip(freq_rels, m_r):
            if m > 0:
                edges.append(_stub_matching_relation(rng, rel, int(m),
                                                     n_ent, a_out, a_in))
        nonderived = _concat(edges)

        body = nonderived[np.isin(nonderived[:, 1], body_rels)] \
            if len(nonderived) else _empty3()
        background = nonderived[np.isin(nonderived[:, 1], rbg)] \
            if len(nonderived) else _empty3()

        # ---- Step 5: derivation, full closure, 50K cap ------------- #
        derived, dropped_cap = self._derive(templates, bodies, heads, body,
                                            n_ent, rng)

        # soft-rule sensitivity (Table 10f): each conclusion survives
        # independently with probability eps ~ U(eps_lo, eps_hi)
        if self.rule_form == "soft":
            eps = float(rng.uniform(*self.epsilon_range))
            if len(derived):
                derived = derived[rng.random(len(derived)) < eps]

        # ---- corpus B/D: destroy rule correlation (App. N) --------- #
        if not self.use_rules:
            derived = self._rewire_derived(derived, rng)

        # ---- Step 6: incompleteness and noise ---------------------- #
        dropped_eta = _empty3()
        n_drop = int(math.floor(stats["eta"] * len(derived)))
        if n_drop:
            keep_mask = np.ones(len(derived), dtype=bool)
            keep_mask[rng.choice(len(derived), size=n_drop,
                                 replace=False)] = False
            dropped_eta = derived[~keep_mask]
            derived = derived[keep_mask]

        noise = self._draw_noise(rng, background, r0, n_ent, stats["nu"],
                                 existing=_concat([body, background, derived]),
                                 n_relations=n_relations)

        # ---- edge budget for the matched controls (App. H) --------- #
        target = None
        if self.target_edges_per_kg is not None:
            target = self.target_edges_per_kg[kg_id % len(self.target_edges_per_kg)]

        # ---- Step 7: premise-faithful split (Eqs. 1-3) ------------- #
        d_train, d_val, d_test = _split(rng, derived,
                                        DERIVED_TRAIN_FRAC, DERIVED_VAL_FRAC)
        b_train, b_val, b_test = _split(rng, background,
                                        BG_TRAIN_FRAC, BG_VAL_FRAC)
        if target is not None:
            over = (len(body) + len(d_train) + len(b_train) + len(noise)) - target
            if over > 0:
                keep = max(0, len(background) - over)
                if keep < len(background):
                    sel = rng.choice(len(background), size=keep, replace=False)
                    bg_all = background[sel]
                    b_train, b_val, b_test = _split(rng, bg_all,
                                                    BG_TRAIN_FRAC, BG_VAL_FRAC)
                    background = bg_all

        e_obs_train = _dedupe(_concat([body, d_train, b_train, noise]))
        e_obs_val = e_obs_train
        e_obs_test = _dedupe(_concat([body, d_train, b_train, noise,
                                      d_val, b_val]))
        q_train = _dedupe(_concat([d_train, b_train, noise]))
        q_val = _dedupe(_concat([d_val, b_val]))
        q_test = _dedupe(_concat([d_test, b_test]))

        # ---- Step 8: acceptance Acc(G) ----------------------------- #
        if not self._accept(e_obs_train, d_train, q_val, n_ent):
            return None

        split = KGSplit(
            kg_id=kg_id, seed=self.seed,
            n_entities=n_ent, n_relations=n_relations,
            stats=stats, templates=templates,
            body=body, background=background, noise=noise,
            derived=derived, dropped_cap=dropped_cap,
            dropped_eta=dropped_eta,
            d_train=d_train, d_val=d_val, d_test=d_test,
            b_train=b_train, b_val=b_val, b_test=b_test,
            e_obs_train=e_obs_train, e_obs_val=e_obs_val,
            e_obs_test=e_obs_test,
            q_train=q_train, q_val=q_val, q_test=q_test,
        )
        # ---- Step 9: cache the three relation graphs --------------- #
        split.gr_train = lift(e_obs_train, n_relations)
        split.gr_val = split.gr_train
        split.gr_test = lift(e_obs_test, n_relations)
        return split

    # -------------------------------------------------------------- #
    def _accept(self, e_obs_train, d_train, q_val, n_ent) -> bool:
        """Acc(G): LCC >= 0.5 n_ent; |E_obs_train| <= 500K;
        |d_train| >= 100; |Q_val| >= 50 (Algorithm 1 step 8)."""
        if len(e_obs_train) > MAX_OBS_EDGES:
            return False
        if len(d_train) < self.min_derived_train:
            return False
        if len(q_val) < self.min_q_val:
            return False
        if _lcc_fraction(e_obs_train, n_ent) < MIN_LCC_FRACTION:
            return False
        return True

    # -------------------------------------------------------------- #
    def _derive(self, templates, bodies, heads, body, n_ent, rng):
        """Hash-join derivation: all groundings per template (one pass
        suffices -- heads are never bodies), uniform subsample to the
        50K per-template cap; capped-away conclusions join the filter
        set F.  Derived triples that are already body edges (possible
        only for symmetry with reciprocal stub edges) are not valid
        queries and stay in the observable body."""
        by_rel: Dict[int, np.ndarray] = {}
        if len(body):
            for rel in np.unique(body[:, 1]):
                rel = int(rel)
                by_rel[rel] = body[body[:, 1] == rel]

        derived_blocks, dropped_blocks = [], []
        for t in templates:
            hd = heads[t]
            concl = self._ground(t, bodies[t], hd, by_rel)
            if len(concl):
                concl = _dedupe(concl)
                if t == "T1":
                    # drop conclusions already observable as body edges
                    bk = _triple_keys(body, hd + 1, n_ent)
                    ck = _triple_keys(concl, hd + 1, n_ent)
                    keep = ~np.isin(ck, bk)
                    concl = concl[keep]
            if len(concl) > DERIVED_CAP_PER_TEMPLATE:
                sel = rng.choice(len(concl), size=DERIVED_CAP_PER_TEMPLATE,
                                 replace=False)
                sel.sort()
                mask = np.zeros(len(concl), dtype=bool)
                mask[sel] = True
                dropped_blocks.append(concl[~mask])
                concl = concl[mask]
            derived_blocks.append(concl)
        derived = _dedupe(_concat(derived_blocks))
        dropped_cap = _concat(dropped_blocks)
        return derived, dropped_cap

    def _ground(self, tid, body_rels, head, by_rel) -> np.ndarray:
        """All groundings of one template via dictionary-free hash joins."""
        e = [by_rel.get(r, _empty3()) for r in body_rels]
        if tid == "T1":
            ed = e[0]
            return _empty3() if len(ed) == 0 else \
                np.stack([ed[:, 2], np.full(len(ed), head, np.int64),
                          ed[:, 0]], axis=1)
        if tid == "T2":
            ed = e[0]
            return _empty3() if len(ed) == 0 else \
                np.stack([ed[:, 2], np.full(len(ed), head, np.int64),
                          ed[:, 0]], axis=1)
        if tid == "T3":
            ed = e[0]
            return _empty3() if len(ed) == 0 else \
                np.stack([ed[:, 0], np.full(len(ed), head, np.int64),
                          ed[:, 2]], axis=1)
        if tid == "T4":                             # r1(x,z) & r2(z,y)
            if len(e[0]) == 0 or len(e[1]) == 0:
                return _empty3()
            ia, ib = _hash_join(e[0], e[1], 2, 0)
            return _empty3() if len(ia) == 0 else np.stack(
                [e[0][ia, 0], np.full(len(ia), head, np.int64),
                 e[1][ib, 2]], axis=1)
        if tid == "T5":                             # same (x, y) pair
            if len(e[0]) == 0 or len(e[1]) == 0:
                return _empty3()
            k0 = e[0][:, 0].astype(np.int64) * 2**32 + e[0][:, 2]
            k1 = e[1][:, 0].astype(np.int64) * 2**32 + e[1][:, 2]
            common = np.intersect1d(k0, k1)
            return _empty3() if len(common) == 0 else np.stack(
                [common >> 32, np.full(len(common), head, np.int64),
                 common & (2**32 - 1)], axis=1)
        if tid == "T6":                             # 3-hop chain
            if any(len(x) == 0 for x in e):
                return _empty3()
            ia, ib = _hash_join(e[0], e[1], 2, 0)   # (x,w) join (w,z)
            if len(ia) == 0:
                return _empty3()
            mid = np.stack([e[0][ia, 0], np.zeros(len(ia), np.int64),
                            e[1][ib, 2]], axis=1)   # (x, ., z)
            im, ic = _hash_join(mid, e[2], 2, 0)    # (x,z) join (z,y)
            return _empty3() if len(im) == 0 else np.stack(
                [mid[im, 0], np.full(len(im), head, np.int64),
                 e[2][ic, 2]], axis=1)
        if tid == "T7":                             # branch on shared tail
            if len(e[0]) == 0 or len(e[1]) == 0:
                return _empty3()
            ia, ib = _hash_join(e[0], e[1], 2, 2)   # z == z
            return _empty3() if len(ia) == 0 else np.stack(
                [e[0][ia, 0], np.full(len(ia), head, np.int64),
                 e[1][ib, 0]], axis=1)
        raise ValueError(f"unknown template {tid}")

    # -------------------------------------------------------------- #
    def _rewire_derived(self, derived: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        """Corpus B/D: per head relation, 10 |E_r| degree-preserving
        directed double-edge swaps destroy the rule correlation while
        preserving the degree sequences (Appendix N).  ~0.4% of edges
        remain rule-supported by chance (defaults
        NO_RULES_RESIDUAL_RULE_FRACTION)."""
        if len(derived) == 0:
            return derived
        new = derived.copy()
        for rel in np.unique(derived[:, 1]):
            rel = int(rel)
            rows = np.nonzero(derived[:, 1] == rel)[0]
            if len(rows) < 2:
                continue
            h = new[rows, 0].astype(np.int64)
            t = new[rows, 2].astype(np.int64).copy()
            seen = set(zip(h.tolist(), t.tolist()))
            n_swaps = self.no_rules_swap_factor * len(h)
            n_edge = len(h)
            for _ in range(n_swaps):
                i, j = int(rng.integers(n_edge)), int(rng.integers(n_edge))
                if i == j or h[i] == h[j] or t[i] == t[j]:
                    continue
                if (h[i], t[j]) in seen or (h[j], t[i]) in seen:
                    continue
                if h[i] == t[j] or h[j] == t[i]:      # no self-loops
                    continue
                seen.discard((h[i], t[i]))
                seen.discard((h[j], t[j]))
                t[i], t[j] = t[j], t[i]                # swap tails
                seen.add((h[i], t[i]))
                seen.add((h[j], t[j]))
            new[rows, 2] = t
        return _dedupe(new)

    # -------------------------------------------------------------- #
    def _draw_noise(self, rng, background, r0, n_ent, nu,
                    existing, n_relations) -> np.ndarray:
        """N = floor(nu |background|) uniform triples from V x R0 x V,
        new only (Algorithm 1 step 19).  Noise lives on non-head
        relations only, so it can never collide with a head-relation
        derived triple."""
        n_target = int(math.floor(nu * len(background)))
        if n_target <= 0 or not r0:
            return _empty3()
        r0_arr = np.asarray(r0, dtype=np.int64)
        n_rel = int(n_relations)
        if len(existing):
            ek = _triple_keys(existing, n_rel, n_ent)
            ek = np.sort(np.unique(ek))
        else:
            ek = _empty_idx()
        collected, got = [], 0
        while got < n_target:
            want = min(2 * (n_target - got) + 16, 4 * n_target)
            h = rng.integers(0, n_ent, size=want)
            r = r0_arr[rng.integers(0, len(r0_arr), size=want)]
            t = rng.integers(0, n_ent, size=want)
            cand = np.stack([h, r, t], axis=1)
            cand = _dedupe(cand)
            ck = _triple_keys(cand, n_rel, n_ent)
            fresh = cand[~np.isin(ck, ek)]
            if len(fresh) == 0:
                continue
            take = min(len(fresh), n_target - got)
            collected.append(fresh[:take])
            ek = np.sort(np.concatenate([ek, _triple_keys(fresh[:take],
                                                          n_rel, n_ent)]))
            got += take
            if got >= n_target:
                break
            if want == 0:
                break
        return _concat(collected)


def _split(rng: np.random.Generator, triples: np.ndarray, frac_train: float,
           frac_val: float):
    """Random (unstratified) split of one raw category into three."""
    if len(triples) == 0:
        return _empty3(), _empty3(), _empty3()
    perm = rng.permutation(len(triples))
    n_tr = int(round(frac_train * len(triples)))
    n_va = int(round(frac_val * len(triples)))
    n_tr = max(0, min(n_tr, len(triples)))
    n_va = max(0, min(n_va, len(triples) - n_tr))
    return (triples[perm[:n_tr]],
            triples[perm[n_tr:n_tr + n_va]],
            triples[perm[n_tr + n_va:]])


# --------------------------------------------------------------------- #
# Real subgraph control (Appendix H, Table 4 row 4)                      #
# --------------------------------------------------------------------- #

def sample_real_subgraph(kg_id: int, triples: np.ndarray, n_relations: int,
                         rng: np.random.Generator, walker_steps: int = 80,
                         min_entities: int = 40,
                         size_range: Tuple[float, float] = (100.0, 30_000.0),
                         retention_range: Tuple[float, float] = (0.3, 1.0),
                         seed: int = 0) -> Optional[KGSplit]:
    """Random-walk entity sample of a real KG -> standard-supervision
    KGSplit (control 4, "real subgraphs": subgraphs of FB15k-237 /
    WN18RR / CoDEx-M, target size ~ LogU(100, 30K), edge retention
    ~ U(0.3, 1), standard task construction, no rules).

    The walk runs over the undirected view of the training graph from a
    uniform seed entity; the subgraph is induced on visited entities and
    each incident edge is kept independently with probability u.
    Returns None when fewer than ``min_entities`` nodes are reachable.
    """
    if len(triples) == 0:
        return None
    n_ent_total = int(max(triples[:, 0].max(), triples[:, 2].max())) + 1
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for h, r, t in triples.tolist():
        adj.setdefault(int(h), []).append((int(t), int(r)))
        adj.setdefault(int(t), []).append((int(h), int(r)))
    n_target = int(round(_log_uniform(rng, *size_range)))
    u_keep = float(rng.uniform(*retention_range))

    nodes = set()
    starts = [int(rng.integers(n_ent_total))
              for _ in range(max(1, n_target // max(walker_steps, 1) + 1))]
    for s in starts:
        cur = s
        nodes.add(cur)
        for _ in range(walker_steps):
            nbrs = adj.get(cur)
            if not nbrs:
                break
            nxt, _ = nbrs[int(rng.integers(len(nbrs)))]
            cur = nxt
            nodes.add(cur)
            if len(nodes) >= n_target:
                break
        if len(nodes) >= n_target:
            break
    if len(nodes) < min_entities:
        return None
    mask = np.isin(triples[:, 0], list(nodes)) & np.isin(triples[:, 2],
                                                         list(nodes))
    sub = triples[mask]
    sub = sub[rng.random(len(sub)) < u_keep]
    if len(sub) < 4:
        return None
    sub = _dedupe(sub)
    node_list = np.unique(np.concatenate([sub[:, 0], sub[:, 2]]))
    remap = -np.ones(n_ent_total, dtype=np.int64)
    remap[node_list] = np.arange(len(node_list))
    sub = np.stack([remap[sub[:, 0]], sub[:, 1], remap[sub[:, 2]]], axis=1)
    n_ent = int(len(node_list))

    # standard supervision arm: no rules, so everything is background;
    # the pooled 80/10/10 split happens in StandardSyntheticCorpus.
    b_train, b_val, b_test = _split(rng, sub, BG_TRAIN_FRAC, BG_VAL_FRAC)
    e_obs_train = _dedupe(b_train)
    e_obs_test = _dedupe(_concat([e_obs_train, b_val]))
    split = KGSplit(
        kg_id=kg_id, seed=seed,
        n_entities=n_ent, n_relations=int(n_relations),
        stats={"source": "real_subgraph", "u_keep": u_keep},
        templates=[],
        body=_empty3(), background=sub, noise=_empty3(),
        derived=_empty3(), dropped_cap=_empty3(), dropped_eta=_empty3(),
        d_train=_empty3(), d_val=_empty3(), d_test=_empty3(),
        b_train=b_train, b_val=b_val, b_test=b_test,
        e_obs_train=e_obs_train, e_obs_val=e_obs_train,
        e_obs_test=e_obs_test,
        q_train=_dedupe(b_train), q_val=_dedupe(b_val),
        q_test=_dedupe(b_test),
    )
    split.gr_train = lift(e_obs_train, int(n_relations))
    split.gr_val = split.gr_train
    split.gr_test = lift(e_obs_test, int(n_relations))
    return split
