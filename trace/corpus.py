"""
Pretraining corpora and batch construction (§3.4, Appendix I).

Every arm draws exactly one KG uniformly per step and applies the
standard ULTRA forward pass; the arms differ ONLY in how queries and
observable edges are assembled:

* ``PremiseFaithfulCorpus``  -- TRACE / TRACE-style synthetic arms.
  Each 64-query batch mixes 32 derived queries (rule conclusions,
  d_train) with 32 non-derived queries (background b_train and noise N)
  so the gradient does not silently reweight rule content by corpus
  volume (§3.2; the 50:50 ratio is the synthetic-validation argmax,
  Table 10a).  Body edges B are always observable and never targets.

* ``StandardSyntheticCorpus`` -- ULTRA-Syn (corpus arm control).
  All categories of each synthetic KG are pooled and randomly split
  80/10/10; queries are sampled uniformly from the training part --
  the standard pretraining construction, on the same corpus (§4.1).

* ``MixedCorpus`` -- control 5 (Appendix H): at each step take the
  three real graphs with probability ``prob_real`` (their own
  supervision) or a corpus KG (premise-faithful supervision).

Model selection for the synthetic arms uses the mean validation MRR
over the 200 held-out KGs with G_r^val and Q_val (§3.4).  Batches are
canonical (h, r, t); inverse edges are added inside the model adapter,
as in the released ULTRA data loader.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional

import numpy as np

from .defaults import (
    BATCH_SIZE,
    DERIVED_FRAC_IN_BATCH,
)
from .generator import KGSplit

__all__ = [
    "CorpusBase",
    "PremiseFaithfulCorpus",
    "StandardSyntheticCorpus",
    "MixedCorpus",
    "batch_to_ultra",
    "write_manifest",
    "read_manifest",
]


# --------------------------------------------------------------------- #
# Corpus directory layout                                                #
# --------------------------------------------------------------------- #

def write_manifest(corpus_dir: str, seed: int, train_ids: List[int],
                   select_ids: List[int], config: Dict) -> None:
    os.makedirs(os.path.join(corpus_dir, "kgs"), exist_ok=True)
    manifest = dict(seed=int(seed), n_kg=len(train_ids) + len(select_ids),
                    train_ids=list(train_ids), select_ids=list(select_ids),
                    config=config)
    with open(os.path.join(corpus_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(corpus_dir: str) -> Dict:
    with open(os.path.join(corpus_dir, "manifest.json")) as f:
        return json.load(f)


class _Graph:
    """Tiny namespace for the entity graph handed to the model."""

    __slots__ = ("edge_index", "edge_type", "num_nodes")

    def __init__(self, edge_index, edge_type, num_nodes):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_nodes = num_nodes


# --------------------------------------------------------------------- #
# Base                                                                   #
# --------------------------------------------------------------------- #

class CorpusBase:
    """Lazy-pickle store over ``corpus_dir/kgs/kg_%05d.pkl`` with an LRU.

    The released corpus (2,000 KGs) fits the 17 GB cache reported in
    Appendix O when the LRU is bounded; ``max_cached=None`` disables
    eviction.
    """

    def __init__(self, corpus_dir: str, seed: int = 0,
                 max_cached: Optional[int] = 256):
        self.corpus_dir = corpus_dir
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self.max_cached = max_cached
        self._cache: "OrderedDict[int, KGSplit]" = OrderedDict()
        manifest = read_manifest(corpus_dir)
        self._train_ids: List[int] = list(manifest["train_ids"])
        self._select_ids: List[int] = list(manifest["select_ids"])

    # -------------------------------------------------------------- #
    def _load(self, kg_id: int) -> KGSplit:
        if kg_id in self._cache:
            self._cache.move_to_end(kg_id)
            return self._cache[kg_id]
        path = os.path.join(self.corpus_dir, "kgs", f"kg_{kg_id:05d}.pkl")
        split = KGSplit.load(path)
        if self.max_cached is not None:
            while len(self._cache) >= self.max_cached:
                self._cache.popitem(last=False)
        self._cache[kg_id] = split
        return split

    def __len__(self) -> int:
        return len(self._train_ids)

    @property
    def train_ids(self) -> List[int]:
        return list(self._train_ids)

    @property
    def select_ids(self) -> List[int]:
        return list(self._select_ids)

    # -------------------------------------------------------------- #
    def _pick_kg(self) -> KGSplit:
        """Uniformly one training KG per step (Algorithm: step 3.4).

        Very small corpora (the 3-cycled control, n <= 16) are visited
        round-robin instead of sampled, matching Appendix H control 1.
        """
        if not hasattr(self, "_cycle_pos"):
            self._cycle_pos = 0
        if not self._train_ids:
            raise ValueError(
                "corpus has no training KGs: every generated graph landed "
                "in the held-out selection split.  Generate more KGs than "
                "--select_kg (or pass a smaller --select_kg).")
        if len(self._train_ids) <= 16:
            kg_id = self._train_ids[self._cycle_pos % len(self._train_ids)]
            self._cycle_pos += 1
        else:
            kg_id = int(self._rng.integers(len(self._train_ids)))
            kg_id = self._train_ids[kg_id]
        return self._load(kg_id)

    # -------------------------------------------------------------- #
    @staticmethod
    def _batch_from(split: KGSplit, gr: Dict, e_obs: np.ndarray,
                    q: np.ndarray, rng: np.random.Generator) -> Dict:
        return {
            "edge_index": np.ascontiguousarray(e_obs[:, [0, 2]].T),
            "edge_type": e_obs[:, 1].astype(np.int64),
            "num_entities": int(split.n_entities),
            "num_relations": int(split.n_relations),
            "gr_edge_index": gr["edge_index"],
            "gr_edge_type": gr["edge_type"],
            "gr_num_nodes": int(gr["num_nodes"]),
            "query_h": q[:, 0].astype(np.int64),
            "query_r": q[:, 1].astype(np.int64),
            "query_t": q[:, 2].astype(np.int64),
            "filter_triples": split.all_true,
        }


# --------------------------------------------------------------------- #
# Premise-faithful (TRACE)                                               #
# --------------------------------------------------------------------- #

class PremiseFaithfulCorpus(CorpusBase):
    """TRACE synthetic arm: balanced derived / non-derived batches.

    ``derived_frac`` is the fraction of each batch drawn from d_train;
    the remainder comes from b_train union N (Eqs. 1-3).  0.5 is the
    configuration selected on synthetic validation MRR (Table 10a) and
    is used by every reported run.
    """

    def __init__(self, corpus_dir: str, seed: int = 0,
                 batch_size: int = BATCH_SIZE,
                 derived_frac: float = DERIVED_FRAC_IN_BATCH,
                 max_cached: Optional[int] = 256,
                 split_variant: str = "provenance",
                 filter_set: str = "full",
                 q_val_includes_noise: bool = False):
        super().__init__(corpus_dir, seed=seed, max_cached=max_cached)
        self.batch_size = int(batch_size)
        if not 0.0 <= derived_frac <= 1.0:
            raise ValueError("derived_frac must be in [0, 1]")
        self.derived_frac = float(derived_frac)
        if split_variant == "fully_naive":
            raise ValueError("fully_naive == StandardSyntheticCorpus; "
                             "use that class instead")
        if split_variant not in ("provenance", "body_observable_pooled",
                                 "category_body_targets"):
            raise ValueError(f"unknown split_variant {split_variant!r}")
        if filter_set not in ("full", "without_eta_dropped"):
            raise ValueError(f"unknown filter_set {filter_set!r}")
        self.split_variant = split_variant
        self.filter_set = filter_set
        self.q_val_includes_noise = bool(q_val_includes_noise)
        self._variant_cache: Dict[int, tuple] = {}

    # -------------------------------------------------------------- #
    def _filter_array(self, split: KGSplit) -> np.ndarray:
        if self.filter_set == "full":
            return split.all_true
        # Table 10e: filter WITHOUT the eta-dropped conclusions -- the
        # known-true set is then B, bg, N, kept derived, cap-excess and
        # the split members only.
        return _dedup_np(_concat_np([
            split.body, split.background, split.noise, split.derived,
            split.dropped_cap, split.d_train, split.d_val, split.d_test,
            split.b_train, split.b_val, split.b_test]))

    def _variant(self, split: KGSplit) -> tuple:
        """Precomputed (e_obs, gr, q_train, e_obs_val, q_val) for the
        Table 10d split-ablation variants; identity for provenance."""
        if self.split_variant == "provenance":
            return (split.e_obs_train, split.gr_train, split.q_train,
                    split.e_obs_val, split.q_val)
        if split.kg_id in self._variant_cache:
            return self._variant_cache[split.kg_id]
        from .generator import lift
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, split.kg_id, 0x10D]))
        if self.split_variant == "body_observable_pooled":
            # observable unchanged; queries uniform over the whole
            # train pool (no derived/background balancing)
            out = (split.e_obs_train, split.gr_train,
                   _concat_np([split.d_train, split.b_train, split.noise]),
                   split.e_obs_val, split.q_val)
        else:  # category_body_targets: body split and used as targets
            bt, bv, be = _split_np(rng, split.body, 0.6, 0.2)
            e_tr = _dedup_np(_concat_np([bt, split.d_train, split.b_train,
                                         split.noise]))
            q_tr = _dedup_np(_concat_np([split.d_train, split.b_train, bt,
                                         split.noise]))
            e_va = _dedup_np(_concat_np([e_tr, split.d_val, split.b_val, bv]))
            q_va = _dedup_np(_concat_np([split.d_val, split.b_val, bv]))
            out = (e_tr, lift(e_tr, split.n_relations), q_tr, e_va, q_va)
        self._variant_cache[split.kg_id] = out
        return out

    # -------------------------------------------------------------- #
    def sample_batch(self) -> Dict:
        split = self._pick_kg()
        rng = self._rng
        e_obs, gr, q_pool, _e_va, _q_va = self._variant(split)
        q = self._sample_queries(split, rng, q_pool)
        batch = self._batch_from(split, gr, e_obs, q, rng)
        batch["filter_triples"] = self._filter_array(split)
        return batch

    def _sample_queries(self, split: KGSplit, rng, q_pool: np.ndarray
                        ) -> np.ndarray:
        if self.split_variant == "body_observable_pooled":
            if len(q_pool) == 0:
                raise RuntimeError(f"KG {split.kg_id} has no training queries")
            return q_pool[rng.choice(len(q_pool), size=self.batch_size,
                                     replace=len(q_pool) < self.batch_size)]
        nonderived = _concat_np([split.b_train, split.noise])
        avail_d, avail_n = len(split.d_train), len(nonderived)
        if avail_d == 0 and avail_n == 0:
            raise RuntimeError(f"KG {split.kg_id} has no training queries")

        # 32/32 target, filled proportionally when a pool is short.
        n_d = min(int(round(self.derived_frac * self.batch_size)), avail_d)
        n_n = self.batch_size - n_d
        if avail_n < n_n:                           # top up from derived
            n_d = min(self.batch_size, n_d + (n_n - avail_n))
            n_n = min(avail_n, self.batch_size - n_d)

        picks = []
        if n_d:
            picks.append(split.d_train[rng.choice(avail_d, size=n_d,
                                                  replace=avail_d < n_d)])
        if n_n:
            picks.append(nonderived[rng.choice(avail_n, size=n_n,
                                               replace=avail_n < n_n)])
        return _concat_np(picks)

    def val_batches(self, max_kgs: Optional[int] = None) -> Iterator[Dict]:
        """Validation batches over the 200 held-out selection KGs."""
        ids = self._select_ids if max_kgs is None else self._select_ids[:max_kgs]
        for kg_id in ids:
            split = self._load(kg_id)
            e_obs, gr, _q_tr, e_va, q_va = self._variant(split)
            q = q_va
            if self.q_val_includes_noise and len(split.noise):
                # Table 10g: deliberately leak observable noise into the
                # selection queries to measure selection sensitivity;
                # this violates Eq. 2 by design (a protocol ablation).
                take = min(len(q), len(split.noise)) if len(q) else len(split.noise)
                q = _concat_np([q, split.noise[self._rng.choice(
                    len(split.noise), size=take, replace=False)]])
            if len(q) == 0:
                continue
            batch = self._batch_from(split, gr, e_va, q, self._rng)
            batch["filter_triples"] = self._filter_array(split)
            yield batch


def _dedup_np(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(a.astype(np.int64), axis=0)


def _split_np(rng, triples: np.ndarray, frac_train: float, frac_val: float):
    if len(triples) == 0:
        return (np.zeros((0, 3), np.int64), np.zeros((0, 3), np.int64),
                np.zeros((0, 3), np.int64))
    perm = rng.permutation(len(triples))
    n_tr = int(round(frac_train * len(triples)))
    n_va = int(round(frac_val * len(triples)))
    return (triples[perm[:n_tr]], triples[perm[n_tr:n_tr + n_va]],
            triples[perm[n_tr + n_va:]])


# --------------------------------------------------------------------- #
# Standard supervision on the synthetic corpus (ULTRA-Syn arm)           #
# --------------------------------------------------------------------- #

class StandardSyntheticCorpus(CorpusBase):
    """ULTRA-Syn: same corpus, standard task construction (§4.1).

    All categories of each synthetic KG (body, background, noise,
    kept derived, and the dropped/filter set) are POOLED and randomly
    split 80/10/10; queries are sampled uniformly from the training
    part.  This reproduces the pretraining construction of the
    literature on top of our generator, so the corpus and supervision
    axes stay separable.  Model selection still uses the SAME 200
    held-out KGs with the same pooled split (Appendix I), so only the
    task construction differs between the synthetic arms.
    """

    def __init__(self, corpus_dir: str, seed: int = 0,
                 batch_size: int = BATCH_SIZE,
                 train_frac: float = 0.8, val_frac: float = 0.1,
                 max_cached: Optional[int] = 256):
        super().__init__(corpus_dir, seed=seed, max_cached=max_cached)
        self.batch_size = int(batch_size)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self._pooled: Dict[int, tuple] = {}

    def _pooled_split(self, split: KGSplit) -> tuple:
        if split.kg_id in self._pooled:
            return self._pooled[split.kg_id]
        pool = split.all_true
        # deterministic per-KG split, independent of global draw order
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, split.kg_id, 0x5AFE]))
        perm = rng.permutation(len(pool))
        n_tr = int(round(self.train_frac * len(pool)))
        n_va = int(round(self.val_frac * len(pool)))
        e_tr = pool[perm[:n_tr]]
        e_va = pool[perm[n_tr:n_tr + n_va]]
        e_te = pool[perm[n_tr + n_va:]]
        from .generator import lift
        gr_tr = lift(e_tr, split.n_relations)
        gr_va = lift(_concat_np([e_tr, e_va]), split.n_relations)
        out = (e_tr, e_va, e_te, gr_tr, gr_va)
        self._pooled[split.kg_id] = out
        return out

    def sample_batch(self) -> Dict:
        split = self._pick_kg()
        e_tr, _e_va, _e_te, gr_tr, _gr_va = self._pooled_split(split)
        rng = self._rng
        if len(e_tr) == 0:
            raise RuntimeError(f"KG {split.kg_id} has no pooled edges")
        idx = rng.choice(len(e_tr), size=self.batch_size,
                         replace=len(e_tr) < self.batch_size)
        q = e_tr[idx]
        return self._batch_from(split, gr_tr, e_tr, q, rng)

    def val_batches(self, max_kgs: Optional[int] = None) -> Iterator[Dict]:
        ids = self._select_ids if max_kgs is None else self._select_ids[:max_kgs]
        for kg_id in ids:
            split = self._load(kg_id)
            _e_tr, e_va, _e_te, _gr_tr, gr_va = self._pooled_split(split)
            if len(e_va) == 0:
                continue
            yield self._batch_from(split, gr_va, e_va, e_va, self._rng)


# --------------------------------------------------------------------- #
# Mixture control (Appendix H, control 5)                                #
# --------------------------------------------------------------------- #

class MixedCorpus:
    """50/50 step mixture of the real graphs and the TRACE corpus, each
    with its own supervision; the compute-matched control differs only
    by the step budget."""

    def __init__(self, real_corpus, synthetic_corpus,
                 prob_real: float = 0.5, seed: int = 0):
        self.real = real_corpus
        self.synthetic = synthetic_corpus
        self.prob_real = float(prob_real)
        self._rng = np.random.default_rng(seed)
        self.batch_size = getattr(synthetic_corpus, "batch_size", BATCH_SIZE)

    def sample_batch(self) -> Dict:
        if self._rng.random() < self.prob_real:
            return self.real.sample_batch()
        return self.synthetic.sample_batch()

    def val_batches(self, max_kgs: Optional[int] = None) -> Iterator[Dict]:
        # Selection follows the synthetic arms (200 held-out KGs); the
        # real arm contributes no validation queries here.
        yield from self.synthetic.val_batches(max_kgs=max_kgs)


# --------------------------------------------------------------------- #
# Tensor conversion                                                      #
# --------------------------------------------------------------------- #

def _concat_np(arrays):
    arrays = [a for a in arrays if a is not None and len(a)]
    if not arrays:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(arrays, axis=0)


def _as_tensor(x, device):
    """Accept numpy arrays (synthetic loaders) or tensors (real-KG
    loaders in trace.real_supervision) for any batch field."""
    import torch
    if hasattr(x, "to") and hasattr(x, "dtype"):
        return x.to(device)
    return torch.as_tensor(x).to(device)


def batch_to_ultra(batch: Dict, device):
    """Convert a batch dict to the (entity_graph, rel_graph, query,
    target) tuple expected by the reasoner (torch tensors on device)."""
    import torch
    edge_index = _as_tensor(batch["edge_index"], device)
    edge_type = _as_tensor(batch["edge_type"], device)
    entity_graph = _Graph(edge_index, edge_type, batch["num_entities"])
    rel_graph = _Graph(
        _as_tensor(batch["gr_edge_index"], device),
        _as_tensor(batch["gr_edge_type"], device),
        batch["gr_num_nodes"],
    )
    query = torch.stack([
        _as_tensor(batch["query_h"], device),
        _as_tensor(batch["query_r"], device),
        _as_tensor(batch["query_t"], device),
    ], dim=1)
    target = query[:, 2].clone()
    return entity_graph, rel_graph, query, target
