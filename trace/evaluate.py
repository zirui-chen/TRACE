"""
Zero-shot evaluation on the 54 benchmark KGs (Table 5, §4.1).

Standard filtered ranking: for each test query (h, q, ?) score every
entity, drop known-true triples other than the target, take the rank.
The observable graph at test time is train + val of the benchmark
split; relation graphs are Lift(E_obs) over canonical edges (the model
adapter adds inverse edges internally, as in the released data
loader).

The benchmark directory layout matches the released ULTRA benchmarks:

    {kg_dir}/{kg_name}/
        train.txt / val.txt / test.txt   (tab-separated h r t)
        entity2id.txt / relation2id.txt

``evaluate_benchmark_suite`` additionally reports the three slices
(source-proximal 22 / source-disjoint 32 / external-hard 14) defined
in trace.benchmarks and writes per-seed rows consumable by
``scripts/verify_numbers.py``.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = [
    "load_id_map",
    "load_triples",
    "evaluate_kg",
    "evaluate_one_kg",
    "evaluate_benchmark_suite",
    "write_results_csv",
]


# --------------------------------------------------------------------- #
# Loaders (shared with trace.real_supervision)                           #
# --------------------------------------------------------------------- #

def load_id_map(path: str) -> Dict[str, int]:
    id_map: Dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                id_map[parts[0]] = int(parts[1])
    return id_map


def load_triples(path: str, e2id: Optional[Dict[str, int]] = None,
                 r2id: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Tab-separated triples; with id maps mapped to ids, without them
    parsed as raw integers (pretraining graphs stored indexed)."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if e2id is None:
                try:
                    rows.append((int(h), int(r), int(t)))
                except ValueError:
                    continue
            else:
                if h in e2id and r in r2id and t in e2id:
                    rows.append((e2id[h], r2id[r], e2id[t]))
    return (np.array(rows, dtype=np.int64) if rows
            else np.empty((0, 3), dtype=np.int64))


# --------------------------------------------------------------------- #
# Filtered ranking                                                       #
# --------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_kg(model,
                triples_obs: np.ndarray,
                triples_test: np.ndarray,
                filter_triples: Optional[np.ndarray],
                n_entities: int,
                n_relations: int,
                device: torch.device,
                batch_size: int = 256,
                ) -> Dict[str, float]:
    """Filtered-ranking metrics for one KG on pre-built arrays."""
    from .generator import lift

    gr = lift(triples_obs, n_relations)
    entity_graph = _NS(
        torch.from_numpy(np.ascontiguousarray(triples_obs[:, [0, 2]].T)
                         .astype(np.int64)).to(device),
        torch.from_numpy(triples_obs[:, 1].astype(np.int64)).to(device),
        int(n_entities),
    )
    rel_graph = _NS(
        torch.from_numpy(gr["edge_index"].astype(np.int64)).to(device),
        torch.from_numpy(gr["edge_type"].astype(np.int64)).to(device),
        int(gr["num_nodes"]),
    )

    filter_set: Dict[Tuple[int, int], np.ndarray] = {}
    if filter_triples is not None and len(filter_triples):
        order = np.lexsort((filter_triples[:, 2], filter_triples[:, 1],
                            filter_triples[:, 0]))
        ft = filter_triples[order]
        keys, starts = np.unique(ft[:, :2], axis=0, return_index=True)
        bounds = np.append(starts[1:], len(ft))
        for (h, r), s, e in zip(keys.tolist(), starts.tolist(),
                                bounds.tolist()):
            filter_set[(int(h), int(r))] = ft[s:e, 2]

    model.eval()
    rr: List[float] = []
    h1: List[float] = []
    h3: List[float] = []
    h10: List[float] = []

    for start in range(0, len(triples_test), batch_size):
        chunk = triples_test[start:start + batch_size]
        query = torch.from_numpy(chunk.astype(np.int64)).to(device)
        scores = model(entity_graph, rel_graph, query)      # [B, n_ent]
        scores = scores.float()
        rows = scores.cpu().numpy()

        for i, (h, r, t) in enumerate(chunk.tolist()):
            row = rows[i].copy()
            t_score = row[int(t)]                 # target keeps its score
            known = filter_set.get((int(h), int(r)))
            if known is not None:
                k = known[(known != int(t)) & (known < n_entities)]
                row[k] = -np.inf
            rank = int((row > t_score).sum()) + 1
            rr.append(1.0 / rank)
            h1.append(float(rank <= 1))
            h3.append(float(rank <= 3))
            h10.append(float(rank <= 10))

    if not rr:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}
    return {"mrr": float(np.mean(rr)), "hits@1": float(np.mean(h1)),
            "hits@3": float(np.mean(h3)), "hits@10": float(np.mean(h10))}


class _NS:
    __slots__ = ("edge_index", "edge_type", "num_nodes")

    def __init__(self, edge_index, edge_type, num_nodes):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_nodes = num_nodes


def evaluate_one_kg(model, kg_path: str, device: torch.device,
                    batch_size: int = 256) -> Dict[str, float]:
    """Evaluate on one benchmark KG directory (ULTRA layout)."""
    e2id = load_id_map(os.path.join(kg_path, "entity2id.txt"))
    r2id = load_id_map(os.path.join(kg_path, "relation2id.txt"))
    n_ent, n_rel = len(e2id), len(r2id)
    train = load_triples(os.path.join(kg_path, "train.txt"), e2id, r2id)
    val = load_triples(os.path.join(kg_path, "val.txt"), e2id, r2id)
    test = load_triples(os.path.join(kg_path, "test.txt"), e2id, r2id)
    obs = np.concatenate([train, val], axis=0) if len(val) else train
    all_true = np.concatenate([train, val, test], axis=0)
    return evaluate_kg(model, obs, test, all_true, n_ent, n_rel, device,
                       batch_size)


# --------------------------------------------------------------------- #
# The 54-graph suite with slice reporting                                #
# --------------------------------------------------------------------- #

def evaluate_benchmark_suite(model,
                             kg_dir: str,
                             device: torch.device,
                             batch_size: int = 256,
                             kg_names: Optional[List[str]] = None,
                             seed: int = 0,
                             method: str = "trace",
                             verbose: bool = True,
                             ) -> Dict[str, Dict]:
    """Evaluate on the Table 5 suite and report the three slices.

    Returns ``{"per_kg": {kg: metrics}, "slices": {...},
    "rows": [csv rows]}`` where rows match the schema parsed by
    scripts/verify_numbers.py (kg, seed, mrr, hits_at_1, hits_at_3,
    hits_at_10, method).
    """
    from . import benchmarks as B

    names = kg_names if kg_names is not None else list(B.BENCHMARK_KGS)
    per_kg: Dict[str, Dict[str, float]] = {}
    rows = []
    for name in names:
        path = os.path.join(kg_dir, name)
        if not os.path.isdir(path):
            if verbose:
                print(f"  {name}: missing, skipped")
            continue
        m = evaluate_one_kg(model, path, device, batch_size)
        per_kg[name] = m
        rows.append({"kg": name, "seed": seed, "mrr": m["mrr"],
                     "hits_at_1": m["hits@1"], "hits_at_3": m["hits@3"],
                     "hits_at_10": m["hits@10"], "method": method})
        if verbose:
            print(f"  {name:30s} MRR {m['mrr']:.3f}  H@1 {m['hits@1']:.3f}"
                  f"  H@10 {m['hits@10']:.3f}", flush=True)

    slices = {}
    for label, kgs in (("all", B.BENCHMARK_KGS),
                       ("source_proximal", B.PROXIMAL_KGS),
                       ("source_disjoint", B.DISJOINT_KGS),
                       ("external_hard", B.EXTERNAL_HARD_KGS)):
        vals = [per_kg[k]["mrr"] for k in kgs if k in per_kg]
        if not vals and label == "all" and per_kg:
            # ad-hoc suites outside the Table 5 manifest (smoke tests,
            # custom graphs): summarize everything that was evaluated
            vals = [m["mrr"] for m in per_kg.values()]
        slices[label] = float(np.mean(vals)) if vals else float("nan")
    if verbose:
        print("  slices: " + "  ".join(f"{k}={v:.3f}"
                                       for k, v in slices.items()))
    return {"per_kg": per_kg, "slices": slices, "rows": rows}


def write_results_csv(rows: List[Dict], path: str,
                      append: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if append else "w"
    header = ["kg", "seed", "mrr", "hits_at_1", "hits_at_3", "hits_at_10",
              "method"]
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not append:
            writer.writeheader()
        writer.writerows(rows)
