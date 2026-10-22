#!/usr/bin/env python3
"""
Benchmark and synthetic-corpus statistics for Figure 3 (§4.1).

Figure 3 places the realized statistics of the accepted synthetic corpus
(bars) against the 54 benchmark graphs (ticks): entities, canonical
relations, edges per entity, degree exponent (discrete MLE, k_min = 1),
normalized relation-frequency entropy, and relation-graph density. The
paper's Figure 3 literally instructs the reader to run this script; without
it the last three benchmark panels read "benchmark: n/a".

For each of the 54 graphs in trace.benchmarks.BENCHMARK_KGS it computes
(n_ent, n_rel, edges_per_entity, degree_exponent, rel_freq_entropy,
rel_graph_density) via trace.analysis.probe.graph_statistics, following the
Table 5 convention (training-graph statistics for transductive, test /
inference-graph statistics for inductive settings; held-out prediction
triples excluded).

With --synthetic_dir it also loads the generator KGSplit pickles (each
exposes e_obs_train, n_entities, n_relations -- the observable training
graph B ∪ d_train ∪ b_train ∪ N of Algorithm 1) and emits the matching
synthetic-corpus distributions for the overlay.

Usage
-----
python scripts/compute_benchmark_stats.py \
    --kg_dir data/benchmarks \
    --out results/benchmark_stats.json \
    --synthetic_dir data/corpus/train
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from trace import benchmarks
from trace.analysis.probe import graph_statistics

STAT_KEYS = ("n_ent", "n_rel", "edges_per_entity",
             "degree_exponent", "rel_freq_entropy", "rel_graph_density")

# ---------------------------------------------------------------------------
# Text-file loaders (same tab-separated format as the shared trace.evaluate
# helpers; local fallback keeps this script standalone until that lands).
# ---------------------------------------------------------------------------

try:
    from trace.evaluate import load_id_map, load_triples
except ImportError:
    def load_id_map(path):
        """'<name>\t<id>' per line (entity2id.txt / relation2id.txt)."""
        mapping = {}
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping

    def load_triples(path, e2id, r2id):
        """'<h>\t<r>\t<t>' id columns, canonical triples (train/val/test)."""
        rows = []
        with open(path) as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                if h in e2id and r in r2id and t in e2id:
                    rows.append((e2id[h], r2id[r], e2id[t]))
        return np.array(rows, dtype=np.int64) if rows else np.zeros((0, 3), dtype=np.int64)


def kg_dir_for(kg_dir, kg):
    for cand in (kg.replace(" ", "_").replace(":", "_"), kg):
        path = os.path.join(kg_dir, cand)
        if os.path.isdir(path):
            return path
    return None


def load_graph_stats(kg_path, setting):
    """Statistic vector for one benchmark KG directory (Table 5 splits)."""
    e2id = load_id_map(os.path.join(kg_path, "entity2id.txt"))
    r2id = load_id_map(os.path.join(kg_path, "relation2id.txt"))
    splits = ("train.txt", "val.txt", "test.txt") if setting == "T" \
        else ("train.txt", "val.txt")
    parts = []
    for split in splits:
        path = os.path.join(kg_path, split)
        if os.path.exists(path):
            parts.append(load_triples(path, e2id, r2id))
    if not parts:
        for split in ("train.txt", "val.txt", "test.txt"):
            path = os.path.join(kg_path, split)
            if os.path.exists(path):
                parts.append(load_triples(path, e2id, r2id))
    triples = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.int64)

    n_ent, n_rel = len(e2id), max(len(r2id), 1)
    stats = graph_statistics(triples, n_ent, n_rel)
    n_edges = len(np.unique(triples, axis=0)) if triples.shape[0] else 0
    stats["n_ent"] = n_ent
    stats["n_rel"] = n_rel
    stats["edges_per_entity"] = float(n_edges) / max(n_ent, 1)
    return {k: stats[k] for k in STAT_KEYS}


# ---------------------------------------------------------------------------
# Synthetic corpus (Figure 3 bars)
# ---------------------------------------------------------------------------

def load_synthetic_stats(synthetic_dir):
    """Per-KG statistics of the accepted synthetic corpus from the generator
    KGSplit pickles (fields e_obs_train, n_entities, n_relations).

    These are trusted research artifacts produced by scripts/generate_corpus.py
    -- pickle.load executes no analysis code beyond unpickling KGSplit."""
    files = sorted(f for f in os.listdir(synthetic_dir) if f.endswith(".pkl"))
    rows, missing = [], []
    for fname in files:
        try:
            with open(os.path.join(synthetic_dir, fname), "rb") as f:
                split = pickle.load(f)
            triples = np.asarray(split.e_obs_train, dtype=np.int64)
            stats = graph_statistics(triples, int(split.n_entities),
                                     int(split.n_relations))
            n_edges = len(np.unique(triples, axis=0)) if triples.shape[0] else 0
            stats["n_ent"] = int(split.n_entities)
            stats["n_rel"] = int(split.n_relations)
            stats["edges_per_entity"] = float(n_edges) / max(int(split.n_entities), 1)
            rows.append({k: stats[k] for k in STAT_KEYS})
        except Exception as exc:
            missing.append(f"{fname}: {exc}")
    if missing:
        print(f"  WARNING: {len(missing)}/{len(files)} synthetic pickles skipped, "
              f"first: {missing[0]}")
    return rows


def summarize(rows):
    """Distribution summary for the overlay: per-statistic values plus
    min / median / max (Figure 3 bars vs. ticks)."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "values": rows}
    for key in STAT_KEYS:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[key] = ({
            "min": float(vals.min()), "median": float(np.median(vals)),
            "max": float(vals.max()), "mean": float(vals.mean()),
        } if vals.size else None)
    return out


def main():
    p = argparse.ArgumentParser(description="Figure 3 benchmark + corpus statistics")
    p.add_argument("--kg_dir", type=str, required=True,
                   help="Benchmark KG directory (one subdir per graph)")
    p.add_argument("--out", type=str, default="results/benchmark_stats.json")
    p.add_argument("--synthetic_dir", type=str, default=None,
                   help="Optional directory of generator KGSplit .pkl files")
    args = p.parse_args()

    benchmark = {}
    n_missing = 0
    for kg in benchmarks.BENCHMARK_KGS:
        path = kg_dir_for(args.kg_dir, kg)
        if path is None:
            print(f"  WARNING: no directory for {kg}")
            benchmark[kg] = None
            n_missing += 1
            continue
        try:
            benchmark[kg] = load_graph_stats(path, benchmarks.setting_of(kg))
        except Exception as exc:
            print(f"  WARNING: {kg}: {exc}")
            benchmark[kg] = None
            n_missing += 1
    print(f"Benchmark graphs computed: {len(benchmark) - n_missing}/"
          f"{len(benchmarks.BENCHMARK_KGS)}")

    synthetic = None
    if args.synthetic_dir:
        if not os.path.isdir(args.synthetic_dir):
            sys.exit(f"--synthetic_dir not found: {args.synthetic_dir}")
        rows = load_synthetic_stats(args.synthetic_dir)
        synthetic = summarize(rows)
        print(f"Synthetic corpus KGs computed: {synthetic['n']}")

    result = {
        "benchmark": benchmark,
        "benchmark_summary": summarize([v for v in benchmark.values() if v]),
        "synthetic": synthetic,
        "stat_keys": list(STAT_KEYS),
        "notes": ("Table 5 conventions: training-graph statistics for T, "
                  "inference-graph (train+val) statistics for E/ER; canonical "
                  "relations; degree exponent = CSN discrete MLE with k_min=1; "
                  "relation-frequency entropy normalized by log|R|; "
                  "rel_graph_density = mean per-type edge density of Lift(E)."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
