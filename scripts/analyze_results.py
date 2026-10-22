#!/usr/bin/env python3
"""
Structural-distance analysis: reproduce Figure 4 (right) and Figure 6.

Fits the Eq. 5 interaction
    MRR[g,m,s] = b0 + b1*m + b2*d_g + b3*(m*d_g) + gamma^T x_g + u_c(g) + eps
per distance definition (min-of-3 default, FB15k-237-only, mean-of-3, and
in-/out-degree variants of the min-of-3 distance; Appendix E), prints beta3
with wild-cluster-bootstrap CIs and the per-method slopes next to
benchmarks.EXPECTED_BETA3 / EXPECTED_SLOPES, and reports the cluster-level
Spearman correlation between mean per-graph ΔMRR and mean distance (§5.1:
rho = 0.55, permutation p = 0.09).

Paper reference values (§5.1, Figure 6):
    min_of_3   beta3 = +0.011 [+0.004, +0.019]   slopes -0.021 / -0.010
    fb_only    beta3 = +0.012 [+0.004, +0.020]
    mean_of_3  beta3 = +0.011 [+0.003, +0.019]
    in_degree  beta3 = +0.010        out_degree beta3 = +0.011

Usage
-----
python scripts/analyze_results.py \
    --results_csv  results/all_results.csv \
    --kg_dir       data/benchmarks \
    --pretraining_dir data/pretraining \
    --out_dir      results/analysis \
    --B 10000

CSV columns: kg, seed, mrr, method, with method in {'ultra_real', 'trace'}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from trace import benchmarks
from trace.analysis.distance import (
    _PRETRAIN_KEYS,
    cluster_spearman,
    fit_interaction,
    structural_distances,
)

METHODS = ("ultra_real", "trace")          # trace -> method indicator m = 1

# distance definition -> structural_distances(aggregation=...); the keys
# match benchmarks.EXPECTED_BETA3
DISTANCE_DEFS = {
    "min_of_3":   "min",
    "fb_only":    "fb_only",
    "mean_of_3":  "mean",
    "in_degree":  "in_degree",
    "out_degree": "out_degree",
}

# canonical repo names first (data/pretraining/{FB15k-237,WN18RR,
# CoDEx-M}, see README); legacy lowercase spellings accepted too
_PRETRAIN_SUBDIRS = (
    ("FB15k-237", "fb15k237"),
    ("WN18RR", "wn18rr"),
    ("CoDEx-M", "codex_m"),
)


def parse_args():
    p = argparse.ArgumentParser(description="Eq. 5 structural-distance analysis (Fig. 4/6)")
    p.add_argument("--results_csv",     type=str, required=True,
                   help="CSV with columns kg,seed,mrr,method (method in ultra_real|trace)")
    p.add_argument("--kg_dir",          type=str, required=True,
                   help="Benchmark KG directory (one subdir per graph)")
    p.add_argument("--pretraining_dir", type=str, required=True,
                   help="Directory with fb15k237/, wn18rr/, codex_m/ subdirs")
    p.add_argument("--out_dir",         type=str, default="results/analysis")
    p.add_argument("--B",               type=int, default=None,
                   help="Wild-cluster-bootstrap replicates for the beta3 CI")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text-file loaders (ULTRA tab-separated format).
# trace.evaluate exposes the same public helpers once it lands; keep a local
# fallback so this script runs standalone.
# ---------------------------------------------------------------------------

try:                                             # reuse the shared loaders
    from trace.evaluate import load_id_map, load_triples
except ImportError:                              # identical tab-separated format
    def load_id_map(path):
        """entity2id.txt / relation2id.txt: '<name>\t<id>' per line."""
        mapping = {}
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping

    def load_triples(path, e2id, r2id):
        """train/val/test.txt: '<h>\t<r>\t<t>' id columns, canonical triples."""
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


def load_kg_graph(kg_path, setting):
    """Inference-graph canonical triples and (n_entities, n_relations).

    Table 5 convention: training/inference-graph statistics -- transductive
    graphs contribute train+val+test, inductive graphs train+val (the
    inference graph; held-out prediction triples are excluded).
    """
    e2id = load_id_map(os.path.join(kg_path, "entity2id.txt"))
    r2id = load_id_map(os.path.join(kg_path, "relation2id.txt"))
    splits = ("train.txt", "val.txt", "test.txt") if setting == "T" \
        else ("train.txt", "val.txt")
    parts = []
    for split in splits:
        path = os.path.join(kg_path, split)
        if os.path.exists(path):
            parts.append(load_triples(path, e2id, r2id))
    if not parts:                                   # fall back to whatever exists
        for split in ("train.txt", "val.txt", "test.txt"):
            path = os.path.join(kg_path, split)
            if os.path.exists(path):
                parts.append(load_triples(path, e2id, r2id))
    triples = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.int64)
    return triples, len(e2id), max(len(r2id), 1)


def kg_dir_for(kg_dir, kg):
    for cand in (kg.replace(" ", "_").replace(":", "_"), kg):
        path = os.path.join(kg_dir, cand)
        if os.path.isdir(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

def load_results_csv(path):
    """{method: {kg: {seed: [mrr,...]}}} from the per-seed results CSV."""
    data = {m: {} for m in METHODS}
    with open(path) as f:
        for row in csv.DictReader(f):
            method = row["method"].strip()
            if method not in METHODS:
                continue
            kg = row["kg"]
            seed = int(row["seed"])
            data[method].setdefault(kg, {}).setdefault(seed, []).append(float(row["mrr"]))
    return data


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results_csv(args.results_csv)
    all_kgs = sorted(set(data["ultra_real"]) & set(data["trace"]))
    manifest_kgs = [kg for kg in all_kgs if kg in benchmarks.GRAPHS]
    dropped = sorted(set(all_kgs) - set(manifest_kgs))
    if dropped:
        print(f"  NOTE: {len(dropped)} kg(s) not in the 54-graph manifest skipped: {dropped}")
    if len(manifest_kgs) < 2:
        sys.exit("Need both 'ultra_real' and 'trace' rows for manifest KGs.")
    print(f"Loaded {len(manifest_kgs)} benchmark KGs, methods {METHODS}")

    # ---- KG triples ----------------------------------------------------- #
    pretraining_triples = {}
    for idx, (canonical, *aliases) in enumerate(_PRETRAIN_SUBDIRS):
        path = next((os.path.join(args.pretraining_dir, cand)
                     for cand in (canonical, *aliases)
                     if os.path.isdir(os.path.join(args.pretraining_dir,
                                                   cand))), None)
        if path is not None:
            t, n_ent, _ = load_kg_graph(path, "T")
            # distance.pretraining_distances keys on _PRETRAIN_KEYS
            pretraining_triples[_PRETRAIN_KEYS[idx]] = (t, n_ent)
        else:
            print(f"  WARNING: pretraining KG not found: "
                  f"{os.path.join(args.pretraining_dir, canonical)}")
    if len(pretraining_triples) < 3:
        sys.exit("All three pretraining KGs (FB15k-237, WN18RR, CoDEx-M) "
                 "are required.")

    benchmark_triples, n_rel_by_kg = {}, {}
    for kg in manifest_kgs:
        path = kg_dir_for(args.kg_dir, kg)
        if path is None:
            print(f"  WARNING: benchmark KG directory not found for {kg}")
            continue
        try:
            t, n_ent, n_rel = load_kg_graph(path, benchmarks.setting_of(kg))
            benchmark_triples[kg] = (t, n_ent)
            n_rel_by_kg[kg] = n_rel
        except Exception as exc:                    # corrupt split files etc.
            print(f"  WARNING: could not load {kg}: {exc}")
    kgs = [kg for kg in manifest_kgs if kg in benchmark_triples]
    if not kgs:
        sys.exit("No benchmark KG triples could be loaded.")

    # ---- Eq. 5 design (per-seed observations) ---------------------------- #
    # covariates x_g: log|V|, log|R|, evaluation-setting dummies (E, ER)
    rows_m, rows_y, rows_dz, rows_cov, rows_clu, rows_g = [], [], [], [], [], []
    deltas, seed_counts = {}, {}
    for gpos, kg in enumerate(kgs):
        spec = benchmarks.GRAPHS[kg]
        n_ent, n_rel = benchmark_triples[kg][1], max(n_rel_by_kg[kg], 2)
        setting = spec.setting
        cov = (np.log(n_ent), np.log(n_rel),
               float(setting == "E"), float(setting == "ER"))
        per_method = {}
        for mpos, method in enumerate(METHODS):
            seeds = data[method][kg]
            seed_ids = sorted(seeds)
            mrrs = [float(np.mean(seeds[s])) for s in seed_ids]
            per_method[method] = mrrs
            for mrr in mrrs:
                rows_m.append(mpos)
                rows_y.append(mrr)
                rows_cov.append(cov)
                rows_clu.append(benchmarks.cluster_of(kg))
                rows_g.append(gpos)
        deltas[kg] = float(np.mean(per_method["trace"]) - np.mean(per_method["ultra_real"]))
        seed_counts[kg] = len(seed_ids)

    y = np.asarray(rows_y)
    m = np.asarray(rows_m)
    cov = np.asarray(rows_cov)
    clu = np.asarray(rows_clu)
    gid = np.asarray(rows_g)

    fit_kw = {}
    if args.B is not None:
        fit_kw["B"] = args.B

    out = {"kgs": kgs, "deltas": deltas, "seed_counts": seed_counts,
           "distances": {}, "fits": {}, "spearman": {}}

    # ---- one Eq. 5 fit per distance definition (Figure 6 sweep) --------- #
    for name, agg in DISTANCE_DEFS.items():
        dist_z = structural_distances(
            {kg: benchmark_triples[kg] for kg in kgs},
            pretraining_triples, aggregation=agg)
        if not dist_z:
            continue
        # broadcast the graph-level distance to the per-seed rows
        dz = np.array([dist_z[kgs[g]] for g in gid])
        fit = fit_interaction(y, m, dz, cov, clu, graph_ids=gid, **fit_kw)

        exp_beta3, exp_ci = benchmarks.EXPECTED_BETA3[name]
        lo, hi = fit["beta3_ci"]
        slopes = fit["slopes"]
        print(f"\n{name} (aggregation={agg})")
        print(f"  beta3 = {fit['beta3']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"   expected {exp_beta3:+.3f}"
              + (f" [{exp_ci[0]:+.3f}, {exp_ci[1]:+.3f}]" if exp_ci else "")
              + f"   (p={fit['p_value']:.3f}, {fit['ci_method']})")
        for mpos, label in enumerate(METHODS):
            exp_slope = benchmarks.EXPECTED_SLOPES[label]
            print(f"  slope[{label:10s}] = {slopes[mpos]:+.4f} per SD"
                  f"   expected {exp_slope:+.3f}")

        delta_arr = np.array([deltas[kg] for kg in kgs])
        clu_arr = benchmarks.cluster_ids(kgs)
        rho = cluster_spearman(delta_arr, np.array([dist_z[kg] for kg in kgs]),
                               clu_arr)
        exp_rho, exp_p = benchmarks.EXPECTED_CLUSTER_SPEARMAN
        print(f"  cluster Spearman: rho={rho['rho']:+.2f} "
              f"perm p={rho['p_value']:.2f}   expected rho={exp_rho:+.2f}, p={exp_p:.2f}")

        out["distances"][name] = dist_z
        out["fits"][name] = {
            "beta0": fit["beta0"], "beta1": fit["beta1"], "beta2": fit["beta2"],
            "beta3": fit["beta3"], "beta3_ci": list(fit["beta3_ci"]),
            "slopes": {METHODS[0]: fit["slopes"][0], METHODS[1]: fit["slopes"][1]},
            "p_value": fit["p_value"], "ci_method": fit["ci_method"],
            "n_obs": fit["n_obs"], "n_clusters": fit["n_clusters"],
        }
        out["spearman"][name] = rho

    out_path = os.path.join(args.out_dir, "structural_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
