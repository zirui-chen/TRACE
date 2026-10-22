#!/usr/bin/env python3
"""
Generate one arm's pretraining corpus (Algorithm 1; Appendix G/N).

Reads a config YAML (the same file consumed by scripts/train.py) and
writes per-KG pickles plus a manifest under ``corpus_dir``:

    corpus_dir/
        manifest.json
        kgs/kg_00000.pkl ...

Example
-------
    python scripts/generate_corpus.py --config configs/trace.yaml
    python scripts/generate_corpus.py --config configs/trace.yaml --limit 8

The released corpus is 2,000 KGs (1,800 training + 200 held out for
model selection), generated in 45 +/- 5 minutes on an 8-core AMD EPYC
7763 (Appendix O).  Accepted-corpus statistics are compared against
the paper's reported realizations (Appendix G) when the full corpus is
generated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace.corpus import write_manifest
from trace.defaults import (
    CORPUS_N_SELECT,
    EXPECTED_TEMPLATES_PER_KG,
    GEOMEAN_ENTITIES,
    GEOMEAN_RELATIONS,
    MEAN_EDGES_PER_KG,
    MEAN_TRAIN_QUERIES_PER_KG,
    REJECTION_RATE,
)
from trace.generator import SyntheticKGGenerator, sample_real_subgraph


def _gen_kwargs(gen: dict, cfg: dict) -> dict:
    kwargs = {}
    if "use_rules" in gen:
        kwargs["use_rules"] = bool(gen["use_rules"])
    if "no_rules_swap_factor" in gen:
        kwargs["no_rules_swap_factor"] = int(gen["no_rules_swap_factor"])
    if gen.get("fixed_stats"):
        kwargs["fixed_stats"] = dict(gen["fixed_stats"])
    if gen.get("template_pool"):
        kwargs["template_pool"] = list(gen["template_pool"])
    if gen.get("n_ent_range"):
        kwargs["n_ent_range"] = tuple(gen["n_ent_range"])
    if gen.get("mu_range"):
        kwargs["mu_range"] = tuple(gen["mu_range"])
    if gen.get("gamma_range"):
        kwargs["gamma_range"] = tuple(gen["gamma_range"])
    if "min_derived_train" in gen:
        kwargs["min_derived_train"] = int(gen["min_derived_train"])
    if "min_q_val" in gen:
        kwargs["min_q_val"] = int(gen["min_q_val"])
    if gen.get("target_edges_per_kg"):
        t = gen["target_edges_per_kg"]
        kwargs["target_edges_per_kg"] = list(t) if isinstance(t, list) else int(t)
    if gen.get("total_edge_budget"):
        kwargs["total_edge_budget"] = int(gen["total_edge_budget"])
        kwargs["n_kg"] = int(gen.get("n_kg", 8))
    if cfg.get("rule_form"):
        kwargs["rule_form"] = str(cfg["rule_form"])
    if cfg.get("epsilon_range"):
        kwargs["epsilon_range"] = tuple(cfg["epsilon_range"])
    return kwargs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, default=None,
                   help="override corpus_dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_kg", type=int, default=None)
    p.add_argument("--select_kg", type=int, default=CORPUS_N_SELECT)
    p.add_argument("--limit", type=int, default=None,
                   help="generate only the first N KGs (smoke runs)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    corpus_dir = args.out_dir or cfg["corpus_dir"]
    gen_cfg = dict(cfg.get("generator", {}) or {})
    n_kg = int(args.n_kg or gen_cfg.get("n_kg", 2000))
    if args.limit:
        n_kg = min(n_kg, args.limit)

    os.makedirs(os.path.join(corpus_dir, "kgs"), exist_ok=True)
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        print(f"manifest exists at {manifest_path}; pass --overwrite")
        sys.exit(1)

    rng_seed = int(args.seed)
    t0 = time.time()
    stats_rows = []

    sub_cfg = (cfg.get("real") or {}).get("subgraph")
    if sub_cfg:
        # control 4: random-walk subgraphs of the real pretraining trio
        from trace.evaluate import load_id_map, load_triples
        graphs = []
        base = sub_cfg.get("kg_dir", "data/pretraining")
        for gname in cfg.get("real", {}).get(
                "graphs", ["FB15k-237", "WN18RR", "CoDEx-M"]):
            d = os.path.join(base, gname)
            e2id = load_id_map(os.path.join(d, "entity2id.txt"))
            r2id = load_id_map(os.path.join(d, "relation2id.txt"))
            tr = load_triples(os.path.join(d, "train.txt"), e2id, r2id)
            graphs.append((tr, len(r2id)))
        rng = np.random.default_rng(rng_seed)
        kg_id, attempt, tried = 0, 0, 0
        while kg_id < n_kg and attempt < 50 * n_kg:
            tr, n_rel = graphs[tried % len(graphs)]
            tried += 1
            sg = sample_real_subgraph(
                kg_id, tr, n_rel, rng,
                walker_steps=int(sub_cfg.get("walker_steps", 80)),
                min_entities=int(sub_cfg.get("min_entities", 40)),
                seed=rng_seed)
            attempt += 1
            if sg is None:
                continue
            sg.save(os.path.join(corpus_dir, "kgs", f"kg_{kg_id:05d}.pkl"))
            stats_rows.append(dict(kg_id=kg_id,
                                   n_entities=sg.n_entities,
                                   n_relations=sg.n_relations,
                                   n_edges=sg.n_edges_train,
                                   n_train_queries=sg.n_train_queries,
                                   n_templates=0))
            kg_id += 1
            if kg_id % 100 == 0:
                print(f"  {kg_id}/{n_kg} subgraphs", flush=True)
        n_kg = kg_id
    else:
        kwargs = _gen_kwargs(gen_cfg, cfg)
        from concurrent.futures import ProcessPoolExecutor
        workers = int(os.environ.get("TRACE_WORKERS", "1"))

        def _job(i):
            g = SyntheticKGGenerator(seed=rng_seed, **kwargs)
            split = g.generate(i)
            split.save(os.path.join(corpus_dir, "kgs", f"kg_{i:05d}.pkl"))
            return dict(kg_id=i, n_entities=split.n_entities,
                        n_relations=split.n_relations,
                        n_edges=split.n_edges_train,
                        n_train_queries=split.n_train_queries,
                        n_templates=len(split.templates))

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, row in enumerate(ex.map(_job, range(n_kg))):
                    stats_rows.append(row)
                    if (i + 1) % 100 == 0:
                        print(f"  {i + 1}/{n_kg} KGs", flush=True)
        else:
            for i in range(n_kg):
                stats_rows.append(_job(i))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{n_kg} KGs", flush=True)

    train_ids = list(range(0, max(0, n_kg - args.select_kg)))
    select_ids = list(range(len(train_ids), n_kg))
    write_manifest(corpus_dir, rng_seed, train_ids, select_ids,
                   config=dict(arm=cfg.get("arm"), generator=gen_cfg))

    # --------------------------------------------------------------- #
    # corpus diagnostics vs. the accepted-corpus realizations (App. G) #
    # --------------------------------------------------------------- #
    ent = np.array([r["n_entities"] for r in stats_rows], dtype=np.float64)
    rel = np.array([r["n_relations"] for r in stats_rows], dtype=np.float64)
    edges = np.array([r["n_edges"] for r in stats_rows], dtype=np.float64)
    queries = np.array([r["n_train_queries"] for r in stats_rows],
                       dtype=np.float64)
    templ = np.array([r["n_templates"] for r in stats_rows], dtype=np.float64)
    geo = lambda a: float(np.exp(np.mean(np.log(np.maximum(a, 1)))))
    print()
    print(f"corpus: {n_kg} KGs ({len(train_ids)} train + "
          f"{len(select_ids)} selection) in {time.time() - t0:.0f}s")
    print(f"  geometric mean entities    {geo(ent):7.0f}   "
          f"(reported {GEOMEAN_ENTITIES})")
    print(f"  geometric mean relations   {geo(rel):7.0f}   "
          f"(reported {GEOMEAN_RELATIONS})")
    print(f"  mean training edges / KG   {edges.mean():7.0f}   "
          f"(reported {MEAN_EDGES_PER_KG})")
    print(f"  mean train queries / KG    {queries.mean():7.0f}   "
          f"(reported {MEAN_TRAIN_QUERIES_PER_KG})")
    if templ.sum():
        print(f"  mean templates / KG        {templ.mean():7.2f}   "
              f"(reported {EXPECTED_TEMPLATES_PER_KG})")
    print(f"  unique edges total         {edges.sum():7.0f}")
    print("  (full-corpus figures are the ones compared in Appendix G; "
          "--limit runs will not match)")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
