#!/usr/bin/env python3
"""
Zero-shot evaluation of a checkpoint on the 54-graph suite (§4.1).

Runs filtered ranking on every benchmark KG, reports the three slices
(source-proximal 22 / source-disjoint 32 / external-hard 14) and
appends per-KG rows to the results CSV consumed by
scripts/verify_numbers.py.

Example
-------
    python scripts/evaluate.py --config configs/trace.yaml \
        --ckpt checkpoints/trace/seed0/ckpt_step0120000.pt \
        --kg_dir data/benchmarks --out_csv results/all_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace.evaluate import evaluate_benchmark_suite, write_results_csv
from trace.ultra import build_reasoner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="arm config used for training (model shape + arm name)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--kg_dir", type=str, default="data/benchmarks")
    p.add_argument("--out_csv", type=str, default="results/all_results.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_relations", type=int, default=256)
    p.add_argument("--num_entities", type=int, default=400_000)
    p.add_argument("--kg_names", type=str, nargs="*", default=None,
                   help="evaluate a subset of the 54 graphs")
    p.add_argument("--append", action="store_true",
                   help="append rows to an existing CSV")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    arm = cfg.get("arm", "trace")
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_reasoner(
        rel_model_cfg=cfg.get("rel_model"),
        entity_model_cfg=cfg.get("entity_model"),
        num_relations=args.num_relations,
        num_entities=args.num_entities,
        allow_fallback=True)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    print(f"loaded {args.ckpt} "
          f"(step {state.get('step', '?')}, "
          f"syn-val MRR {state.get('val_mrr', float('nan')):.4f})")

    out = evaluate_benchmark_suite(
        model, args.kg_dir, device,
        batch_size=args.batch_size,
        kg_names=args.kg_names,
        seed=args.seed,
        method=arm,
    )
    write_results_csv(out["rows"], args.out_csv, append=args.append)
    print(f"wrote {len(out['rows'])} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
