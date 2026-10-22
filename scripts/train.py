#!/usr/bin/env python3
"""
Pretrain one arm of the corpus x supervision factorial (§4.1).

All arms share the frozen ULTRA configuration (Appendix O) and differ
only in (corpus, supervision), selected by the config's ``arm`` /
``supervision`` keys:

    ultra_real   real graphs, standard task construction
    trace_real   real graphs + AnyBURL premise-faithful tasks
    ultra_syn    TRACE synthetic corpus, standard (pooled 80/10/10)
    trace        TRACE synthetic corpus, premise-faithful (50/50)
    ablation_[ABCD] / control_*   same dispatch, own corpus_dir

Model selection uses mean synthetic validation MRR over the 200
held-out KGs (synthetic arms) or the real validation splits (real
arms), as specified in §3.4 / Appendix I.

Example
-------
    python scripts/train.py --config configs/trace.yaml --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace.corpus import (
    MixedCorpus,
    PremiseFaithfulCorpus,
    StandardSyntheticCorpus,
)
from trace.defaults import REAL_PRETRAINING_GRAPHS
from trace.train import train_trace
from trace.ultra import build_reasoner


def _real_dirs(cfg):
    real = cfg.get("real", {}) or {}
    base = real.get("kg_dir", "data/pretraining")
    graphs = real.get("graphs", list(REAL_PRETRAINING_GRAPHS))
    return [os.path.join(base, g) for g in graphs], base, graphs


def build_corpora(cfg, seed):
    """Return (train_corpus, val_corpus) for the config's arm."""
    arm = cfg.get("arm", "trace")
    corpus_dir = cfg["corpus_dir"]
    batch = int(cfg.get("batch_size", 64))
    derived_frac = float(cfg.get("derived_frac", 0.5))

    synthetic_pf = lambda: PremiseFaithfulCorpus(
        corpus_dir, seed=seed, batch_size=batch,
        derived_frac=derived_frac,
        split_variant=cfg.get("split_ablation", "provenance"),
        filter_set=cfg.get("filter_set", "full"),
        q_val_includes_noise=bool(cfg.get("q_val_includes_noise", False)))
    synthetic_std = lambda: StandardSyntheticCorpus(
        corpus_dir, seed=seed, batch_size=batch)

    if arm == "ultra_real":
        kg_dirs, _, _ = _real_dirs(cfg)
        from trace.real_supervision import StandardRealCorpus
        c = StandardRealCorpus(kg_dirs, batch_size=batch, seed=seed)
        return c, c
    if arm == "trace_real":
        kg_dirs, _, _ = _real_dirs(cfg)
        rules_dir = (cfg.get("real") or {}).get("rules_dir", "data/rules")
        from trace.real_supervision import PremiseFaithfulRealCorpus
        c = PremiseFaithfulRealCorpus(kg_dirs, rules_dir, batch_size=batch,
                                      derived_frac=derived_frac, seed=seed)
        return c, c
    if arm == "ultra_syn" or (arm.startswith("control_")
                              and cfg.get("supervision") == "standard"):
        return _mk_std(synthetic_std, seed)
    if arm == "control_mixture":
        kg_dirs, _, _ = _real_dirs(cfg.get("mixture", {}))
        from trace.real_supervision import StandardRealCorpus
        mix = cfg.get("mixture", {}) or {}
        real = StandardRealCorpus(kg_dirs, batch_size=batch, seed=seed)
        syn = synthetic_pf()
        c = MixedCorpus(real, syn,
                        prob_real=float(mix.get("prob_real", 0.5)),
                        seed=seed)
        return c, syn
    # default: premise-faithful synthetic arms (trace, ablation_*,
    # control_3cycled / edge_matched / many_small / compute_matched)
    c = synthetic_pf()
    return c, c


def _mk_std(factory, seed):
    c = factory()
    return c, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--checkpoint_every", type=int, default=None,
                   help="override checkpoint/validation cadence "
                        "(default: config value, frozen at 10000)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_relations", type=int, default=256)
    p.add_argument("--num_entities", type=int, default=400_000)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_corpus, val_corpus = build_corpora(cfg, args.seed)

    model = build_reasoner(
        rel_model_cfg=cfg.get("rel_model"),
        entity_model_cfg=cfg.get("entity_model"),
        num_relations=args.num_relations,
        num_entities=args.num_entities,
        allow_fallback=True)

    out_dir = args.out_dir or os.path.join(
        cfg.get("out_dir", "checkpoints"), f"seed{args.seed}")
    best = train_trace(
        model, train_corpus, val_corpus,
        out_dir=out_dir,
        n_steps=int(args.n_steps or cfg.get("n_steps", 200_000)),
        lr=float(cfg.get("lr", 5e-4)),
        batch_size=int(cfg.get("batch_size", 64)),
        n_negatives=int(cfg.get("n_negatives", 128)),
        adversarial_temperature=float(cfg.get("adversarial_temperature", 1.0)),
        checkpoint_every=int(args.checkpoint_every
                             or cfg.get("checkpoint_every", 10_000)),
        device=device,
        seed=args.seed,
    )
    print(best or "no checkpoint beat syn-val MRR 0")


if __name__ == "__main__":
    main()
