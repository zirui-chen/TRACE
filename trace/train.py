"""
Pretraining loop, identical across arms except for the corpus (§3.4).

One KG is sampled uniformly per step and the standard ULTRA forward
pass plus the self-adversarial loss are applied unchanged: AdamW with
lr 5e-4, batch of 64 queries, 128 self-adversarial negatives at
temperature 1.0, 200K steps, checkpoints every 10K steps (Appendix O).
Checkpoints are selected by mean validation MRR over the 200 held-out
KGs using G_r^val and Q_val; every pipeline choice is selected on
synthetic validation MRR, and real-graph results are computed once per
frozen configuration.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW

from .corpus import _as_tensor, batch_to_ultra
from .defaults import (
    ADVERSARIAL_TEMPERATURE,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    LR,
    N_NEGATIVES,
    N_STEPS,
    OPTIMIZER,
)

__all__ = ["train_trace", "evaluate_val"]


def train_trace(model,
                train_corpus,
                val_corpus,
                out_dir: str,
                n_steps: int = N_STEPS,
                lr: float = LR,
                batch_size: int = BATCH_SIZE,
                n_negatives: int = N_NEGATIVES,
                adversarial_temperature: float = ADVERSARIAL_TEMPERATURE,
                checkpoint_every: int = CHECKPOINT_EVERY,
                device: Optional[torch.device] = None,
                seed: int = 0,
                log_every: int = 1000) -> Optional[str]:
    """Pretrain the frozen ULTRA reasoner on one arm's corpus.

    ``train_corpus`` exposes ``sample_batch() -> dict`` (any of
    PremiseFaithfulCorpus / StandardSyntheticCorpus / *RealCorpus /
    MixedCorpus); ``val_corpus`` exposes ``val_batches()``.  Returns
    the path of the best checkpoint (model selection: mean validation
    MRR over the 200 held-out KGs, §3.4).
    """
    if OPTIMIZER != "adamw":
        raise ValueError("the released configuration fixes AdamW")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.makedirs(out_dir, exist_ok=True)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_mrr = -1.0
    best_ckpt_path = None
    t0 = time.time()

    for step in range(1, n_steps + 1):
        model.train()
        optimizer.zero_grad()

        batch = train_corpus.sample_batch()
        entity_graph, rel_graph, query, target = batch_to_ultra(batch, device)
        filter_triples = _as_tensor(batch["filter_triples"], device)

        loss = _step(model, entity_graph, rel_graph, query, target,
                     filter_triples, n_negatives, adversarial_temperature)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"step {step:>7d} | loss {loss.item():.4f} "
                  f"| {time.time() - t0:8.1f}s", flush=True)

        if step % checkpoint_every == 0:
            val_mrr = evaluate_val(model, val_corpus, device)
            elapsed = time.time() - t0
            print(f"step {step:>7d} | loss {loss.item():.4f} "
                  f"| syn-val MRR {val_mrr:.4f} | {elapsed / 60:.1f} min",
                  flush=True)
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_ckpt_path = os.path.join(out_dir,
                                              f"ckpt_step{step:07d}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "val_mrr": val_mrr, "seed": seed},
                           best_ckpt_path)

    print(f"training done: best syn-val MRR {best_val_mrr:.4f} "
          f"at {best_ckpt_path}", flush=True)
    return best_ckpt_path


# --------------------------------------------------------------------- #
# Self-adversarial negative sampling (released loss, untouched)          #
# --------------------------------------------------------------------- #

def _step(model, entity_graph, rel_graph, query, target,
          filter_triples, n_negatives, adversarial_temperature):
    """One self-adversarial step: score all entities, mask known-true
    triples out of the negative pool, sample 128 negatives per query
    proportional to their (detached) softmax, and optimize the
    log-sigmoid objective."""
    n_ent = entity_graph.num_nodes
    scores = model(entity_graph, rel_graph, query)          # [B, n_ent]
    B = query.size(0)

    # known-true mask: the target plus every filter triple sharing (h,r)
    mask = torch.zeros(B, n_ent, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, target.unsqueeze(1), True)
    if filter_triples.numel():
        f_h, f_r, f_t = (filter_triples[:, 0], filter_triples[:, 1],
                         filter_triples[:, 2])
        same = (f_h.unsqueeze(0) == query[:, 0].unsqueeze(1)) & \
               (f_r.unsqueeze(0) == query[:, 1].unsqueeze(1))
        rows, cols = same.nonzero(as_tuple=True)
        t_sel = f_t[cols]
        ok = t_sel < n_ent
        mask[rows[ok], t_sel[ok]] = True

    neg_scores = scores.masked_fill(mask, float("-inf"))
    adv_probs = torch.softmax(neg_scores / adversarial_temperature, dim=-1)
    adv_probs = adv_probs.detach()
    adv_probs = torch.nan_to_num(adv_probs, nan=0.0)
    # uniform fallback if a row was fully masked
    degenerate = adv_probs.sum(dim=1) <= 0
    if degenerate.any():
        adv_probs[degenerate] = 1.0 / max(n_ent, 1)
    neg_idx = torch.multinomial(adv_probs, min(n_negatives, n_ent - 1),
                                replacement=False)
    sampled_neg_scores = scores.gather(1, neg_idx)          # [B, k]
    neg_probs = adv_probs.gather(1, neg_idx)

    pos_scores = scores.gather(1, target.unsqueeze(1)).squeeze(1)
    pos_loss = -torch.nn.functional.logsigmoid(pos_scores).mean()
    neg_loss = -(neg_probs
                 * torch.nn.functional.logsigmoid(-sampled_neg_scores)
                 ).sum(dim=1).mean()
    return pos_loss + neg_loss


# --------------------------------------------------------------------- #
# Synthetic validation MRR (model selection)                             #
# --------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_val(model, val_corpus, device,
                 max_queries_per_kg: Optional[int] = 500) -> float:
    """Mean filtered MRR over the held-out selection KGs' Q_val."""
    model.eval()
    total, n = 0.0, 0
    for batch in val_corpus.val_batches():
        if len(batch["query_h"]) == 0:
            continue
        entity_graph, rel_graph, query, target = batch_to_ultra(batch, device)
        if max_queries_per_kg is not None and query.size(0) > max_queries_per_kg:
            sel = torch.randperm(query.size(0), device=query.device)[:max_queries_per_kg]
            query, target = query[sel], target[sel]
        scores = model(entity_graph, rel_graph, query)

        # filtered ranking: known-true (h, r, t') with t' != t go to -inf
        n_ent = entity_graph.num_nodes
        ft = _as_tensor(batch["filter_triples"], device)
        if ft.numel():
            f_h, f_r, f_t = ft[:, 0], ft[:, 1], ft[:, 2]
            same = (f_h.unsqueeze(0) == query[:, 0].unsqueeze(1)) & \
                   (f_r.unsqueeze(0) == query[:, 1].unsqueeze(1)) & \
                   (f_t.unsqueeze(0) != target.unsqueeze(1))
            rows_, cols_ = same.nonzero(as_tuple=True)
            t_sel = f_t[cols_]
            ok = t_sel < n_ent
            scores[rows_[ok], t_sel[ok]] = float("-inf")

        tgt = scores.gather(1, target.unsqueeze(1)).squeeze(1)
        better = (scores > tgt.unsqueeze(1)).sum(dim=1)
        total += float((1.0 / (better.float() + 1.0)).sum().item())
        n += query.size(0)
    return total / n if n else 0.0
