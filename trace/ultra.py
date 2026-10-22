"""
ULTRA reasoner adapter (frozen across every arm; Appendix O).

Every reported run uses the officially released ULTRA code
(commit ``3f2c1a9``, config ``transductive/pretrain_3g.yaml``):
6 layers in each of GNN_r and GNN_e, hidden dimension 64, 168,070
trainable parameters (``sum(p.numel())`` of the released config).
TRACE changes nothing in this module's semantics -- the paper's claim
is that only the pretraining pipeline (p_C, p_task) differs across
arms.

``build_reasoner`` prefers the official package (``pip install`` the
ULTRA repo, or clone it and export PYTHONPATH as described in the
README) and only falls back to the bundled clean-room reimplementation
below when the official code is absent.  The bundled model reproduces
the released architecture at the interface level (relation
parameterization conditioned on G_r, entity parameterization conditioned
on the observed entity graph, DistMult-style scoring) so the data
pipeline, corpora, and analysis code are testable standalone; it is NOT
byte-identical to the released weights module and is not what produced
the paper's tables.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .defaults import (
    HIDDEN_DIM,
    NUM_LAYERS_ENTITY,
    NUM_LAYERS_REL,
)

__all__ = ["build_reasoner", "BundledULTRA"]

_OFFICIAL_IMPORT_HINT = (
    "Official ULTRA not found on PYTHONPATH. Clone the release at commit "
    "3f2c1a9 and `export PYTHONPATH=$PWD/ULTRA/src` (see README), or rely "
    "on the bundled fallback with allow_fallback=True."
)


def build_reasoner(rel_model_cfg: Optional[Dict] = None,
                   entity_model_cfg: Optional[Dict] = None,
                   num_relations: int = 256,
                   num_entities: int = 400_000,
                   allow_fallback: bool = True):
    """Instantiate the reasoner; official ULTRA preferred (see module
    docstring).  Both models expose ``forward(entity_graph, rel_graph,
    query) -> [B, n_entities]``."""
    rel_model_cfg = dict(rel_model_cfg or {})
    entity_model_cfg = dict(entity_model_cfg or {})
    rel_model_cfg.setdefault("num_layers", NUM_LAYERS_REL)
    rel_model_cfg.setdefault("hidden_dim", HIDDEN_DIM)
    entity_model_cfg.setdefault("num_layers", NUM_LAYERS_ENTITY)
    entity_model_cfg.setdefault("hidden_dim", HIDDEN_DIM)
    try:
        from ultra import ULTRA  # type: ignore
        return ULTRA(rel_model_cfg=rel_model_cfg,
                     entity_model_cfg=entity_model_cfg)
    except ImportError:
        if not allow_fallback:
            raise ImportError(_OFFICIAL_IMPORT_HINT)
        return BundledULTRA(num_relations=num_relations,
                            num_entities=num_entities,
                            num_layers_rel=rel_model_cfg["num_layers"],
                            num_layers_ent=entity_model_cfg["num_layers"],
                            hidden_dim=rel_model_cfg["hidden_dim"])


class _TypedGNN(nn.Module):
    """Message passing over the binary relation graph with the four
    interaction types + self-loop, ReLU residual updates (as in the
    released relation GNN)."""

    def __init__(self, hidden: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        # 4 interaction types + self-loop, one linear per type per layer
        self.msg = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden, hidden, bias=False)
                           for _ in range(5)])
            for _ in range(num_layers)
        ])
        self.upd = nn.ModuleList([nn.Linear(hidden, hidden)
                                  for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_type):
        n = x.size(0)
        for layer in range(self.num_layers):
            agg = torch.zeros_like(x)
            src, dst = edge_index[0], edge_index[1]
            for t in range(5):
                sel = edge_type == t
                if not torch.any(sel):
                    continue
                m = self.msg[layer][t](x[src[sel]])
                agg.index_add_(0, dst[sel], m)
            x = torch.relu(self.upd[layer](x + agg))
        return x


class BundledULTRA(nn.Module):
    """Clean-room fallback reasoner (interface-compatible; see module
    docstring).  Relation side: a learnable relation table feeds
    ``_TypedGNN`` over G_r; the query relation's trajectory across
    layers is concatenated into r_q.  Entity side: a learnable entity
    table feeds ``_TypedGNN`` over the observed entity graph with the
    query-conditioned initial feature; scores are DistMult-style
    dot products against a projected r_q."""

    def __init__(self, num_relations: int = 256, num_entities: int = 400_000,
                 num_layers_rel: int = NUM_LAYERS_REL,
                 num_layers_ent: int = NUM_LAYERS_ENTITY,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers_rel = num_layers_rel
        self.num_layers_ent = num_layers_ent
        self.rel_table = nn.Embedding(num_relations, hidden_dim)
        self.ent_table = nn.Embedding(num_entities, hidden_dim)
        self.gnn_r = _TypedGNN(hidden_dim, num_layers_rel)
        self.gnn_e = _TypedGNN(hidden_dim, num_layers_ent)
        self.rq_proj = nn.Linear(hidden_dim * num_layers_rel, hidden_dim)
        nn.init.xavier_uniform_(self.rel_table.weight)
        nn.init.xavier_uniform_(self.ent_table.weight)

    def _rel_traj(self, n_rel: int, q_emb: torch.Tensor,
                  ei: torch.Tensor, et: torch.Tensor):
        """Relation-GNN trajectory for ONE query relation: every node
        is initialized with the query relation embedding (relation
        parameterization), then the typed 6-layer pass runs; the
        trajectory is concatenated into r_q."""
        x = q_emb.unsqueeze(0).expand(n_rel, -1).contiguous()
        traj = []
        src, dst = ei[0], ei[1]
        for layer in range(self.num_layers_rel):
            agg = torch.zeros_like(x)
            for t in range(5):
                sel = et == t
                if not torch.any(sel):
                    continue
                m = self.gnn_r.msg[layer][t](x[src[sel]])
                agg.index_add_(0, dst[sel], m)
            x = torch.relu(self.gnn_r.upd[layer](x + agg))
            traj.append(x)
        return traj

    def relation_embedding(self, rel_graph, query_rel: torch.Tensor):
        if isinstance(rel_graph, dict):
            ei, et = rel_graph["edge_index"], rel_graph["edge_type"]
            n_rel = rel_graph["num_nodes"]
        else:
            ei, et, n_rel = (rel_graph.edge_index, rel_graph.edge_type,
                             rel_graph.num_nodes)
        # tolerate inverse-augmented G_r (real arms) as well as the
        # canonical cached G_r (synthetic arms)
        n_rel = int(max(n_rel,
                        int(ei.max().item()) + 1 if ei.numel() else 0,
                        int(query_rel.max().item()) + 1
                        if query_rel.numel() else 0))
        n_rel = min(n_rel, self.rel_table.num_embeddings)
        q_emb = self.rel_table(query_rel.clamp(max=self.rel_table.num_embeddings - 1))
        ei = ei.clamp(max=n_rel - 1)
        r_q = []
        for b in range(query_rel.size(0)):
            traj = self._rel_traj(n_rel, q_emb[b], ei, et)
            r_q.append(torch.cat([t[query_rel[b]] for t in traj], dim=-1))
        return torch.stack(r_q)                          # [B, d*L]

    def forward(self, entity_graph, rel_graph, query: torch.Tensor):
        ei = entity_graph.edge_index
        n_ent = entity_graph.num_nodes
        h_e, q_r = query[:, 0], query[:, 1]
        r_q = self.relation_embedding(rel_graph, q_r)      # [B, d*L]
        r_proj = torch.tanh(self.rq_proj(r_q))             # [B, d]

        # entity side: parameterized once over the observed graph
        # (canonical edges; the model adapter treats them as type 0)
        x0 = self.ent_table.weight[:n_ent]                 # [n_ent, d]
        ent_types = torch.zeros(ei.size(1), dtype=torch.long,
                                device=ei.device)
        x = self.gnn_e(x0, ei, ent_types)
        # DistMult-style score against the query-conditioned anchor
        e_h = x[h_e.clamp(max=n_ent - 1)]                  # [B, d]
        scores = (e_h * r_proj) @ x.T                      # [B, n_ent]
        return scores
