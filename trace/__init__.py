"""
TRACE: Premise-Faithful Supervision for Zero-Shot Knowledge Graph Reasoning.

TRACE fixes the ULTRA reasoner (architecture, loss, optimizer, negative
sampling, 200K-step budget, zero-shot inference protocol) and varies only
the pretraining pipeline: the corpus p_C (real vs. synthetic) and the task
construction p_task (standard vs. premise-faithful).

Premise-faithful task construction assigns observable/query roles by
derivational provenance:
  * on real KGs, rule bodies come from AnyBURL-mined rules (§3.2);
  * on synthetic KGs, provenance is exact, from the factorized generator
    that samples rule templates and graph statistics independently (§3.3,
    Algorithm 1 in trace/generator.py).

Heavy imports are lazy so that `import trace` stays cheap.
"""

__version__ = "1.0.0"


def __getattr__(name):
    if name == "SyntheticKGGenerator":
        from .generator import SyntheticKGGenerator
        return SyntheticKGGenerator
    if name == "KGSplit":
        from .generator import KGSplit
        return KGSplit
    if name == "PremiseFaithfulCorpus":
        from .corpus import PremiseFaithfulCorpus
        return PremiseFaithfulCorpus
    if name == "StandardSyntheticCorpus":
        from .corpus import StandardSyntheticCorpus
        return StandardSyntheticCorpus
    if name == "train_trace":
        from .train import train_trace
        return train_trace
    if name == "mine_rules" :
        from .real_supervision import mine_rules
        return mine_rules
    if name == "PremiseFaithfulRealCorpus":
        from .real_supervision import PremiseFaithfulRealCorpus
        return PremiseFaithfulRealCorpus
    raise AttributeError(f"module 'trace' has no attribute {name!r}")


__all__ = [
    "SyntheticKGGenerator",
    "KGSplit",
    "PremiseFaithfulCorpus",
    "StandardSyntheticCorpus",
    "train_trace",
    "mine_rules",
    "PremiseFaithfulRealCorpus",
]
