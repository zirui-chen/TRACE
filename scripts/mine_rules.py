#!/usr/bin/env python3
"""
Mine AnyBURL rules for the TRACE-Real arm (paper Sec. 3.2, Appendix I).

Runs the frozen mining protocol on the three real pretraining graphs
(FB15k-237, WN18RR, CoDEx-Medium; trace.defaults.REAL_PRETRAINING_GRAPHS):
AnyBURL, 100 s per graph, rule length <= 3, keeping rules with
confidence >= 0.8 and support >= 10.  The retained set (1,240 rules in
the paper's frozen run, covering 23% of training triples) is cached as
JSON under --rules_dir so mining never has to be re-run; the corpus
loader (trace.real_supervision.PremiseFaithfulRealCorpus) reads those
caches directly.

Usage
-----
# First-time mining (requires the AnyBURL jar + a JVM; see
# trace/real_supervision.py for download instructions):
python scripts/mine_rules.py --data_dir data/pretraining \
    --rules_dir data/rules --jar /path/to/anyburl.jar

# Re-print counts from the cached (frozen) rule JSON, no Java needed:
python scripts/mine_rules.py --data_dir data/pretraining \
    --rules_dir data/rules
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from trace.defaults import (  # noqa: E402
    ANYBURL_MAX_RULE_LEN,
    ANYBURL_MIN_CONF,
    ANYBURL_MIN_SUPPORT,
    ANYBURL_TIMEOUT_S,
    EXPECTED_DERIVED_COVERAGE,
    EXPECTED_TOTAL_RULES,
    REAL_PRETRAINING_GRAPHS,
)
from trace.real_supervision import (  # noqa: E402
    RealKG,
    construct_premise_faithful_tasks,
    load_rules_json,
    mine_rules,
    rules_json_path,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Mine/cache AnyBURL rules for the TRACE-Real arm")
    p.add_argument("--data_dir", type=str, default="data/pretraining",
                   help="Parent dir holding one subdirectory per real KG")
    p.add_argument("--rules_dir", type=str, default="data/rules",
                   help="Directory for per-graph rule JSON caches")
    p.add_argument("--jar", type=str, default=None,
                   help="Path to anyburl.jar (else env ANYBURL_JAR or "
                        "anyburl.jar next to the KG files)")
    p.add_argument("--java", type=str, default="java",
                   help="Java launcher (default: java)")
    p.add_argument("--force", action="store_true",
                   help="Re-run the miner even when a cache exists")
    p.add_argument("--min_conf", type=float, default=ANYBURL_MIN_CONF,
                   help="Minimum rule confidence (paper: 0.8)")
    p.add_argument("--min_support", type=int, default=ANYBURL_MIN_SUPPORT,
                   help="Minimum rule support (paper: 10)")
    p.add_argument("--max_len", type=int, default=ANYBURL_MAX_RULE_LEN,
                   help="Maximum AnyBURL rule length (paper: 3)")
    p.add_argument("--timeout", type=int, default=ANYBURL_TIMEOUT_S,
                   help="AnyBURL timeout per graph in seconds (paper: 100)")
    return p.parse_args()


def main():
    args = parse_args()
    print("AnyBURL rule mining / cache report (Sec. 3.2, Appendix I)")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  rules_dir  : {args.rules_dir}")
    print(f"  filters    : conf >= {args.min_conf}, support >= "
          f"{args.min_support}, len <= {args.max_len}, "
          f"timeout {args.timeout} s/graph")
    print()

    total_rules = 0
    coverages = []
    missing = []
    for graph in REAL_PRETRAINING_GRAPHS:
        kg_dir = os.path.join(args.data_dir, graph)
        out_json = rules_json_path(args.rules_dir, graph)

        if not os.path.isdir(kg_dir):
            print(f"  {graph:<12} MISSING: no KG directory {kg_dir}")
            missing.append(graph)
            continue

        cached = os.path.exists(out_json) and not args.force
        rules = mine_rules(
            kg_dir, out_json,
            jar_path=args.jar, java=args.java,
            timeout_s=args.timeout, max_len=args.max_len,
            min_conf=args.min_conf, min_support=args.min_support,
            force=args.force,
        )
        source = "cache" if cached else "mined"

        kg = RealKG(kg_dir)
        tasks = construct_premise_faithful_tasks(
            kg.train, rules, relation2id=kg.relation2id)
        total_rules += len(rules)
        coverages.append(tasks["coverage"])
        print(f"  {graph:<12} rules={len(rules):>5} ({source:>5})  "
              f"|train|={len(kg.train):>7}  "
              f"derived={tasks['n_derived']:>7}  "
              f"premises={len(tasks['premises']):>7}  "
              f"coverage={tasks['coverage']:.3f}")

    print()
    n_seen = len(coverages)
    mean_cov = float(np.mean(coverages)) if coverages else float("nan")
    print(f"  TOTAL RULES      : {total_rules}   "
          f"(paper expectation: {EXPECTED_TOTAL_RULES}, "
          f"diff {total_rules - EXPECTED_TOTAL_RULES:+d})")
    print(f"  MEAN COVERAGE    : {mean_cov:.3f}   "
          f"(paper expectation: {EXPECTED_DERIVED_COVERAGE:.2f})")
    if missing:
        print(f"\n  WARNING: missing graphs: {', '.join(missing)}")
    if n_seen and n_seen < len(REAL_PRETRAINING_GRAPHS):
        print("  WARNING: totals above are partial (see missing graphs)")
    if n_seen == len(REAL_PRETRAINING_GRAPHS):
        ok_rules = abs(total_rules - EXPECTED_TOTAL_RULES) <= 0.05 * EXPECTED_TOTAL_RULES
        ok_cov = abs(mean_cov - EXPECTED_DERIVED_COVERAGE) <= 0.02
        if not (ok_rules and ok_cov):
            print("  NOTE: counts deviate from the frozen paper run; re-mine "
                  "with --force or restore the shipped data/rules caches.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
