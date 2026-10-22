# TRACE: Premise-Faithful Supervision for Zero-Shot Knowledge Graph Reasoning

## What TRACE changes

Zero-shot KG reasoning systems pretrain on a corpus of graphs and are then
evaluated on unseen graphs. The ULTRA reasoner (Galkin et al., ICLR 2021) is
kept **frozen and unchanged** — architecture, loss, optimizer, negative
sampling, and the 200K-step budget (official release, commit `3f2c1a9`,
config `transductive/pretrain_3g.yaml`: 6+6 GNN layers, hidden dimension 64,
168,070 trainable parameters). TRACE changes only the **pretraining
pipeline**, along two axes that are varied independently (a corpus ×
supervision factorial, §4.1):

| Arm | Corpus p_C | Task construction p_task |
|---|---|---|
| ULTRA-Real | 3 real KGs (FB15k-237, WN18RR, CoDEx-M) | standard |
| TRACE-Real | 3 real KGs | premise-faithful (AnyBURL rules) |
| ULTRA-Syn | 1,800 synthetic KGs | standard (pooled 80/10/10) |
| **TRACE** | 1,800 synthetic KGs | **premise-faithful** |

**Premise-faithful task construction** assigns observable/query roles by
derivational provenance: the body (premise) triples of each rule stay
permanently observable and are never used as query targets, while rule
conclusions serve as prediction targets. This fixes the *premise hiding* of
random 80/10/10 splits, which routinely hides rule bodies from the input and
asks the model to guess them.

* On **real KGs** (§3.2, App. I), rule bodies come from AnyBURL-mined rules
  (100 s/graph, length ≤ 3, confidence ≥ 0.8, support ≥ 10; 1,240 retained
  rules; derived triples cover 23% of training triples).
* On **synthetic KGs** (§3.3, App. N), provenance is *exact*: a factorized
  generator samples rule templates T and graph statistics S independently —
  `p(T, S) = p(T) p(S)` — so the corpus itself disentangles rule content
  from topology (Algorithm 1: `trace/generator.py`).

## Headline results (54 benchmark KGs, 5 seeds; paper Table 1)

| | All | Disjoint-32 | Proximal-22 | Ext-hard-14 |
|---|---|---|---|---|
| ULTRA-Real | .364 | .348 | .387 | .161 |
| TRACE-Real | .366 | .352 | .387 | .165 |
| ULTRA-Syn | .355 | .354 | .356 | .177 |
| **TRACE** | **.369** | **.369** | .369 | **.195** |

* Pipeline effect (TRACE − ULTRA-Real): **+0.005** all, **+0.020** disjoint,
  −0.018 proximal, **+0.034** external-hard.
* Non-inferiority established at δ = 0.01 (wild cluster bootstrap one-sided
  lower bound −0.006); wins on 30/54 KGs (sign test p = 0.25); with the
  ±0.005 tie band: 25 wins / 11 ties / 18 losses.
* Corpus × supervision interaction: +0.012 [0.004, 0.020] — premise-faithful
  supervision pays off far more on the synthetic corpus than on the real
  graphs.
* H@1 / H@10: .276 / .523 vs ULTRA-Real's .271 / .517.
* Per-setting reproduction of published ULTRA(3g): .312/.428/.343 vs
  published .312/.431/.345.

---

## Repository structure

```text
TRACE/
├── trace/
│   ├── defaults.py          # every frozen hyper-parameter, one contract file
│   ├── benchmarks.py        # Table 5 manifest: 54 graphs, 3 slices, 11 clusters
│   ├── generator.py         # factorized generator, Algorithm 1 (App. N)
│   ├── corpus.py            # PremiseFaithfulCorpus / StandardSyntheticCorpus / MixedCorpus
│   ├── real_supervision.py  # AnyBURL rule mining + premise-faithful real arm
│   ├── ultra.py             # reasoner adapter (official ULTRA preferred)
│   ├── train.py             # 200K-step loop, self-adversarial loss (unchanged)
│   ├── evaluate.py          # filtered ranking, slice reporting
│   └── analysis/
│       ├── bootstrap.py     # wild cluster bootstrap (Webb weights), NI test
│       ├── distance.py      # JS degree/relation-graph distances, Eq. 5 fit
│       └── probe.py         # Table 6 probes on frozen relation embeddings
├── scripts/
│   ├── generate_corpus.py   # generate any arm's corpus from its config
│   ├── mine_rules.py        # AnyBURL mining -> frozen per-graph rule caches
│   ├── train.py             # train one arm
│   ├── evaluate.py          # zero-shot eval on the 54-graph suite
│   ├── verify_numbers.py    # recompute every table cell from raw per-KG CSV
│   ├── analyze_results.py   # Fig. 4/6: Eq. 5 distance interaction
│   └── compute_benchmark_stats.py   # Fig. 3: benchmark vs corpus statistics
├── configs/
│   ├── ultra_real.yaml  trace_real.yaml  ultra_syn.yaml  trace.yaml
│   ├── ablation_[BCD].yaml                      # Table 3 cells
│   ├── control_*.yaml                           # Table 4 (6 controls)
│   ├── templates/   (10 configs)                # Table 9
│   └── sensitivity/ (15 configs)                # Table 10
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
conda create -n trace python=3.9 && conda activate trace
pip install -r requirements.txt          # numpy, scipy, torch, torch-geometric, pyyaml, tqdm
pip install -e .
```

**Reasoner.** All paper numbers use the official ULTRA code:

```bash
git clone https://github.com/DeepGraphLearning/ULTRA
cd ULTRA && git checkout 3f2c1a9 && pip install -e .
```

`trace/ultra.py` imports it when present. A bundled clean-room fallback
(interface-compatible, same 6+6/64 configuration at the interface level)
lets the data pipeline and analysis modules run standalone; it is not what
produced the paper's tables.

**AnyBURL** (only for the real-KG arms; Java 8+):

```bash
# https://web.imm.dtu.dk/anyburl/  -> anyburl.jar
python scripts/mine_rules.py --data_dir data/pretraining --rules_dir data/rules \
    --jar /path/to/anyburl.jar
# or reuse the frozen release set: place the cached JSONs in data/rules/
```

Mining runs 100 s/graph and retains rules with confidence ≥ 0.8 and support
≥ 10; the released run keeps 1,240 rules total, with derived triples
covering 23% of training triples. Subsequent runs read the JSON caches and
never re-mine (this is how the paper's frozen rule distribution is used).

## Data layout

```text
data/
├── pretraining/{FB15k-237,WN18RR,CoDEx-M}/
│     train.txt  entity2id.txt  relation2id.txt      (tab-separated)
├── benchmarks/<54 Table-5 graph names>/
│     train.txt  val.txt  test.txt  entity2id.txt  relation2id.txt
├── rules/                        # AnyBURL caches (<Graph>.rules.json)
└── corpus/                       # generated corpora (see configs)
```

The full 54-graph manifest (graph set, source slice, external-hard flag,
size, published MRRs) lives in `trace/benchmarks.py`; the slice definitions
(source-proximal 22, source-disjoint 32, external-hard 14) and the 11
provenance clusters used by the wild cluster bootstrap are asserted there.

## Compute

≈ 9,460 GPU-hours total (core experiments ≈ 3,880) on 2× NVIDIA H100:
main comparison 560, non-inferiority/bootstrap reuse the same runs; 2×2
ablation 960; corpus controls ≈ 1,780; template ablations ≈ 1,920;
sensitivity ≈ 2,880; evaluation ≈ 150; corpus generation ≈ 6 CPU-hours.

## Datasets

FB15k-237 (Toutanova et al., 2015), WN18RR (Dettmers et al., 2018), CoDEx
(Safavi & Koutra, 2020), NELL-995 (Xiong et al., 2017), Hetionet
(Himmelstein et al., 2017), InGram benchmarks (Lee et al., 2023), and the
remaining Table 5 sources; standard zero-shot splits as distributed with
the ULTRA benchmarking suite.

## License

MIT. See `LICENSE`.
