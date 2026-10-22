"""
Default hyper-parameters shared by every TRACE arm.

All arms (ULTRA-Real, TRACE-Real, ULTRA-Syn, TRACE) use the released
ULTRA configuration unchanged (Appendix O of the paper): 6 layers in
each of GNN_r and GNN_e, hidden dimension 64, AdamW with lr 5e-4,
self-adversarial negative sampling with the released number of
negatives (128) and temperature, batch size 64, and 200K steps.
Trainable parameters, counted as sum(p.numel()), number 168,070.

The only thing that differs across arms is the pretraining pipeline:
the corpus distribution p_C and the task distribution p_task.
"""

# --------------------------------------------------------------------- #
# ULTRA reasoner (frozen across every arm)                               #
# --------------------------------------------------------------------- #

NUM_LAYERS_REL = 6
NUM_LAYERS_ENTITY = 6
HIDDEN_DIM = 64
N_PARAMS = 168_070          # sum(p.numel()) of the released config

OPTIMIZER = "adamw"
LR = 5e-4
BATCH_SIZE = 64             # queries per step
N_NEGATIVES = 128           # released self-adversarial negative count
ADVERSARIAL_TEMPERATURE = 1.0
N_STEPS = 200_000
CHECKPOINT_EVERY = 10_000

# --------------------------------------------------------------------- #
# Premise-faithful task construction (§3.2)                              #
# --------------------------------------------------------------------- #

DERIVED_TRAIN_FRAC = 0.60   # d_train / d_val / d_test = 60 / 20 / 20
DERIVED_VAL_FRAC   = 0.20
DERIVED_TEST_FRAC  = 0.20

BG_TRAIN_FRAC = 0.80        # b_train / b_val / b_test = 80 / 10 / 10
BG_VAL_FRAC   = 0.10
BG_TEST_FRAC  = 0.10

# Each 64-query batch contains 32 derived + 32 non-derived queries.
# 50/50 is the synthetic-validation argmax (Table 10a).
DERIVED_FRAC_IN_BATCH = 0.5

# --------------------------------------------------------------------- #
# Real-KG instantiation (§3.2, Appendix I)                               #
# --------------------------------------------------------------------- #

ANYBURL_TIMEOUT_S    = 100      # per graph
ANYBURL_MAX_RULE_LEN = 3
ANYBURL_MIN_CONF     = 0.8
ANYBURL_MIN_SUPPORT  = 10
# 1,240 rules are retained across the three real graphs; derived triples
# cover 23% of training triples.
EXPECTED_TOTAL_RULES = 1_240
EXPECTED_DERIVED_COVERAGE = 0.23

REAL_PRETRAINING_GRAPHS = ("FB15k-237", "WN18RR", "CoDEx-M")
REAL_TRAIN_SPLITS_EDGES = (272_115, 86_835, 185_584)   # = 544,534 canonical

# --------------------------------------------------------------------- #
# Factorized generator (§3.3, Appendix N / Algorithm 1)                  #
# --------------------------------------------------------------------- #

LEVEL_PROB = {1: 0.70, 2: 0.62, 3: 0.44}   # per-level inclusion probs
EXPECTED_TEMPLATES_PER_KG = 4.2
FALLBACK_TEMPLATE = "T4"                    # composition if draw is empty

DERIVED_CAP_PER_TEMPLATE = 50_000
STUB_REMOVAL_RATE = 0.061                   # self-loops / duplicates dropped

# Acceptance filter Acc(G) (Step 8 of Algorithm 1)
MIN_LCC_FRACTION   = 0.5
MAX_OBS_EDGES      = 500_000
MIN_DERIVED_TRAIN  = 100
MIN_Q_VAL          = 50
REJECTION_RATE     = 0.072                  # ~2,155 draws for 2,000 KGs

# Accepted-corpus realisations (Table 7 / Appendix G)
GEOMEAN_ENTITIES  = 1_730
GEOMEAN_RELATIONS = 38
MEAN_EDGES_PER_KG = 79_000
MEAN_TRAIN_QUERIES_PER_KG = 21_300
UNIQUE_TRAIN_EDGES_TOTAL  = 128_000_000     # ~128 M over the 1,800 KGs
VISITS_PER_KG     = 111                     # 200K steps x 64 / 1,800 KGs

CORPUS_N_TOTAL  = 2_000
CORPUS_N_TRAIN  = 1_800
CORPUS_N_SELECT = 200                       # held-out model-selection KGs

# Corpus C/D fixed statistics (prior medians, §4.2)
FIXED_STATS = dict(
    n_ent=1732,
    n_rel=34,
    kappa=1.0,
    gamma=2.25,
    mu=12.2,
    eta=0.25,
    nu=0.05,
)

# Corpus B rewiring (Appendix N): 10 |E_r| degree-preserving
# directed double-edge swaps per head relation.
NO_RULES_SWAP_FACTOR = 10
NO_RULES_RESIDUAL_RULE_FRACTION = 0.004     # rewired edges still grounded: 0.4%

# --------------------------------------------------------------------- #
# Statistical protocol (Appendix C)                                      #
# --------------------------------------------------------------------- #

BOOTSTRAP_REPLICATES = 10_000
NON_INFERIORITY_MARGIN = 0.01               # delta, fixed before evaluation
TIE_BAND = 0.005                            # +/-0.005 for win/tie/loss counts
SEEDS_MAIN = (0, 1, 2, 3, 4)
SEEDS_SHORT = (0, 1, 2)                     # 3-seed experiments

# --------------------------------------------------------------------- #
# Compute (Appendix O)                                                   #
# --------------------------------------------------------------------- #

GPU_HOURS_ULTRA_REAL = 48                   # per seed
GPU_HOURS_TRACE      = 64                   # per seed
CORPUS_GEN_MINUTES   = 45                   # +/- 5, 8-core AMD EPYC 7763
