"""
Premise-faithful supervision on the three real pretraining graphs
(ULTRA-Real vs. TRACE-Real arms; paper Sec. 3.2, Sec. 4.1, Appendix I).

This module implements the real-KG instantiation of premise-faithful task
construction and the two real-corpus data loaders that consume it.  Per
Sec. 3.2 / Appendix I:

  * AnyBURL (Galkin et al., 2020) runs on each real training split for
    100 s per graph with rule length <= 3; rules with confidence >= 0.8
    and support >= 10 are retained -- 1,240 rules in total across
    FB15k-237, WN18RR and CoDEx-Medium (trace.defaults).
  * Each training triple that is predicted by a retained rule and is
    present in the training split becomes a derived query; its body
    groundings are marked premises, kept permanently observable, and
    never used as query targets.  All other training triples are
    background queries.  Derived triples cover 23% of training triples.
  * Each 64-query batch mixes 50/50 derived and background queries
    (DERIVED_FRAC_IN_BATCH), and model selection uses the real
    validation splits exactly as in ULTRA-Real (standard 80/10/10 as
    inherited from benchmarking).

The rule set is cached as JSON (``load_rules_json`` / ``save_rules_json``)
so mining never has to be re-run; the frozen 1,240-rule JSON files shipped
with this release are distributed that way (``data/rules/<Graph>.rules.json``).

Everything except the corpus distribution p_C and the task distribution
p_task is shared with the other three arms (Fig. 2d): the released ULTRA
reasoner, loss, optimizer, negative sampling, 200K-step budget and
zero-shot inference protocol are unchanged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from .defaults import (
    ANYBURL_MAX_RULE_LEN,
    ANYBURL_MIN_CONF,
    ANYBURL_MIN_SUPPORT,
    ANYBURL_TIMEOUT_S,
    DERIVED_FRAC_IN_BATCH,
)

# ---------------------------------------------------------------------------
# KG file loaders
#
# trace/evaluate.py owns the public loaders (load_id_map / load_triples).
# It is imported here when available; identical local fallbacks keep this
# module importable on its own.  Format (ULTRA convention):
#   entity2id.txt / relation2id.txt : "<name>\t<id>" per line
#   train.txt / val.txt / test.txt  : "<h>\t<r>\t<t>" (names) per line
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on sibling contributor's file landing
    from .evaluate import load_id_map, load_triples  # type: ignore
except Exception:  # module missing, half-written, or helpers renamed

    def load_id_map(path: str) -> Dict[str, int]:
        """Read a tab-separated '<name>\\t<id>' file into a dict."""
        id_map: Dict[str, int] = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    id_map[parts[0]] = int(parts[1])
        return id_map

    def load_triples(path: str,
                     e2id: Dict[str, int],
                     r2id: Dict[str, int]) -> np.ndarray:
        """Read a tab-separated '<h>\\t<r>\\t<t>' name file into int ids."""
        rows: List[List[int]] = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    h, r, t = parts
                    if h in e2id and r in r2id and t in e2id:
                        rows.append([e2id[h], r2id[r], e2id[t]])
        if rows:
            return np.array(rows, dtype=np.int64)
        return np.empty((0, 3), dtype=np.int64)


Triple = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Rule representation (AnyBURL syntax: r1(x,y) ^ r2(y,z) -> r3(x,z))
# ---------------------------------------------------------------------------

# One atom "name(var_a,var_b)"; variable names are identifier-like strings.
_ATOM_RE = re.compile(
    r"([^\s\^\(\)>]+)\(\s*([^\s,()\[\]]+)\s*,\s*([^\s,()\[\]]+)\s*\)"
)
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")


class Atom:
    """A rule atom ``rel(x, y)``.

    ``rel`` is a relation *name* (str) as parsed from AnyBURL output; it is
    mapped to an integer relation id through the KG's ``relation2id`` map at
    grounding time (see :meth:`Rule.groundings`).  Int ids are also accepted
    directly.  ``x``/``y`` are variable-name strings such as ``"x"``, ``"y"``.
    """

    __slots__ = ("rel", "x", "y")

    def __init__(self, rel: Union[int, str], x: str, y: str):
        self.rel = rel
        self.x = str(x)
        self.y = str(y)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Atom)
                and self.rel == other.rel
                and self.x == other.x and self.y == other.y)

    def __hash__(self) -> int:
        return hash((self.rel, self.x, self.y))

    def __repr__(self) -> str:
        return f"{self.rel}({self.x},{self.y})"


class Rule:
    """A definite Horn rule ``body -> head`` with AnyBURL statistics.

    ``body`` is a tuple of 1-3 :class:`Atom` (AnyBURL is run with rule
    length <= 3; a length-3 rule has 3 body atoms and one head atom).
    ``confidence`` and ``support`` are the mining statistics; either may
    be ``None`` when the source line carried no numbers (hand-written
    rule files), in which case :func:`mine_rules` treats the rule as
    passing the corresponding threshold.
    """

    __slots__ = ("body", "head", "confidence", "support")

    def __init__(self, body: Tuple[Atom, ...], head: Atom,
                 confidence: Optional[float] = None,
                 support: Optional[int] = None):
        body = tuple(body)
        if not 1 <= len(body) <= 3:
            raise ValueError(f"rule body must have 1-3 atoms, got {len(body)}")
        self.body = body
        self.head = head
        self.confidence = None if confidence is None else float(confidence)
        self.support = None if support is None else int(support)

    # -- equality / display ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Rule)
                and self.body == other.body and self.head == other.head
                and self.confidence == other.confidence
                and self.support == other.support)

    def __hash__(self) -> int:
        return hash((self.body, self.head))

    def __repr__(self) -> str:
        body = " ^ ".join(repr(a) for a in self.body)
        return (f"Rule({body} -> {self.head!r}, "
                f"conf={self.confidence}, support={self.support})")

    def __str__(self) -> str:
        body = " ^ ".join(f"{a.rel}({a.x},{a.y})" for a in self.body)
        return f"{body} -> {self.head.rel}({self.head.x},{self.head.y})"

    # -- parsing / serialization -------------------------------------------

    @classmethod
    def from_anyburl_line(cls, line: str) -> "Rule":
        """Parse one AnyBURL rules-file line.

        Accepted shapes (best-effort, AnyBURL's ``-writeRulesTo`` output):

            "12 0.87 r1(x,y) ^ r2(y,z) -> r3(x,z)"
            "0.87 r1(x,y) -> r2(x,y)"          (confidence only)
            "r1(x,y) ^ r2(y,z) -> r3(x,z)"     (hand-written; stats=None)
            "# comment"                        (raises ValueError)

        Leading numeric tokens are interpreted positionally: an integer
        outside [0, 1] is the support, a float in [0, 1] is the
        confidence (``support=..``/``confidence=..``/``conf=..`` tokens
        are honoured when present).
        """
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            raise ValueError(f"not a rule line: {line!r}")
        if "->" not in stripped:
            raise ValueError(f"missing '->' in rule line: {line!r}")
        left, right = stripped.rsplit("->", 1)

        head_match = _ATOM_RE.search(right)
        if head_match is None:
            raise ValueError(f"no head atom in rule line: {line!r}")
        head = Atom(head_match.group(1), head_match.group(2),
                    head_match.group(3))

        body_matches = list(_ATOM_RE.finditer(left))
        if not body_matches:
            raise ValueError(f"no body atoms in rule line: {line!r}")
        body = tuple(Atom(m.group(1), m.group(2), m.group(3))
                     for m in body_matches)

        prefix = left[:body_matches[0].start()]
        confidence, support = _parse_stats(prefix)
        return cls(body, head, confidence, support)

    def to_dict(self) -> dict:
        return {
            "body": [[a.rel, a.x, a.y] for a in self.body],
            "head": [self.head.rel, self.head.x, self.head.y],
            "confidence": self.confidence,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        try:
            body = tuple(Atom(r, x, y) for r, x, y in d["body"])
            hr, hx, hy = d["head"]
            return cls(body, Atom(hr, hx, hy),
                       d.get("confidence"), d.get("support"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed rule record: {d!r}") from exc

    # -- grounding ----------------------------------------------------------

    def resolve(self, relation2id: Optional[Dict[str, int]] = None
                ) -> Optional[Tuple[List[Tuple[int, str, str]], int]]:
        """Map relation names through ``relation2id`` at grounding time.

        Returns ``((rel, x, y), ..., head_rel)`` with integer relation ids,
        or ``None`` when any relation name is unknown (the rule then grounds
        to nothing).  Rules whose atoms already carry int ids resolve
        without a map.
        """
        body: List[Tuple[int, str, str]] = []
        for a in self.body:
            rel = a.rel
            if isinstance(rel, str):
                if relation2id is None or rel not in relation2id:
                    return None
                rel = relation2id[rel]
            body.append((int(rel), a.x, a.y))
        head_rel = self.head.rel
        if isinstance(head_rel, str):
            if relation2id is None or head_rel not in relation2id:
                return None
            head_rel = relation2id[head_rel]
        return body, int(head_rel)

    def bind(self, relation2id: Dict[str, int]) -> Optional["Rule"]:
        """Return a copy with all relation names replaced by int ids
        (``None`` if any name is unknown)."""
        resolved = self.resolve(relation2id)
        if resolved is None:
            return None
        body_ids, head_id = resolved
        body = tuple(Atom(r, x, y) for r, x, y in body_ids)
        return Rule(body, Atom(head_id, self.head.x, self.head.y),
                    self.confidence, self.support)

    def _bindings(self, triples: np.ndarray,
                  relation2id: Optional[Dict[str, int]] = None
                  ) -> Tuple[List[Tuple[int, str, str]], int,
                             List[Dict[str, int]]]:
        """Hash-join the body against ``triples``; return (body_ids,
        head_rel, complete bindings) -- empty binding list if unresolvable."""
        resolved = self.resolve(relation2id)
        if resolved is None or len(triples) == 0:
            return [], -1, []
        body, head_rel = resolved
        return body, head_rel, _join_body(body, triples)

    def groundings(self, triples: np.ndarray,
                   relation2id: Optional[Dict[str, int]] = None) -> List[Triple]:
        """Head groundings ``(h, head_rel, t)`` implied by this rule on the
        given [N, 3] triple array, via hash joins over shared variables
        (1-3 body atoms).  Relation names are mapped through the KG's
        ``relation2id`` map now, at grounding time."""
        _, head_rel, bindings = self._bindings(triples, relation2id)
        hx, hy = self.head.x, self.head.y
        seen: Set[Triple] = set()
        for b in bindings:
            if hx in b and hy in b:
                seen.add((b[hx], head_rel, b[hy]))
        return sorted(seen)

    def derive(self, triples: np.ndarray,
               relation2id: Optional[Dict[str, int]] = None
               ) -> Tuple[Set[Triple], Set[Triple]]:
        """Return (head groundings, body-grounding premises) on ``triples``."""
        body, head_rel, bindings = self._bindings(triples, relation2id)
        heads: Set[Triple] = set()
        premises: Set[Triple] = set()
        hx, hy = self.head.x, self.head.y
        for b in bindings:
            if hx in b and hy in b:
                heads.add((b[hx], head_rel, b[hy]))
            for rel, x, y in body:
                if x in b and y in b:
                    premises.add((b[x], rel, b[y]))
        return heads, premises


def _parse_stats(prefix: str) -> Tuple[Optional[float], Optional[int]]:
    """Pull (confidence, support) out of the numeric prefix of a rule line."""
    confidence: Optional[float] = None
    support: Optional[int] = None
    plain: List[float] = []
    for tok in prefix.replace(",", " ").split():
        key, sep, val = tok.partition("=")
        if sep and key.lower() in ("conf", "confidence"):
            try:
                confidence = float(val)
            except ValueError:
                pass
        elif sep and key.lower() == "support":
            try:
                support = int(float(val))
            except ValueError:
                pass
        elif _NUM_RE.match(tok):
            plain.append(float(tok))
    if confidence is None:
        for v in plain:
            if 0.0 <= v <= 1.0:
                confidence = v
                break
    if support is None:
        for v in plain:
            if v > 1.0:
                support = int(round(v))
                break
        if support is None and confidence is not None:
            # A single number in [0, 1] that was consumed as confidence can
            # also legitimately be a support of 0/1; leave support unknown.
            pass
    return confidence, support


# ---------------------------------------------------------------------------
# Grounding: hash joins over the triple array
# ---------------------------------------------------------------------------

def _rows_by_rel(triples: np.ndarray) -> Dict[int, np.ndarray]:
    """Split an [N, 3] triple array into per-relation [n, 2] (head, tail) views."""
    if len(triples) == 0:
        return {}
    rels = triples[:, 1]
    order = np.argsort(rels, kind="stable")
    s_rel = rels[order]
    s_head = triples[order, 0]
    s_tail = triples[order, 2]
    u, starts = np.unique(s_rel, return_index=True)
    ends = np.append(starts[1:], len(s_rel))
    out: Dict[int, np.ndarray] = {}
    for r, s, e in zip(u.tolist(), starts.tolist(), ends.tolist()):
        out[r] = np.stack([s_head[s:e], s_tail[s:e]], axis=1)
    return out


def _join_body(body: Sequence[Tuple[int, str, str]],
               triples: np.ndarray) -> List[Dict[str, int]]:
    """Hash-join an integer-id rule body against ``triples``; return one
    variable binding dict per distinct complete body grounding.

    Joins are dict-index based over shared variables and handle 1-3 body
    atoms (AnyBURL is run with rule length <= 3).
    """
    if not body or len(triples) == 0:
        return []
    rows_by_rel = _rows_by_rel(triples)

    # Seed with the first body atom.
    r0, x0, y0 = body[0]
    rows = rows_by_rel.get(r0)
    if rows is None or len(rows) == 0:
        return []
    if x0 == y0:
        rows = rows[rows[:, 0] == rows[:, 1]]
        if len(rows) == 0:
            return []
    bindings = [{x0: int(h), y0: int(t)} for h, t in rows]

    for r, x, y in body[1:]:
        rows = rows_by_rel.get(r)
        if rows is None or len(rows) == 0:
            return []
        x_bound = x in bindings[0]
        y_bound = y in bindings[0]
        new: List[Dict[str, int]] = []
        if x_bound and y_bound:
            rowset = {(int(h), int(t)) for h, t in rows}
            for b in bindings:
                if (b[x], b[y]) in rowset:
                    new.append(dict(b))
        elif x_bound:
            idx: Dict[int, List[int]] = {}
            for h, t in rows:
                idx.setdefault(int(h), []).append(int(t))
            for b in bindings:
                for t in idx.get(b[x], ()):
                    nb = dict(b)
                    nb[y] = t
                    new.append(nb)
        elif y_bound:
            idx = {}
            for h, t in rows:
                idx.setdefault(int(t), []).append(int(h))
            for b in bindings:
                for h in idx.get(b[y], ()):
                    nb = dict(b)
                    nb[x] = h
                    new.append(nb)
        else:
            # Disconnected body atom (AnyBURL rules are connected, but be
            # correct for hand-written rules): plain cross join.
            for b in bindings:
                for h, t in rows:
                    nb = dict(b)
                    nb[x] = int(h)
                    nb[y] = int(t)
                    new.append(nb)
        bindings = new
        if not bindings:
            return []

    # Deduplicate distinct bindings (several join paths may coincide).
    seen: Set[Tuple[Tuple[str, int], ...]] = set()
    out: List[Dict[str, int]] = []
    for b in bindings:
        key = tuple(sorted(b.items()))
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Rule I/O and mining
# ---------------------------------------------------------------------------

def save_rules_json(rules: Sequence[Rule], path: str) -> None:
    """Cache a rule list as JSON (the format of the frozen 1,240-rule set)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {"format": "trace-rules/1",
               "n_rules": len(rules),
               "rules": [r.to_dict() for r in rules]}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def load_rules_json(path: str) -> List[Rule]:
    """Load a rule list cached by :func:`save_rules_json`."""
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        entries = payload.get("rules")
        if entries is None:
            raise ValueError(f"{path}: JSON object has no 'rules' key")
    elif isinstance(payload, list):
        entries = payload          # tolerate a bare list of rule records
    else:
        raise ValueError(f"{path}: unexpected rules JSON payload")
    return [Rule.from_dict(d) for d in entries]


def rules_json_path(rules_dir: str, graph_name: str) -> str:
    """Canonical cache location for one graph's rule set."""
    return os.path.join(rules_dir, f"{graph_name}.rules.json")


_INSTALL_MSG = """\
AnyBURL could not be run for {kg_dir!r}.

AnyBURL ships as a Java jar.  To enable on-the-fly mining:
  1. Download the AnyBURL release jar (anyburl.jar) from the official
     distribution at http://www.anyburl.ai/ (a maintained mirror with
     jar releases is https://github.com/mgalkin/AnyBURL/releases);
     a JVM (OpenJDK 11+) must be on PATH.
  2. Point to it via --jar /JAR/PATH/anyburl.jar, the ANYBURL_JAR
     environment variable, or drop anyburl.jar next to the KG files.

Alternatively, skip mining entirely: this release ships the paper's
frozen rule set (1,240 rules, Sec. 3.2 / Appendix I) as cached JSON
under data/rules/<Graph>.rules.json.  load_rules_json() / the corpus
classes load those caches directly and never invoke Java."""


def mine_rules(kg_dir: str,
               out_json: str,
               jar_path: Optional[str] = None,
               java: str = "java",
               timeout_s: int = ANYBURL_TIMEOUT_S,
               max_len: int = ANYBURL_MAX_RULE_LEN,
               min_conf: float = ANYBURL_MIN_CONF,
               min_support: int = ANYBURL_MIN_SUPPORT,
               force: bool = False) -> List[Rule]:
    """Run AnyBURL on ``{kg_dir}/train.txt`` and cache the retained rules.

    Sec. 3.2 / Appendix I: 100 s per graph, rule length <= 3, retain
    confidence >= 0.8 and support >= 10.  Cached JSON at ``out_json`` is
    returned as-is unless ``force=True`` re-runs the miner (the frozen
    paper rule set is distributed through this cache).

    Raises RuntimeError with installation instructions when the jar or a
    JVM is unavailable.
    """
    if os.path.exists(out_json) and not force:
        return load_rules_json(out_json)

    train_txt = os.path.join(kg_dir, "train.txt")
    if not os.path.exists(train_txt):
        raise FileNotFoundError(
            f"{train_txt} not found: AnyBURL mines from the training split; "
            f"expected ULTRA-format files under {kg_dir!r}")

    jar = jar_path or os.environ.get("ANYBURL_JAR")
    if not jar:
        for cand in (os.path.join(kg_dir, "anyburl.jar"),
                     "anyburl.jar"):
            if os.path.exists(cand):
                jar = cand
                break
    if not jar or not os.path.exists(jar):
        raise RuntimeError(_INSTALL_MSG.format(kg_dir=kg_dir)
                           + f"\n\n(jar not found; tried --jar/env ANYBURL_JAR/"
                             f"{os.path.join(kg_dir, 'anyburl.jar')})")

    rules_txt = out_json + ".rules.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    cmd = [
        java, "-jar", jar,
        "-KBdir", kg_dir,
        "-trainSplit", "train.txt",
        "-testSplit", "test.txt",
        "-learnRulesFrom", "train.txt",
        "-maxRuleLength", str(max_len),
        "-timeout", str(timeout_s),
        "-writeRulesTo", rules_txt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s + 120)
        if proc.returncode != 0 and not os.path.exists(rules_txt):
            raise RuntimeError(
                _INSTALL_MSG.format(kg_dir=kg_dir)
                + f"\n\nAnyBURL exited {proc.returncode}; stderr tail:\n"
                + (proc.stderr or "")[-2000:])
    except FileNotFoundError as exc:               # `java` not on PATH
        raise RuntimeError(_INSTALL_MSG.format(kg_dir=kg_dir)
                           + f"\n\n(launcher {java!r} not found: {exc})")
    except subprocess.TimeoutExpired:
        # AnyBURL is an anytime learner; keep whatever it wrote so far.
        if not os.path.exists(rules_txt):
            raise RuntimeError(
                _INSTALL_MSG.format(kg_dir=kg_dir)
                + f"\n\n(AnyBURL exceeded {timeout_s + 120} s without "
                  f"writing rules)")

    rules: List[Rule] = []
    unparsed = 0
    with open(rules_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rules.append(Rule.from_anyburl_line(line))
            except ValueError:
                unparsed += 1
    rules = [
        r for r in rules
        if (r.confidence is None or r.confidence >= min_conf)
        and (r.support is None or r.support >= min_support)
    ]
    save_rules_json(rules, out_json)
    if unparsed:
        print(f"[mine_rules] {kg_dir}: skipped {unparsed} unparsable lines")
    return rules


# ---------------------------------------------------------------------------
# Premise-faithful task construction (Sec. 3.2)
# ---------------------------------------------------------------------------

def construct_premise_faithful_tasks(train_triples: np.ndarray,
                                     rules: Sequence[Rule],
                                     rng: Optional[np.random.Generator] = None,
                                     relation2id: Optional[Dict[str, int]] = None,
                                     ) -> dict:
    """Assign observable/query roles by derivational provenance (Sec. 3.2).

    Each training triple predicted by a retained rule and present in the
    training split becomes a **derived query**; every triple used as a
    body grounding along the way is marked a **premise** -- permanently
    observable and never a query target (premise takes precedence over
    derived).  All remaining training triples are **background queries**.
    Triples covered by several rules are deduplicated.

    ``train_triples`` is an [N, 3] int-id array.  ``rules`` may carry
    relation *names* (as parsed by AnyBURL) or int ids; name-based rules
    are mapped through the KG's ``relation2id`` map at grounding time --
    pass it here (``PremiseFaithfulRealCorpus`` does), or pre-bind rules
    with :meth:`Rule.bind`.

    Returns a dict with:
        'derived_queries'    np.int64 [M, 3] unique derived targets
        'premises'           set of canonical (h, r, t) tuples (observable,
                             never targets)
        'background_queries' np.int64 [K, 3] unique remaining targets
        'n_derived'          int
        'coverage'           n_derived / |train unique triples|
    """
    if rng is None:
        rng = np.random.default_rng()
    train = np.asarray(train_triples, dtype=np.int64).reshape(-1, 3)
    train_set: Set[Triple] = {
        (int(h), int(r), int(t)) for h, r, t in train
    }

    head_hits: Set[Triple] = set()
    premises: Set[Triple] = set()
    for rule in rules:
        heads, prem = rule.derive(train, relation2id)
        head_hits |= heads
        premises |= prem

    # Premises never enter either query pool; derived = rule conclusions
    # that are present in the training split, minus premises.
    derived = (head_hits & train_set) - premises
    background = train_set - derived - premises

    # --- Sec. 3.2 invariants -------------------------------------------
    assert not (premises & (derived | background)), \
        "premise triples must never be query targets"
    assert premises <= train_set, \
        "premises must be observable training-split triples"

    def _arr(rows: Set[Triple]) -> np.ndarray:
        if not rows:
            return np.empty((0, 3), dtype=np.int64)
        arr = np.array(sorted(rows), dtype=np.int64)
        return arr[rng.permutation(len(arr))]

    n_train = max(len(train_set), 1)
    derived_arr = _arr(derived)
    return {
        "derived_queries": derived_arr,
        "premises": premises,
        "background_queries": _arr(background),
        "n_derived": int(len(derived_arr)),
        "coverage": len(derived_arr) / n_train,
    }


# ---------------------------------------------------------------------------
# KG loading
# ---------------------------------------------------------------------------

class RealKG:
    """One real KG in ULTRA layout (train/val/test + id maps, tab-separated)."""

    def __init__(self, kg_dir: str):
        self.kg_dir = kg_dir
        if not os.path.isdir(kg_dir):
            raise FileNotFoundError(f"No such KG directory: {kg_dir!r}")
        self.entity2id = load_id_map(os.path.join(kg_dir, "entity2id.txt"))
        self.relation2id = load_id_map(os.path.join(kg_dir, "relation2id.txt"))
        self.n_entities = len(self.entity2id)
        self.n_relations = len(self.relation2id)
        if self.n_entities == 0 or self.n_relations == 0:
            raise ValueError(f"{kg_dir}: empty entity2id/relation2id map")
        self.train = load_triples(
            os.path.join(kg_dir, "train.txt"), self.entity2id, self.relation2id)
        self.val = load_triples(
            os.path.join(kg_dir, "val.txt"), self.entity2id, self.relation2id)
        self.test = load_triples(
            os.path.join(kg_dir, "test.txt"), self.entity2id, self.relation2id)

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.kg_dir))

    def __repr__(self) -> str:
        return (f"RealKG({self.name!r}, |V|={self.n_entities}, "
                f"|R|={self.n_relations}, train={len(self.train)}, "
                f"val={len(self.val)}, test={len(self.test)})")


# ---------------------------------------------------------------------------
# Shared graph helpers
# ---------------------------------------------------------------------------

def _add_inverses(triples: np.ndarray, n_rel: int) -> np.ndarray:
    """Canonical + inverse closure (t, r + n_rel, h) for each (h, r, t)."""
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    if len(triples) == 0:
        return triples
    inv = np.stack([triples[:, 2], triples[:, 1] + n_rel, triples[:, 0]],
                   axis=1).astype(np.int64)
    return np.concatenate([triples, inv], axis=0)


def _lift_fallback(triples: np.ndarray, n_relations: int) -> Dict:
    """Local stand-in mirroring trace/generator.py's ``lift`` contract
    (nodes 0..n_relations-1 with one type-0 self-loop each; four binary
    entity-co-occurrence interaction types between relation nodes).
    Used only when trace.generator is not importable."""
    src: List[int] = []
    dst: List[int] = []
    typ: List[int] = []
    if len(triples):
        h = triples[:, 0].astype(np.int64)
        r = triples[:, 1].astype(np.int64)
        t = triples[:, 2].astype(np.int64)
        head_rels: Dict[int, Set[int]] = {}
        tail_rels: Dict[int, Set[int]] = {}
        for e, rid in zip(h.tolist(), r.tolist()):
            head_rels.setdefault(e, set()).add(rid)
        for e, rid in zip(t.tolist(), r.tolist()):
            tail_rels.setdefault(e, set()).add(rid)
        for e in head_rels.keys() | tail_rels.keys():
            hs = sorted(head_rels.get(e, ()))
            ts = sorted(tail_rels.get(e, ()))
            if hs and ts:
                for i in ts:
                    for j in hs:          # 0: tail_i == head_j
                        src.append(i); dst.append(j); typ.append(0)
                for i in hs:
                    for j in ts:          # 3: head_i == tail_j
                        src.append(i); dst.append(j); typ.append(3)
            for i in hs:
                for j in hs:              # 1: head_i == head_j
                    src.append(i); dst.append(j); typ.append(1)
            for i in ts:
                for j in ts:              # 2: tail_i == tail_j
                    src.append(i); dst.append(j); typ.append(2)
    if src:
        rows = np.unique(np.stack([typ, src, dst], axis=1), axis=0)
        ei = rows[:, 1:3].T.copy()
        et = rows[:, 0].copy()
    else:
        ei = np.zeros((2, 0), dtype=np.int64)
        et = np.zeros(0, dtype=np.int64)
    loop = np.tile(np.arange(n_relations, dtype=np.int64), (2, 1))
    ei = np.concatenate([ei, loop], axis=1)
    et = np.concatenate([et, np.zeros(n_relations, dtype=np.int64)])
    return {"edge_index": ei, "edge_type": et,
            "num_nodes": int(n_relations)}


def _relation_graph(triples: np.ndarray, n_relations: int) -> Dict:
    """Build the lifted relation graph, preferring trace.generator.lift.

    The released real-graph data loader augments the observable graph
    with inverse edges (t, r + R, h), so the relation space is the
    doubled one: nodes 0..2R-1, one self-loop per relation node.  Derive
    the node count from the edges actually passed in so G_r stays
    consistent whether it was built over canonical or inverse-augmented
    triples.
    """
    lift = None
    try:
        from trace.generator import lift  # lazy; canonical ULTRA Lift
    except Exception:
        lift = None
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    if triples.size:
        n_relations = int(max(n_relations, triples[:, 1].max() + 1))
    if lift is not None:
        return lift(triples, n_relations)
    return _lift_fallback(triples, n_relations)


def _to_tensors(queries: np.ndarray, obs: np.ndarray,
                gr: Dict, n_entities: int, n_relations: int,
                filter_triples: np.ndarray) -> Dict:
    """Pack one step into the shared batch dict (same shape as the
    synthetic loader in trace/corpus.py)."""
    queries = np.asarray(queries, dtype=np.int64).reshape(-1, 3)
    obs = np.asarray(obs, dtype=np.int64)
    filter_triples = np.asarray(filter_triples, dtype=np.int64).reshape(-1, 3)
    return dict(
        edge_index=torch.from_numpy(
            np.stack([obs[:, 0], obs[:, 2]], axis=0).astype(np.int64)),
        edge_type=torch.from_numpy(obs[:, 1].astype(np.int64)),
        query_h=torch.from_numpy(queries[:, 0].astype(np.int64)),
        query_r=torch.from_numpy(queries[:, 1].astype(np.int64)),
        query_t=torch.from_numpy(queries[:, 2].astype(np.int64)),
        num_entities=int(n_entities),
        num_relations=int(n_relations),
        gr_edge_index=torch.from_numpy(gr["edge_index"].astype(np.int64)),
        gr_edge_type=torch.from_numpy(gr["edge_type"].astype(np.int64)),
        gr_num_nodes=int(gr["num_nodes"]),
        filter_triples=torch.from_numpy(filter_triples),
    )


def _known_true(kg: RealKG) -> np.ndarray:
    """All known-true (canonical) triples for filtered scoring."""
    parts = [kg.train]
    if len(kg.val):
        parts.append(kg.val)
    if len(kg.test):
        parts.append(kg.test)
    return np.unique(np.concatenate(parts, axis=0), axis=0)


# ---------------------------------------------------------------------------
# ULTRA-Real arm: standard task construction over the three real KGs
# ---------------------------------------------------------------------------

class StandardRealCorpus:
    """ULTRA-Real arm loader (Fig. 2d, Sec. 4.1): the three real KGs with
    standard ULTRA task construction.

    One graph is sampled uniformly per step; the observable graph is the
    train split plus inverses, and the 64 queries are drawn uniformly
    (with replacement) from that graph's train split -- the standard
    80/10/10 protocol as inherited from benchmarking, where every
    training triple is a query.  Model selection uses the real validation
    splits (:meth:`val_batches`), identical to ULTRA-Real.  The returned
    batch dict has exactly the shape of the synthetic loader's.
    """

    def __init__(self, kg_dirs: Sequence[str], batch_size: int = 64,
                 seed: Optional[int] = None):
        self.kgs = [RealKG(d) for d in kg_dirs]
        if not self.kgs:
            raise ValueError("StandardRealCorpus needs at least one KG dir")
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self._cache: List[dict] = []
        for kg in self.kgs:
            obs = _add_inverses(kg.train, kg.n_relations)
            self._cache.append({
                "obs": obs,
                "gr": _relation_graph(obs, kg.n_relations),
                "filter": _known_true(kg),
            })

    def __len__(self) -> int:
        return len(self.kgs)

    def sample_batch(self) -> Dict:
        """One training step: uniform graph, uniform queries from its train split."""
        i = int(self.rng.integers(0, len(self.kgs)))
        return self._make_batch(i)

    def _make_batch(self, kg_idx: int) -> Dict:
        kg = self.kgs[kg_idx]
        c = self._cache[kg_idx]
        if len(kg.train) == 0:
            raise ValueError(f"{kg.name}: empty train split")
        idx = self.rng.integers(0, len(kg.train), size=self.batch_size)
        return _to_tensors(kg.train[idx], c["obs"], c["gr"],
                           kg.n_entities, kg.n_relations, c["filter"])

    def val_batches(self) -> List[Dict]:
        """Real validation queries per graph (model selection, as in ULTRA-Real)."""
        out = []
        for i, kg in enumerate(self.kgs):
            c = self._cache[i]
            out.append(_to_tensors(kg.val, c["obs"], c["gr"],
                                   kg.n_entities, kg.n_relations, c["filter"]))
        return out


# ---------------------------------------------------------------------------
# TRACE-Real arm: premise-faithful task construction over the three real KGs
# ---------------------------------------------------------------------------

class PremiseFaithfulRealCorpus:
    """TRACE-Real arm loader (Sec. 3.2, Appendix I, Fig. 2d).

    Rules are loaded from the per-graph JSON caches in ``rules_dir``
    (run ``scripts/mine_rules.py`` once, or use the frozen release set);
    :func:`construct_premise_faithful_tasks` is called once per graph at
    init.  Per step, one graph is sampled uniformly and the 64-query
    batch is 32 derived + 32 background (``derived_frac=0.5``), falling
    back proportionally when a pool is short.  Observable edges are the
    train split union the premises (premises are permanently observable
    and never used as targets); model selection uses the REAL validation
    splits (:meth:`val_batches`), exactly as in ULTRA-Real.
    """

    def __init__(self, kg_dirs: Sequence[str], rules_dir: str,
                 batch_size: int = 64,
                 derived_frac: float = DERIVED_FRAC_IN_BATCH,
                 seed: Optional[int] = None):
        self.kgs = [RealKG(d) for d in kg_dirs]
        if not self.kgs:
            raise ValueError("PremiseFaithfulRealCorpus needs >= 1 KG dir")
        self.rules_dir = rules_dir
        self.batch_size = int(batch_size)
        self.derived_frac = float(derived_frac)
        self.rng = np.random.default_rng(seed)

        self.rules: List[List[Rule]] = []
        self.tasks: List[dict] = []
        self._cache: List[dict] = []
        for kg in self.kgs:
            path = rules_json_path(rules_dir, kg.name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No cached rules for {kg.name!r} at {path!r}. Run "
                    f"`python scripts/mine_rules.py --data_dir <pretraining> "
                    f"--rules_dir {rules_dir}` once, or ship the frozen "
                    f"rule JSON from the paper release.")
            rules = load_rules_json(path)
            tasks = construct_premise_faithful_tasks(
                kg.train, rules, self.rng, relation2id=kg.relation2id)
            # Observable = train split ∪ premises (+ inverses).  Premises
            # are already train-split triples, but take the explicit union
            # so the invariant is structural, not incidental.
            train = np.unique(kg.train, axis=0)
            if tasks["premises"]:
                prem = np.array(sorted(tasks["premises"]), dtype=np.int64)
                base = np.unique(np.concatenate([train, prem], axis=0), axis=0)
            else:
                base = train
            obs = _add_inverses(base, kg.n_relations)
            self.rules.append(rules)
            self.tasks.append(tasks)
            self._cache.append({
                "obs": obs,
                "gr": _relation_graph(obs, kg.n_relations),
                "filter": _known_true(kg),
            })

    def __len__(self) -> int:
        return len(self.kgs)

    def summary(self) -> str:
        lines = []
        for kg, tasks in zip(self.kgs, self.tasks):
            lines.append(
                f"  {kg.name}: |train|={len(kg.train)} "
                f"derived={tasks['n_derived']} "
                f"premises={len(tasks['premises'])} "
                f"coverage={tasks['coverage']:.3f}")
        return "PremiseFaithfulRealCorpus:\n" + "\n".join(lines)

    # ------------------------------------------------------------------

    def sample_batch(self) -> Dict:
        i = int(self.rng.integers(0, len(self.kgs)))
        return self._make_batch(i)

    def _make_batch(self, kg_idx: int) -> Dict:
        """64-query batch: derived_frac derived + rest background, with
        proportional fallback when a pool is short."""
        kg = self.kgs[kg_idx]
        tasks = self.tasks[kg_idx]
        c = self._cache[kg_idx]

        want_d = int(round(self.batch_size * self.derived_frac))
        want_b = self.batch_size - want_d
        have_d = len(tasks["derived_queries"])
        have_b = len(tasks["background_queries"])

        n_d = min(want_d, have_d)
        n_b = min(want_b, have_b)
        short = self.batch_size - n_d - n_b
        if short > 0 and have_b > n_b:               # backfill from bg slack
            extra = min(short, have_b - n_b)
            n_b += extra
            short -= extra
        if short > 0 and have_d > n_d:               # then from derived slack
            extra = min(short, have_d - n_d)
            n_d += extra
            short -= extra
        if short > 0:
            # Both pools are exhausted: top up with replacement from the
            # larger available pool (tiny toy KGs / degenerate rule sets).
            pool = tasks["derived_queries"] if have_d >= have_b \
                else tasks["background_queries"]
            if len(pool):
                n_d += short if have_d >= have_b else 0
                n_b += 0 if have_d >= have_b else short
            else:
                raise ValueError(f"{kg.name}: no queries available for a batch")

        parts = []
        if n_d > 0:
            parts.append(self._sample_pool(tasks["derived_queries"], n_d))
        if n_b > 0:
            parts.append(self._sample_pool(tasks["background_queries"], n_b))
        queries = np.concatenate(parts, axis=0)
        return _to_tensors(queries, c["obs"], c["gr"],
                           kg.n_entities, kg.n_relations, c["filter"])

    def _sample_pool(self, pool: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or len(pool) == 0:
            return np.empty((0, 3), dtype=np.int64)
        if len(pool) >= k:
            idx = self.rng.choice(len(pool), size=k, replace=False)
        else:
            idx = self.rng.integers(0, len(pool), size=k)
        return pool[idx]

    def val_batches(self) -> List[Dict]:
        """REAL validation queries per graph (standard protocol; model
        selection as in ULTRA-Real).  Val targets are never observable."""
        out = []
        for i, kg in enumerate(self.kgs):
            c = self._cache[i]
            out.append(_to_tensors(kg.val, c["obs"], c["gr"],
                                   kg.n_entities, kg.n_relations, c["filter"]))
        return out
