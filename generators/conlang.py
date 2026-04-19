"""Conlang-Regular generator.

Agglutinative constructed language with:
  - 2-syllable CVCV roots assigned 1:1 to English concepts
  - strict regular morphology with fixed suffix order
  - nom/acc case marking enables free word order (toggleable)
  - same sentence templates as MicroEnglish for fair comparison
  - grounded metadata identical in structure to MicroEnglish metadata

Suffix inventory (short and phonotactically distinct):
  NOUN:  ROOT [-NUM] [-CASE] [-DEF]
    NUM:   sg = ""       pl = "pa"
    CASE:  nom = ""      acc = "ki"    dat = "li"    loc = "ne"
    DEF:   indef = ""    def = "na"
  VERB:  ROOT [-TENSE] [-ASPECT] [-AGR]
    TENSE:   prs = ""  pst = "du"  fut = "mo"
    ASPECT:  pfv = ""  prog = "si"
    AGR:     sg = ""   pl = "ta"

Closed-class particles:
  ke   = subordinator (rel/comp clauses)
  vo   = copula ("is")
  mi   = negation
  sa   = conjunction "and"
  ra   = because       zu = while       to = after       fi = before
  pe   = "to" (allative for motion goals)

Sentence-final period token kept as "." for tokenizer parity with MicroEnglish.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional
from .world import World, Entity
from .microenglish import (TRANSITIVE_VERBS, INTRANSITIVE_VERBS, MOTION_VERBS,
                           MENTAL_VERBS, COLORS, SIZES, LOCATIONS)
from .world import PERSON_NAMES, ANIMAL_KINDS, OBJECT_KINDS


CONSONANTS = list("ptkbdgmnslrvfh")
VOWELS = list("aeiou")


def _gen_roots(n: int, rng: random.Random) -> list[str]:
    """Generate n unique CVCV roots avoiding suffix collisions."""
    forbidden_suffixes = {"pa", "ki", "li", "ne", "na", "du", "si", "mo", "ta",
                           "ke", "vo", "mi", "sa", "ra", "zu", "to", "fi", "pe"}
    seen = set()
    out = []
    attempts = 0
    while len(out) < n and attempts < 100000:
        attempts += 1
        r = (rng.choice(CONSONANTS) + rng.choice(VOWELS)
             + rng.choice(CONSONANTS) + rng.choice(VOWELS))
        if r in seen:
            continue
        if r[2:] in forbidden_suffixes:
            continue
        if r in forbidden_suffixes:
            continue
        seen.add(r)
        out.append(r)
    return out


# Collect all English concepts we need to assign roots to
def _concept_list() -> list[str]:
    items = []
    items += PERSON_NAMES
    items += ANIMAL_KINDS
    items += OBJECT_KINDS
    items += COLORS
    items += SIZES
    items += LOCATIONS
    items += TRANSITIVE_VERBS
    items += INTRANSITIVE_VERBS
    items += MOTION_VERBS
    items += MENTAL_VERBS
    return items


@dataclass
class CLConfig:
    vocab_nouns: int = 0
    vocab_verbs: int = 0
    max_clause_depth: int = 2
    allow_relative: bool = True
    allow_complement: bool = True
    allow_coordination: bool = True
    grounded_fraction: float = 0.5
    seed: int = 0
    word_order: str = "SOV"   # SOV | SVO | free
    case_marking: bool = True


def noun_form(root: str, plural: bool, case: str, definite: bool) -> str:
    s = root
    if plural:
        s += "pa"
    if case == "acc":
        s += "ki"
    elif case == "dat":
        s += "li"
    elif case == "loc":
        s += "ne"
    if definite:
        s += "na"
    return s


def verb_form(root: str, tense: str, prog: bool, plural: bool) -> str:
    s = root
    if tense == "past":
        s += "du"
    elif tense == "future":
        s += "mo"
    if prog:
        s += "si"
    if plural:
        s += "ta"
    return s


class Conlang:
    def __init__(self, cfg: CLConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        concepts = _concept_list()
        roots = _gen_roots(len(concepts) + 20, self.rng)
        self.dict = {c: roots[i] for i, c in enumerate(concepts)}
        self.world = World(self.rng)
        self.transitive = TRANSITIVE_VERBS[:cfg.vocab_verbs] if cfg.vocab_verbs else TRANSITIVE_VERBS
        self.intransitive = INTRANSITIVE_VERBS[:]
        self.mental = MENTAL_VERBS[:]

    def dictionary(self) -> dict[str, str]:
        return dict(self.dict)

    # --- NP generation ---
    def _np(self, ent: Entity, case: str, depth: int = 0,
            definite: Optional[bool] = None) -> list[str]:
        definite = self.rng.random() < 0.7 if definite is None else definite
        tokens: list[str] = []
        # adjective goes before noun as a separate token (keep simple; no agreement)
        if ent.etype != "person" and self.rng.random() < 0.4:
            if ent.color and self.rng.random() < 0.6:
                tokens.append(self.dict[ent.color])
            elif ent.size:
                tokens.append(self.dict[ent.size])
        if ent.etype == "person":
            # proper nouns don't get DEF (they're inherently definite) but do take case
            root = self.dict[ent.name]
            tokens.append(noun_form(root, False, case, False))
        else:
            root = self.dict[ent.kind]
            tokens.append(noun_form(root, False, case, definite))
        if (self.cfg.allow_relative and depth < self.cfg.max_clause_depth
                and self.rng.random() < 0.15):
            tokens += self._rel_clause(ent, depth + 1)
        return tokens

    def _rel_clause(self, head: Entity, depth: int) -> list[str]:
        """[inner-clause] ke — places modifier clause before particle."""
        verb = self.rng.choice(self.transitive)
        other = self.rng.choice([e for e in self.world.entities if e.id != head.id])
        # In the relative clause head is understood as subject (gap at subject)
        obj_toks = self._np(other, case="acc", depth=depth, definite=True)
        tense = self.rng.choice(["present", "past"])
        v = verb_form(self.dict[verb], tense, False, False)
        # Order inside relative: OBJ V ke  (SOV minus subject gap)
        return obj_toks + [v, "ke"]

    # --- Sentence templates ---
    def _arrange(self, s_toks: list[str], o_toks: list[str], v_tok: str) -> list[str]:
        order = self.cfg.word_order
        if order == "free":
            order = self.rng.choice(["SOV", "SVO", "OSV", "OVS"])
        if order == "SOV":
            return s_toks + o_toks + [v_tok]
        if order == "SVO":
            return s_toks + [v_tok] + o_toks
        if order == "OSV":
            return o_toks + s_toks + [v_tok]
        if order == "OVS":
            return o_toks + [v_tok] + s_toks
        return s_toks + o_toks + [v_tok]

    def _svo(self, tense: str) -> tuple[list[str], dict]:
        subj = self.rng.choice(self.world.agents())
        obj = self.rng.choice([e for e in self.world.entities if e.id != subj.id])
        verb = self.rng.choice(self.transitive)
        s_toks = self._np(subj, case="nom")
        o_toks = self._np(obj, case="acc" if self.cfg.case_marking else "nom", definite=True)
        v_tok = verb_form(self.dict[verb], tense, False, False)
        toks = self._arrange(s_toks, o_toks, v_tok) + ["."]
        return toks, {"template": "SVO", "subject_id": subj.id, "object_id": obj.id,
                     "verb": verb, "tense": tense}

    def _sv(self, tense: str) -> tuple[list[str], dict]:
        subj = self.rng.choice(self.world.agents())
        verb = self.rng.choice(self.intransitive)
        s_toks = self._np(subj, case="nom")
        v_tok = verb_form(self.dict[verb], tense, False, False)
        toks = s_toks + [v_tok, "."]
        return toks, {"template": "SV", "subject_id": subj.id, "verb": verb, "tense": tense}

    def _copular(self) -> tuple[list[str], dict]:
        ent = self.rng.choice(self.world.entities)
        prop_choices = []
        if ent.color:
            prop_choices.append(("color", ent.color))
        if ent.size:
            prop_choices.append(("size", ent.size))
        prop_choices.append(("location", ent.location))
        kind, val = self.rng.choice(prop_choices)
        s_toks = self._np(ent, case="nom", definite=True)
        if kind == "location":
            # "X is in LOC" -> X LOC-loc vo
            loc_tok = noun_form(self.dict[val], False, "loc", True)
            toks = s_toks + [loc_tok, "vo", "."]
        else:
            toks = s_toks + [self.dict[val], "vo", "."]
        return toks, {"template": "COP", "subject_id": ent.id, "property": kind, "value": val}

    def _complement(self, tense: str) -> tuple[list[str], dict]:
        subj = self.rng.choice(self.world.people())
        verb = self.rng.choice(self.mental)
        inner, inner_meta = self._svo(tense="present")
        if inner[-1] == ".":
            inner = inner[:-1]
        s_toks = self._np(subj, case="nom")
        v_tok = verb_form(self.dict[verb], tense, False, False)
        # Structure: SUBJ [inner-clause ke] VERB .
        toks = s_toks + inner + ["ke", v_tok, "."]
        return toks, {"template": "COMP", "subject_id": subj.id, "verb": verb, "inner": inner_meta}

    def _coordinated(self, tense: str) -> tuple[list[str], dict]:
        a, ma = self._svo(tense)
        b, mb = self._svo(tense)
        if a[-1] == ".":
            a = a[:-1]
        if b[-1] == ".":
            b = b[:-1]
        conn_map = {"because": "ra", "while": "zu", "after": "to", "before": "fi"}
        conn = self.rng.choice(list(conn_map.keys()))
        return a + [conn_map[conn]] + b + ["."], {"template": "SUB", "outer": ma, "inner": mb, "conn": conn}

    def generate(self, n_sentences: int, fresh_world_every: int = 20) -> list[tuple[list[str], dict]]:
        out = []
        for i in range(n_sentences):
            if i % fresh_world_every == 0:
                self.world.new()
            grounded = self.rng.random() < self.cfg.grounded_fraction
            tense = self.rng.choice(["present", "past"])
            r = self.rng.random()
            if r < 0.45:
                toks, meta = self._svo(tense)
            elif r < 0.65:
                toks, meta = self._sv(tense)
            elif r < 0.80:
                toks, meta = self._copular()
            elif r < 0.92 and self.cfg.allow_complement:
                toks, meta = self._complement(tense)
            elif self.cfg.allow_coordination:
                toks, meta = self._coordinated(tense)
            else:
                toks, meta = self._svo(tense)
            meta["grounded"] = grounded
            out.append((toks, meta))
        return out
