"""Unified MicroLang generator.

Single grammar producing the full spectrum from English-surface to
agglutinative-conlang, driven by configuration.

Architecture:
  - `AbstractSentence` is produced by the grammar: template + argument roles
    (subject, object, modifiers) filled from world state.
  - A `Realizer` turns the abstract sentence into tokens. Swap realizers to
    change surface form without touching grammar.

Toggles (all orthogonal):
  morphology:    "english" | "agglutinative"
  word_order:    "SVO"     | "SOV"           | "free"
  case_marking:  bool    (only relevant for agglutinative)
  articles:      bool    (only relevant for english)
  relatives, complements, coordination: bool
  max_clause_depth: int
  grounded_fraction: float in [0, 1]
  vocab_nouns, vocab_verbs: subsample counts (0 = all)

By flipping these, the same grammar produces MicroEnglish, Conlang-Regular,
or any hybrid in between.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional

from .world import (World, Entity, PERSON_NAMES, ANIMAL_KINDS, OBJECT_KINDS,
                    COLORS, SIZES, LOCATIONS)
from .microenglish import (TRANSITIVE_VERBS, INTRANSITIVE_VERBS, MOTION_VERBS,
                           MENTAL_VERBS)


# =========================================================================
# Abstract sentence representation
# =========================================================================

@dataclass
class AbsNP:
    """Abstract noun phrase."""
    entity: Entity
    role: str            # "nom" | "acc" | "dat" | "loc"
    definite: bool
    plural: bool = False
    adjective: Optional[str] = None    # "color:red" | "size:big" | None
    modifier: Optional["AbsClause"] = None  # relative clause


@dataclass
class AbsClause:
    """Abstract clause."""
    kind: str             # SVO | SV | COP | COMP | SUB | REL
    tense: str            # present | past
    subject: Optional[AbsNP] = None
    object: Optional[AbsNP] = None
    verb: Optional[str] = None          # lemma (lookup via dictionary in realizer)
    property_kind: Optional[str] = None  # for COP: color | size | location
    property_value: Optional[str] = None
    inner: Optional["AbsClause"] = None
    conn: Optional[str] = None           # for SUB: because/while/after/before


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class LangConfig:
    # realization toggles
    morphology: str = "english"         # "english" | "agglutinative"
    word_order: str = "SVO"             # "SVO" | "SOV" | "free"
    case_marking: bool = False          # agglutinative only
    articles: bool = True               # english only

    # grammar toggles
    allow_relative: bool = True
    allow_complement: bool = True
    allow_coordination: bool = True
    allow_plural_subjects: bool = False   # enables agreement probes
    plural_subject_prob: float = 0.3       # only used when allow_plural_subjects
    max_clause_depth: int = 2

    # corpus knobs
    grounded_fraction: float = 0.5
    vocab_nouns: int = 0
    vocab_verbs: int = 0
    seed: int = 0
    lexicon_seed: int = 42    # fixed by default so probe corpora share vocab


# =========================================================================
# Realizers
# =========================================================================

class EnglishRealizer:
    """Regularized English surface forms."""

    def __init__(self, cfg: LangConfig):
        self.cfg = cfg

    def noun(self, lemma: str, plural: bool) -> str:
        return lemma + ("s" if plural else "")

    def verb(self, lemma: str, tense: str, plural_subj: bool) -> str:
        if tense == "past":
            return lemma + ("d" if lemma.endswith("e") else "ed")
        return lemma if plural_subj else lemma + "s"

    def np(self, np: AbsNP) -> list[str]:
        toks: list[str] = []
        ent = np.entity
        if ent.etype == "person":
            toks.append(ent.name)
        else:
            if self.cfg.articles:
                if np.plural:
                    # plural indefinite: bare; plural definite: "the"
                    if np.definite:
                        toks.append("the")
                else:
                    toks.append("the" if np.definite else "a")
            if np.adjective:
                toks.append(np.adjective.split(":")[1])
            toks.append(self.noun(ent.kind, np.plural))
        if np.modifier:
            toks += self.rel(np, np.modifier)
        return toks

    def rel(self, head_np: AbsNP, clause: AbsClause) -> list[str]:
        marker = "who" if head_np.entity.etype == "person" else "that"
        v = self.verb(clause.verb, clause.tense, head_np.plural)
        obj_toks = self.np(clause.object)
        return [marker, v] + obj_toks

    def arrange(self, s: list[str], o: list[str], v: str) -> list[str]:
        order = self.cfg.word_order
        if order == "SOV":  return s + o + [v]
        if order == "OSV":  return o + s + [v]
        if order == "OVS":  return o + [v] + s
        return s + [v] + o

    def copular(self, np_toks: list[str], prop_kind: str, value: str,
                plural_subj: bool = False) -> list[str]:
        be = "are" if plural_subj else "is"
        if prop_kind == "location":
            return np_toks + [be, "in", "the", value]
        return np_toks + [be, value]

    def connective(self, conn: str) -> str:
        return conn

    def complement_marker(self) -> str:
        return "that"

    def complement_assemble(self, outer_subj: list[str], outer_verb: str,
                            inner_toks: list[str]) -> list[str]:
        return outer_subj + [outer_verb, "that"] + inner_toks


class AgglutinativeRealizer:
    """Agglutinative conlang surface forms."""

    CONSONANTS = list("ptkbdgmnslrvfh")
    VOWELS = list("aeiou")
    FORBIDDEN = {"pa", "ki", "li", "ne", "na", "du", "si", "mo", "ta",
                 "ke", "vo", "mi", "sa", "ra", "zu", "to", "fi", "pe"}

    def __init__(self, cfg: LangConfig, seed: int):
        self.cfg = cfg
        self.rng = random.Random(seed + 7)
        concepts = (PERSON_NAMES + ANIMAL_KINDS + OBJECT_KINDS + COLORS + SIZES
                    + LOCATIONS + TRANSITIVE_VERBS + INTRANSITIVE_VERBS
                    + MOTION_VERBS + MENTAL_VERBS)
        self.dict = self._make_dict(concepts)

    def _make_dict(self, concepts: list[str]) -> dict[str, str]:
        seen, roots = set(), []
        while len(roots) < len(concepts):
            r = (self.rng.choice(self.CONSONANTS) + self.rng.choice(self.VOWELS)
                 + self.rng.choice(self.CONSONANTS) + self.rng.choice(self.VOWELS))
            if r in seen or r[2:] in self.FORBIDDEN or r in self.FORBIDDEN:
                continue
            seen.add(r); roots.append(r)
        return {c: roots[i] for i, c in enumerate(concepts)}

    def dictionary(self) -> dict[str, str]:
        return dict(self.dict)

    def noun(self, lemma: str, plural: bool, case: str, definite: bool) -> str:
        s = self.dict[lemma]
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

    def verb(self, lemma: str, tense: str, plural_subj: bool) -> str:
        s = self.dict[lemma]
        if tense == "past":
            s += "du"
        if plural_subj:
            s += "ta"
        return s

    def np(self, np: AbsNP) -> list[str]:
        toks: list[str] = []
        ent = np.entity
        case = np.role if self.cfg.case_marking else "nom"
        if ent.etype == "person":
            # proper nouns bare-rooted except for case
            root = self.dict[ent.name]
            if case == "acc": root += "ki"
            elif case == "dat": root += "li"
            elif case == "loc": root += "ne"
            toks.append(root)
        else:
            if np.adjective:
                toks.append(self.dict[np.adjective.split(":")[1]])
            toks.append(self.noun(ent.kind, np.plural, case, np.definite))
        if np.modifier:
            toks += self.rel(np, np.modifier)
        return toks

    def rel(self, head_np: AbsNP, clause: AbsClause) -> list[str]:
        obj_toks = self.np(clause.object)
        v = self.verb(clause.verb, clause.tense, head_np.plural)
        return obj_toks + [v, "ke"]

    def arrange(self, s: list[str], o: list[str], v: str) -> list[str]:
        order = self.cfg.word_order
        if order == "free":
            order = self.rng.choice(["SOV", "SVO", "OSV", "OVS"])
        if order == "SOV":   return s + o + [v]
        if order == "SVO":   return s + [v] + o
        if order == "OSV":   return o + s + [v]
        if order == "OVS":   return o + [v] + s
        return s + o + [v]

    def copular(self, np_toks: list[str], prop_kind: str, value: str,
                plural_subj: bool = False) -> list[str]:
        cop = "vo" + ("ta" if plural_subj else "")
        if prop_kind == "location":
            loc_tok = self.noun(value, False, "loc", True)
            return np_toks + [loc_tok, cop]
        return np_toks + [self.dict[value], cop]

    def connective(self, conn: str) -> str:
        return {"because": "ra", "while": "zu", "after": "to", "before": "fi"}[conn]

    def complement_marker(self) -> str:
        return "ke"

    def complement_assemble(self, outer_subj: list[str], outer_verb: str,
                            inner_toks: list[str]) -> list[str]:
        # Agglutinative: SUBJ [inner] ke VERB
        return outer_subj + inner_toks + ["ke", outer_verb]


# =========================================================================
# Grammar (abstract sentence builder)
# =========================================================================

class Grammar:
    def __init__(self, cfg: LangConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.world = World(self.rng)
        self.transitive = (TRANSITIVE_VERBS[:cfg.vocab_verbs]
                           if cfg.vocab_verbs else TRANSITIVE_VERBS)[:]
        self.intransitive = INTRANSITIVE_VERBS[:]
        self.mental = MENTAL_VERBS[:]
        if cfg.morphology == "english":
            self.realizer = EnglishRealizer(cfg)
        elif cfg.morphology == "agglutinative":
            self.realizer = AgglutinativeRealizer(cfg, seed=cfg.lexicon_seed)
        else:
            raise ValueError(f"unknown morphology: {cfg.morphology}")

    # --- abstract builders ---
    def _mk_np(self, ent: Entity, role: str, definite: Optional[bool] = None,
               depth: int = 0, plural: Optional[bool] = None) -> AbsNP:
        if definite is None:
            definite = self.rng.random() < 0.7
        if plural is None:
            # only non-persons can be plural, and only when enabled
            if (self.cfg.allow_plural_subjects and ent.etype != "person"
                    and role == "nom"
                    and self.rng.random() < self.cfg.plural_subject_prob):
                plural = True
            else:
                plural = False
        adj = None
        if ent.etype != "person" and self.rng.random() < 0.4:
            if ent.color and self.rng.random() < 0.6:
                adj = f"color:{ent.color}"
            elif ent.size:
                adj = f"size:{ent.size}"
        modifier = None
        if (self.cfg.allow_relative and depth < self.cfg.max_clause_depth
                and self.rng.random() < 0.15):
            other = self.rng.choice([e for e in self.world.entities if e.id != ent.id])
            modifier = AbsClause(
                kind="REL",
                tense=self.rng.choice(["present", "past"]),
                verb=self.rng.choice(self.transitive),
                object=self._mk_np(other, "acc", definite=True, depth=depth + 1),
            )
        return AbsNP(entity=ent, role=role, definite=definite, plural=plural,
                     adjective=adj, modifier=modifier)

    def _mk_svo(self, tense: str) -> AbsClause:
        subj = self.rng.choice(self.world.agents())
        obj = self.rng.choice([e for e in self.world.entities if e.id != subj.id])
        return AbsClause(kind="SVO", tense=tense,
                         subject=self._mk_np(subj, "nom"),
                         object=self._mk_np(obj, "acc", definite=True),
                         verb=self.rng.choice(self.transitive))

    def _mk_sv(self, tense: str) -> AbsClause:
        subj = self.rng.choice(self.world.agents())
        return AbsClause(kind="SV", tense=tense,
                         subject=self._mk_np(subj, "nom"),
                         verb=self.rng.choice(self.intransitive))

    def _mk_cop(self) -> AbsClause:
        ent = self.rng.choice(self.world.entities)
        props = []
        if ent.color: props.append(("color", ent.color))
        if ent.size:  props.append(("size", ent.size))
        props.append(("location", ent.location))
        kind, val = self.rng.choice(props)
        return AbsClause(kind="COP", tense="present",
                         subject=self._mk_np(ent, "nom", definite=True),
                         property_kind=kind, property_value=val)

    def _mk_comp(self, tense: str) -> AbsClause:
        subj = self.rng.choice(self.world.people())
        return AbsClause(kind="COMP", tense=tense,
                         subject=self._mk_np(subj, "nom"),
                         verb=self.rng.choice(self.mental),
                         inner=self._mk_svo(tense="present"))

    def _mk_sub(self, tense: str) -> AbsClause:
        return AbsClause(kind="SUB", tense=tense,
                         subject=None,
                         inner=self._mk_svo(tense),
                         object=None,
                         conn=self.rng.choice(["because", "while", "after", "before"]),
                         verb=None)

    # --- realization ---
    def _realize(self, clause: AbsClause) -> list[str]:
        R = self.realizer
        subj_pl = bool(clause.subject and clause.subject.plural)
        if clause.kind == "SVO":
            s = R.np(clause.subject)
            o = R.np(clause.object)
            v = R.verb(clause.verb, clause.tense, subj_pl)
            return R.arrange(s, o, v) + ["."]
        if clause.kind == "SV":
            s = R.np(clause.subject)
            v = R.verb(clause.verb, clause.tense, subj_pl)
            return s + [v, "."]
        if clause.kind == "COP":
            s = R.np(clause.subject)
            return R.copular(s, clause.property_kind, clause.property_value,
                             plural_subj=subj_pl) + ["."]
        if clause.kind == "COMP":
            outer_s = R.np(clause.subject)
            outer_v = R.verb(clause.verb, clause.tense, subj_pl)
            inner = self._realize_non_final(clause.inner)
            return R.complement_assemble(outer_s, outer_v, inner) + ["."]
        if clause.kind == "SUB":
            # two SVO clauses joined by connective particle
            a = self._realize_non_final(clause.inner)
            # build a second SVO
            b_clause = self._mk_svo(clause.tense)
            b = self._realize_non_final(b_clause)
            return a + [R.connective(clause.conn)] + b + ["."]
        raise ValueError(clause.kind)

    def _realize_non_final(self, clause: AbsClause) -> list[str]:
        toks = self._realize(clause)
        if toks and toks[-1] == ".":
            toks = toks[:-1]
        return toks

    # --- public ---
    def generate(self, n: int, fresh_world_every: int = 20) -> list[tuple[list[str], dict]]:
        out = []
        for i in range(n):
            if i % fresh_world_every == 0:
                self.world.new()
            grounded = self.rng.random() < self.cfg.grounded_fraction
            tense = self.rng.choice(["present", "past"])
            r = self.rng.random()
            if r < 0.45:
                clause = self._mk_svo(tense)
            elif r < 0.65:
                clause = self._mk_sv(tense)
            elif r < 0.80:
                clause = self._mk_cop()
            elif r < 0.92 and self.cfg.allow_complement:
                clause = self._mk_comp(tense)
            elif self.cfg.allow_coordination:
                clause = self._mk_sub(tense)
            else:
                clause = self._mk_svo(tense)
            toks = self._realize(clause)
            meta = {"kind": clause.kind, "tense": clause.tense, "grounded": grounded,
                    "abstract": self._serialize_clause(clause)}
            out.append((toks, meta))
        return out

    def _serialize_clause(self, c: AbsClause) -> dict:
        d = {"kind": c.kind, "tense": c.tense}
        if c.subject: d["subject_id"] = c.subject.entity.id
        if c.object:  d["object_id"] = c.object.entity.id
        if c.verb:    d["verb"] = c.verb
        if c.property_kind: d["property"] = (c.property_kind, c.property_value)
        if c.conn:    d["conn"] = c.conn
        if c.inner:   d["inner"] = self._serialize_clause(c.inner)
        return d
