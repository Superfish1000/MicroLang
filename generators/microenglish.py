"""MicroEnglish generator.

A regular, unambiguous subset of English with:
  - strict SVO, required articles, regular morphology
  - optional relative clauses and complement clauses (toggleable, depth-capped)
  - grounded mode (sentences consistent with world state) OR distributional mode
  - per-sentence metadata for probe evals

Output token stream has no capitalization; sentence-final period is its own token.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional
from .world import World, Entity, COLORS, SIZES, LOCATIONS


# ---- Lexicon ------------------------------------------------------------
TRANSITIVE_VERBS = [
    "see", "hold", "carry", "want", "find", "push", "pull", "lift",
    "drop", "take", "give", "make", "break", "fix", "like", "hate",
    "watch", "follow", "meet", "help", "call", "hear", "smell", "touch",
]
INTRANSITIVE_VERBS = [
    "sleep", "wait", "run", "walk", "laugh", "cry", "jump", "sing",
    "dance", "rest", "arrive", "leave",
]
MOTION_VERBS = ["go", "run", "walk", "move"]   # used with "to LOC"
MENTAL_VERBS = ["think", "know", "believe", "say"]  # take complement clauses
QUANTIFIERS = ["every", "some", "no"]
CONNECTIVES_SUB = ["because", "while", "after", "before"]

FUNCTION_WORDS = {
    "the", "a", "to", "in", "on", "under", "near", "with",
    "and", "that", "who", "is", "are", "was", "were", "not",
    "of", "by",
}


# ---- Morphology ---------------------------------------------------------
def verb_form(lemma: str, tense: str, subject_plural: bool) -> str:
    """Regular English verb morphology. No irregulars allowed."""
    if tense == "past":
        # regular -ed (we enforce verbs end in consonant or 'e')
        return lemma + "d" if lemma.endswith("e") else lemma + "ed"
    if tense == "prog":
        return (lemma[:-1] if lemma.endswith("e") else lemma) + "ing"
    # present
    if subject_plural:
        return lemma
    return lemma + "s"


def noun_form(lemma: str, plural: bool) -> str:
    return lemma + "s" if plural else lemma


# ---- Config -------------------------------------------------------------
@dataclass
class MEConfig:
    vocab_nouns: int = 0             # 0 = use all
    vocab_verbs: int = 0
    max_clause_depth: int = 2
    allow_relative: bool = True
    allow_complement: bool = True
    allow_coordination: bool = True
    grounded_fraction: float = 0.5
    seed: int = 0


# ---- Generator ----------------------------------------------------------
class MicroEnglish:
    def __init__(self, cfg: MEConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.world = World(self.rng)
        self.transitive = (TRANSITIVE_VERBS[:cfg.vocab_verbs]
                           if cfg.vocab_verbs else TRANSITIVE_VERBS)[:]
        self.intransitive = INTRANSITIVE_VERBS[:]
        self.motion = MOTION_VERBS[:]
        self.mental = MENTAL_VERBS[:]

    # --- NP generation ---
    def _np(self, entity: Optional[Entity] = None, depth: int = 0,
            definite: Optional[bool] = None, plural: bool = False) -> tuple[list[str], Entity]:
        """Return (tokens, referent). If entity given, describe it; else sample."""
        if entity is None:
            entity = self.rng.choice(self.world.entities)
        tokens: list[str] = []

        # determiner
        if entity.etype == "person":
            # proper nouns bare
            tokens.append(entity.name)
            # optional relative clause
            if (self.cfg.allow_relative and depth < self.cfg.max_clause_depth
                    and self.rng.random() < 0.15):
                tokens += self._rel_clause(entity, depth + 1)
            return tokens, entity

        det = "the" if (definite if definite is not None else self.rng.random() < 0.7) else "a"
        tokens.append(det)
        # adjective
        if self.rng.random() < 0.4:
            if entity.color and self.rng.random() < 0.6:
                tokens.append(entity.color)
            elif entity.size:
                tokens.append(entity.size)
        tokens.append(noun_form(entity.kind, plural))
        if (self.cfg.allow_relative and depth < self.cfg.max_clause_depth
                and self.rng.random() < 0.15):
            tokens += self._rel_clause(entity, depth + 1)
        return tokens, entity

    def _rel_clause(self, head: Entity, depth: int) -> list[str]:
        """`who/that V [NP]` modifying head."""
        rel = "who" if head.etype == "person" else "that"
        verb = self.rng.choice(self.transitive)
        obj = self.rng.choice([e for e in self.world.entities if e.id != head.id])
        obj_toks, _ = self._np(obj, depth=depth, definite=True)
        tense = self.rng.choice(["present", "past"])
        return [rel, verb_form(verb, tense, False)] + obj_toks

    # --- Sentence templates ---
    def _svo(self, tense: str) -> tuple[list[str], dict]:
        subj = self.rng.choice(self.world.agents())
        obj = self.rng.choice([e for e in self.world.entities if e.id != subj.id])
        verb = self.rng.choice(self.transitive)
        s_toks, _ = self._np(subj, depth=0)
        o_toks, _ = self._np(obj, depth=0, definite=True)
        toks = s_toks + [verb_form(verb, tense, False)] + o_toks + ["."]
        meta = {"template": "SVO", "subject_id": subj.id, "object_id": obj.id,
                "verb": verb, "tense": tense}
        return toks, meta

    def _sv(self, tense: str) -> tuple[list[str], dict]:
        subj = self.rng.choice(self.world.agents())
        verb = self.rng.choice(self.intransitive)
        s_toks, _ = self._np(subj, depth=0)
        toks = s_toks + [verb_form(verb, tense, False), "."]
        meta = {"template": "SV", "subject_id": subj.id, "verb": verb, "tense": tense}
        return toks, meta

    def _copular(self) -> tuple[list[str], dict]:
        ent = self.rng.choice(self.world.entities)
        # pick a true property
        prop_choices = []
        if ent.color:
            prop_choices.append(("color", ent.color))
        if ent.size:
            prop_choices.append(("size", ent.size))
        prop_choices.append(("location", ent.location))
        kind, val = self.rng.choice(prop_choices)
        s_toks, _ = self._np(ent, depth=0, definite=True)
        if kind == "location":
            toks = s_toks + ["is", "in", "the", val, "."]
        else:
            toks = s_toks + ["is", val, "."]
        meta = {"template": "COP", "subject_id": ent.id, "property": kind, "value": val}
        return toks, meta

    def _complement(self, tense: str) -> tuple[list[str], dict]:
        subj = self.rng.choice(self.world.people())
        verb = self.rng.choice(self.mental)
        inner, inner_meta = self._svo(tense="present")
        # strip trailing period from inner
        if inner[-1] == ".":
            inner = inner[:-1]
        s_toks, _ = self._np(subj, depth=0)
        toks = s_toks + [verb_form(verb, tense, False), "that"] + inner + ["."]
        meta = {"template": "COMP", "subject_id": subj.id, "verb": verb, "inner": inner_meta}
        return toks, meta

    def _coordinated(self, tense: str) -> tuple[list[str], dict]:
        a, ma = self._svo(tense)
        b, mb = self._svo(tense)
        if a[-1] == ".":
            a = a[:-1]
        if b[-1] == ".":
            b = b[:-1]
        conn = self.rng.choice(CONNECTIVES_SUB)
        return a + [conn] + b + ["."], {"template": "SUB", "outer": ma, "inner": mb, "conn": conn}

    # --- Public ---
    def generate(self, n_sentences: int, fresh_world_every: int = 20) -> list[tuple[list[str], dict]]:
        out = []
        for i in range(n_sentences):
            if i % fresh_world_every == 0:
                self.world.new()
            grounded = self.rng.random() < self.cfg.grounded_fraction
            tense = self.rng.choice(["present", "past"])
            # weights: SVO common, SV some, COP some, COMP rarer, SUB rarer
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
