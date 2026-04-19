"""Shared mini-world simulator for grounded sentence generation.

The world holds entities with properties and relations. Generators sample
from this state to produce sentences tagged with ground-truth metadata.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional


ENTITY_TYPES = ["person", "animal", "object"]
PERSON_NAMES = ["alice", "bob", "carol", "dan", "eve", "frank", "gina", "hank",
                "ivy", "jack", "kate", "leo", "mia", "nick", "olga", "paul"]
ANIMAL_KINDS = ["cat", "dog", "bird", "fish", "horse", "mouse", "rabbit", "fox"]
OBJECT_KINDS = ["book", "cup", "ball", "chair", "table", "box", "key", "hat",
                "rock", "stick", "lamp", "coin", "rope", "bag"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "gray"]
SIZES = ["small", "big", "tiny", "huge"]
LOCATIONS = ["house", "garden", "park", "forest", "kitchen", "room", "field",
             "river", "hill", "road"]


@dataclass
class Entity:
    id: int
    name: str           # "alice" or "cat3"
    etype: str          # person | animal | object
    kind: str           # for animals/objects, the kind word; for people = name
    color: Optional[str]
    size: Optional[str]
    location: str
    holds: list[int] = field(default_factory=list)   # entity ids this entity holds

    def referring_noun(self) -> str:
        """Base noun for referring expressions."""
        if self.etype == "person":
            return self.name
        return self.kind


@dataclass
class World:
    rng: random.Random
    entities: list[Entity] = field(default_factory=list)

    def new(self, n_people=4, n_animals=4, n_objects=8) -> None:
        self.entities.clear()
        next_id = 0
        names = self.rng.sample(PERSON_NAMES, n_people)
        for nm in names:
            self.entities.append(Entity(next_id, nm, "person", nm, None,
                                        self.rng.choice(SIZES),
                                        self.rng.choice(LOCATIONS)))
            next_id += 1
        for _ in range(n_animals):
            k = self.rng.choice(ANIMAL_KINDS)
            self.entities.append(Entity(next_id, f"{k}{next_id}", "animal", k,
                                        self.rng.choice(COLORS),
                                        self.rng.choice(SIZES),
                                        self.rng.choice(LOCATIONS)))
            next_id += 1
        for _ in range(n_objects):
            k = self.rng.choice(OBJECT_KINDS)
            self.entities.append(Entity(next_id, f"{k}{next_id}", "object", k,
                                        self.rng.choice(COLORS),
                                        self.rng.choice(SIZES),
                                        self.rng.choice(LOCATIONS)))
            next_id += 1

    def people(self) -> list[Entity]:
        return [e for e in self.entities if e.etype == "person"]

    def animals(self) -> list[Entity]:
        return [e for e in self.entities if e.etype == "animal"]

    def objects(self) -> list[Entity]:
        return [e for e in self.entities if e.etype == "object"]

    def agents(self) -> list[Entity]:
        return [e for e in self.entities if e.etype in ("person", "animal")]
