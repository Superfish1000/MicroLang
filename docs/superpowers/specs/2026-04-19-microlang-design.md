# MicroLang Design

**Date:** 2026-04-19
**Status:** validated by empirical comparison (see `report/report.md`)

## Motivation

Validating novel LLM architectures on real human-language corpora is
prohibitively expensive. A single architecture change costs thousands to
tens of thousands of GPU-hours before we can tell whether it is working
correctly. We need a training corpus that is:

1. **Fast to train on** (minutes on a single GPU, seconds on CPU at sanity-check scale).
2. **Structurally rich** enough that learning it exercises the same
   capabilities a model would use on real language: long-range dependency,
   compositional generalization, recursion, morphology, grounding.
3. **Directly mappable to a human language** so that performance on the
   synthetic corpus is a believable proxy for performance on real language.

MicroLang is a tunable synthetic-language generator designed to meet
these constraints.

## Core design

A single grammar generates abstract sentence structures from a shared world
simulator. A **realizer** (pluggable) turns abstract structures into surface
tokens. Realizers differ in morphology and word order but not in the
underlying structure they encode — so behavior can be compared apples-to-apples
across surface styles.

```
  World simulator  ──▶  Grammar  ──▶  Abstract sentence  ──▶  Realizer  ──▶  Tokens
    (entities,          (template      (subject, object,       (english |     + metadata
     state)               selection,     property, nested       agglutinative)
                          argument       clauses, tense)
                          sampling)
```

### Realizers

- **English**: regularized English lemmas (~500 roots), English morphology
  (`-s`, `-ed`, `-ing`, `-er`), strict word order, articles, English-style
  relatives (`that` / `who`).
- **Agglutinative**: 2-syllable CVCV roots (1:1 map to English concepts),
  stacked suffixes (NUM-CASE-DEF for nouns, TENSE-ASPECT-AGR for verbs),
  free/fixed word order, subordinator particle `ke`.

Both realizers share the same grammar, the same world simulator, and the
same metadata schema. Any config combination produces valid output.

### Configurable axes

| Axis                | Values                                     | Effect                                      |
|---------------------|--------------------------------------------|---------------------------------------------|
| `morphology`        | `english` / `agglutinative`                | Surface forms                               |
| `word_order`        | `SVO` / `SOV` / `free` / `OSV` / `OVS`     | Constituent ordering                        |
| `case_marking`      | bool                                       | Enable nom/acc/dat/loc suffixes             |
| `articles`          | bool                                       | Required articles (English only)            |
| `allow_relative`    | bool                                       | Relative clauses                            |
| `allow_complement`  | bool                                       | `X thinks that Y` style                     |
| `allow_coordination`| bool                                       | Subordinated conjunctions                   |
| `max_clause_depth`  | int                                        | Cap on clause nesting                       |
| `grounded_fraction` | float [0, 1]                               | Share of sentences consistent with world    |
| `vocab_nouns/verbs` | int (0 = all)                              | Subsample lexicon                           |

Every axis is orthogonal: flipping one does not require re-engineering others.

### Grounding

The world simulator holds ~16 entities (people, animals, objects) with
properties (color, size, location, holdings). In **grounded mode** a
sentence is generated from a true fact or a valid state transition; in
**distributional mode** the same grammar samples without world constraint.
Each sentence carries metadata identifying which mode it came from and
what entities it references — this metadata drives evaluation probes.

Switching is per-sentence (`grounded_fraction` controls the mix). No change
to the training stream format; metadata lives in a parallel jsonl file.

## Evaluation

Three probe categories, all runnable on CPU in seconds given a trained model:

1. **Held-out perplexity** — the standard "does it model the distribution?"
2. **Structure discrimination** — `PP(real) / PP(shuffled)`. Measures whether
   the model distinguishes grammatical from random orderings. Higher = more
   structure-aware.
3. **Role-swap minimal pairs** — for SVO sentences, does the model prefer
   the original argument order to a subject/object swap? Fraction correct
   is a direct readout of positional/case sensitivity.

Planned additions (follow-on work):
- Agreement-across-distance (requires plural subjects)
- Grounded QA (given world state, does model complete correct propositions?)
- Recursion-depth stress test (PP at depths 1, 2, 3, 5)

## Empirical validation (see `report/report.md`)

With 20 k sentences, 1,500 steps, 0.8M-param tiny GPT on CPU (~4.5 min per
language), both realizers produce Zipfian distributions and train stably:

|                    | English preset | Agglutinative preset |
|--------------------|---------------:|---------------------:|
| vocab              | 151            | 193                  |
| Zipf slope         | −0.95          | −0.76                |
| trigram PP         | 22.6           | 50.3                 |
| transformer val PP | 14.7           | 29.0                 |

The agglutinative preset has a higher n-gram floor: it forces models to
track non-local structure (case markers, verb morphology) rather than
local co-occurrence. The English preset converges faster and produces
eye-readable samples.

## Recommendation

Ship the unified generator as the single source of truth. Two default
presets ship for the common cases:

- `english` — fast iteration, debugging, eyeball-readable samples.
- `agglutinative` — stress test for whether an architecture actually
  captures morphology and long-range structure.

Users run their candidate architecture against both and read off:
perplexity curve, structure ratio, role-swap accuracy. A healthy new
architecture should show smooth loss curves, structure ratios above ~5×
on both presets, and role-swap accuracy well above chance. If an
architecture passes all three at this scale, it is worth the GPU-hour
investment for a real-language run.

## Non-goals

- Matching real English's statistical distribution exactly. We want
  *enough* language-likeness, not a faithful corpus model.
- Semantic depth (abstract reasoning, world knowledge). MicroLang
  tests syntactic/structural learning. Semantic eval comes from the
  grounded-mode metadata, not from the surface text.
- Producing natural-sounding text. Regularity is prioritized over
  fluency (we keep `heared`, `thinked`, `holded`).

## File layout

```
generators/
  world.py          shared entity-state simulator
  unified.py        grammar + realizers + config      [new]
  microenglish.py   legacy, kept for reference
  conlang.py        legacy, kept for reference
eval/
  stats.py          distributional + structural stats
  tiny_transformer.py   minimal GPT trainer
  probes.py         capability probes                  [new]
  run_unified.py    end-to-end driver                  [new]
  samples.py        aligned sample viewer + decoder
corpora/            generated data
models/             saved model checkpoints            [new]
report/             results and writeups
docs/superpowers/specs/
  2026-04-19-microlang-design.md    this spec
```

## Follow-on work

1. **Probe suite expansion**: agreement-across-distance (needs plural subjects),
   grounded QA from metadata, explicit recursion-depth stress ladder.
2. **Single-model scan**: train one model, run probes at multiple corpus
   sizes, plot capability-vs-data curves. Fast because training is fast.
3. **Architecture ablation harness**: script that takes a model class,
   runs both presets, and emits a scorecard — the intended consumer
   interface.
4. **Larger world sim**: more entities, actions, state transitions, so
   grounded-mode has richer ground-truth to probe against.
