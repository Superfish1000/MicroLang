# MicroLang

**A tunable synthetic language for fast LLM architecture validation.**

Training a new architecture on real human language costs thousands of
GPU-hours before you can tell if it's working. MicroLang is a generator
for synthetic languages that are simple enough to train on in minutes,
but structurally rich enough that learning them exercises the same
capabilities real language does — long-range dependency, compositional
generalization, recursion, morphology, and grounded semantics.

Run your new architecture against MicroLang *first*. If it doesn't
converge cleanly here, don't spend the GPU-hours on real language.

---

## Headline results

Tiny GPT (4 layers, 4 heads, d=128, ~0.8M params), trained on CPU for
~5 minutes at 1,500 steps over 20,000 generated sentences (plural
subjects enabled):

| metric / probe                          | english preset | agglutinative preset |
|-----------------------------------------|---------------:|---------------------:|
| vocab size                              | 210            | 296                  |
| Zipf slope                              | −1.24          | −1.13                |
| trigram-baseline PP                     | 29.2           | 74.4                 |
| **transformer val PP**                  | **17.6**       | **37.0**             |
| held-out PP (new seed)                  | 22.0           | 89.0                 |
| **structure ratio** (shuf/real PP)      | **27.7×**      | **10.5×**            |
| **role-swap correct**                   | **76.4%**      | **75.2%**            |
| agreement across distance (attractor)   | 6.4%           | 1.0%                 |
| grounded-QA consistency                 | 50.8%          | 49.6%                |
| recursion-depth PP (d=0 → d=3)          | 23.1 → 22.3    | 104.9 → 87.7         |
| wall time (CPU, 1,500 steps)            | 305 s          | 269 s                |

### How to read these

- **Trigram vs. transformer PP** — the gap shows how much beyond-local
  structure the transformer captured. Both presets show a large gap,
  meaning both languages actually test structure learning, not just
  local co-occurrence.
- **Structure ratio** = `PP(shuffled) / PP(real)`. A model that has
  learned grammar assigns much higher probability to real token orders
  than to shuffled ones.
- **Role-swap** — for SVO sentences, swap subject and object and ask
  whether the model prefers the correct order. Chance is 50 %.
- **Agreement across distance** — the main verb must agree with the
  plural subject, but a singular noun sits right before the verb
  (inside a relative clause). Chance is 50 %. A 0.8 M-param model at
  20 k sentences **fails this** (6 % / 1 %) — the attractor wins. This
  is expected and is exactly the discriminating signal you want: a
  better architecture should resist the attractor. If your candidate
  arch lifts this number above chance at the same compute budget,
  that's meaningful evidence.
- **Grounded-QA consistency** — prefer "the red cat is red" over "the
  red cat is blue"? At this scale both models are at chance (50 %),
  meaning they haven't learned the compositional-semantic constraint.
  Also a useful discriminator — an architecture that clears this at the
  same budget is doing something real about grounding.
- **Recursion-depth PP** — held-out PP at `max_clause_depth` 0, 1, 2, 3.
  English is flat (no recursion cliff). Agglutinative is non-monotonic;
  depth-0 sentences are simpler and shorter so the distribution shifts —
  interpret with the full ladder, not a single point.

### The lower three are failure probes

Role-swap and structure discrimination pass easily at this tiny scale —
they confirm "the model learned *something*." The agreement, grounded-QA,
and recursion probes are where current architectures run out of gas.
They give you headroom to observe an improvement. Don't interpret
6 %/1 % agreement as "MicroLang is broken" — interpret it as
"MicroLang is hard enough to distinguish architectures that can resist
agreement attractors from ones that can't."

---

## What the language looks like

Both presets are realizations of the **same abstract grammar**, working
from the **same world simulator**. They differ only in surface form.

### English preset

```
the tiny fish hated the cat that pushs the lamp .
eve walks .
a rabbit touched the lamp while mia fixed the red cat .
nick thinks that jack who lifted the small lamp pushs the brown coin .
the red rabbit walked .
```

All morphology is regularized: `heared`, `thinked`, `holded`, `watchs`.
No irregulars, no idioms, no silent ambiguity. Articles are required on
singular count nouns by rule. PP attachment is nearest-head only.

### Agglutinative preset

```
muvi podona fuvukina natekina gaha ke fehedu .
podona natekina bibodu .
poge natekina dohedu zu kunu bavu fuvukina degedu .
futu geva rabo natekina hahedu ke liti lenokina gaha ke fafo .
bavu pogena gosedu .
```

Decoded (with glossing suffixes):

```
alice gray mouse.ACC.DEF watch.PST .
horse.DEF arrive .
alice [gray mouse jack.ACC touch.PST] SUB coin.ACC.DEF fix SUB say .
horse.DEF table.ACC.DEF push.PST BEFORE fish dog.ACC.DEF give.PST .
```

Agglutinative morphology: `ROOT + NUM + CASE + DEF` for nouns,
`ROOT + TENSE + ASPECT + AGR` for verbs. Free/fixed word order (here
SOV). Subordinator particle `ke` marks clause boundaries.

---

## Design

```
  World simulator  ──▶  Grammar  ──▶  Abstract sentence  ──▶  Realizer  ──▶  Tokens
    (entities,          (templates      (subject, object,      (english |     + metadata
     properties,          + sampling)     property, nested      agglutinative)
     state)                               clauses, tense)
```

### Orthogonal configuration axes

| Axis                | Values                                     | Effect                                      |
|---------------------|--------------------------------------------|---------------------------------------------|
| `morphology`        | `english` / `agglutinative`                | Surface forms                               |
| `word_order`        | `SVO` / `SOV` / `free` / `OSV` / `OVS`     | Constituent order                           |
| `case_marking`      | bool                                       | nom/acc/dat/loc suffixes                    |
| `articles`          | bool                                       | Required articles (English)                 |
| `allow_relative`    | bool                                       | `who` / `that` relative clauses             |
| `allow_complement`  | bool                                       | `X thinks that Y`                           |
| `allow_coordination`| bool                                       | `because` / `while` / `after` / `before`    |
| `max_clause_depth`  | int                                        | Cap on nesting                              |
| `grounded_fraction` | float [0, 1]                               | Share of sentences consistent with world    |
| `vocab_nouns/verbs` | int (0 = all)                              | Subsample lexicon                           |
| `lexicon_seed`      | int (fixed default)                        | Agglutinative dictionary seed (stable)      |

Every axis is orthogonal. Flip one to ablate a specific capability;
the others are unaffected. This is what makes MicroLang useful for
architecture validation — you can isolate which structural feature
an arch is failing at.

### Grounded mode

The world simulator holds ~16 entities (people, animals, objects) with
properties (color, size, location) and holdings. In **grounded mode**
each sentence is consistent with true world state; in **distributional
mode** the grammar is sampled without a world constraint. Each sentence
carries metadata identifying its mode and all entity references — this
drives grounded-QA and consistency probes.

The mix is controlled by a single float (`grounded_fraction`), per-
sentence. No change to the training stream format; metadata lives in a
parallel jsonl file.

### Residual issues addressed

| Issue in vanilla English                | MicroEnglish fix                                  |
|-----------------------------------------|---------------------------------------------------|
| Irregular verbs / plurals               | Fully regularized (`heared`, `thinked`)           |
| Article optionality                     | Required on singular count nouns, rule-driven     |
| PP attachment ambiguity                 | Grammar attaches nearest-head only                |
| Garden-path sentences                   | Disallowed by template restriction                |
| Preposition polysemy                    | Fixed set of ~15, one sense each                  |
| Phrasal verb idioms                     | Banned                                            |
| Silent / zero morphology                | All suffixes explicit                             |
| Tense-aspect irregularity               | Restricted to productive combos                   |
| `-s` ambiguity (3sg verb vs. plural)    | Kept intentionally — preserves real texture       |

---

## Quick start

```bash
pip install torch numpy

# generate corpora, train both presets, run basic probes
python -m eval.run_unified --sentences 20000 --steps 1500

# enable the extended probes (agreement, grounded-QA, recursion ladder)
python -m eval.run_unified --sentences 20000 --steps 1500 --all-probes
# or cherry-pick: --agreement  --grounded-qa  --recursion

# re-run just the probes on cached models
python -m eval.rerun_probes --all-probes

# view side-by-side samples (microenglish / conlang / decoded)
python -m eval.samples
```

Extended probes are gated behind flags so the default run stays cheap.

### Use the generator directly

```python
from generators.unified import Grammar, LangConfig

cfg = LangConfig(
    morphology="english",
    word_order="SVO",
    articles=True,
    allow_relative=True,
    allow_complement=True,
    max_clause_depth=2,
    grounded_fraction=0.5,
    seed=42,
)
g = Grammar(cfg)
for tokens, meta in g.generate(10):
    print(" ".join(tokens), "||", meta["kind"])
```

Swap `morphology="agglutinative"` and add `case_marking=True` for the
conlang preset. Any combination of the other toggles is valid.

---

## Repository layout

```
generators/
  world.py             shared entity-state simulator
  unified.py           grammar + realizers + config    [primary]
  microenglish.py      legacy English-only generator
  conlang.py           legacy conlang-only generator

eval/
  stats.py             distributional + structural statistics
  tiny_transformer.py  minimal GPT trainer (saves checkpoints)
  probes.py            capability probes
  run_unified.py       end-to-end driver (primary)
  run_all.py           legacy paired-generator driver
  rerun_probes.py      probe-only driver
  samples.py           side-by-side samples + conlang decoder

corpora/               generated data + metadata
models/                saved checkpoints
report/                results, logs, metrics
docs/superpowers/specs/
  2026-04-19-microlang-design.md     full design spec
```

---

## Probe definitions

**1. Held-out perplexity.** Generate a fresh test set from the same
grammar with a different seed. Measure model PP on it. Baseline sanity
check.

**2. Structure discrimination.** For each test sentence, also measure
PP of a *token-shuffled* version. Report `PP(shuffled) / PP(real)`. A
model that has only learned unigram frequencies gets ratio ≈ 1; a
model that has learned grammar gets ratio ≫ 1.

**3. Role-swap minimal pairs.** For each SVO sentence, generate the
sentence with subject and object swapped. Report the fraction of pairs
where the model assigns higher log-prob to the correct ordering. Chance
= 50 %. Directly probes argument-role learning.

### Extended probes (gated by CLI flags)

**4. Agreement across distance** (`--agreement`). Requires
`allow_plural_subjects=True` in the grammar config. For each plural-subject
SVO sentence with a relative clause between subject and verb (a singular
noun attractor inside the rel clause), build the minimal pair where only
the main verb's agreement suffix differs. Does the model prefer the form
that matches the distant head noun, not the adjacent attractor?

**5. Grounded-QA consistency** (`--grounded-qa`). For a copular sentence
`"the [ADJ] [N] is [VAL]"` where ADJ and VAL are the same property type
(color or size), build the *consistent* version (ADJ == VAL) and the
*inconsistent* version (ADJ ≠ VAL). Does the model prefer the
self-consistent one? Probes compositional-semantic constraint learning.

**6. Recursion-depth ladder** (`--recursion`). Compute held-out
perplexity at `max_clause_depth` 0, 1, 2, 3. A flat curve means the
architecture handles recursion uniformly; a cliff at some depth means
it runs out.

---

## Training setup

Architecture (`eval/tiny_transformer.py`):
- 4 transformer blocks, 4 attention heads
- d_model = 128, block_size = 64, tied embeddings
- ~0.82M parameters total
- AdamW, lr=3e-4, batch=32
- 90/10 train/val split on flattened token stream

Deliberately tiny so CPU training at 1,500 steps takes ~4.5 minutes.
Loss curves descend smoothly, no plateaus, no instability.

---

## Recommendation for architecture validation

1. Run your candidate architecture against **both presets**.
2. Check three signals:
   - Clean loss curve (no plateaus, smooth descent).
   - Structure ratio ≥ ~5× on both presets.
   - Role-swap accuracy well above 50 %.
3. If all three pass at this scale, invest GPU-hours on real-language
   training. If any fail, debug the architecture *here*, where
   iteration is minutes not days.

The gap between n-gram baseline PP and transformer PP is also a useful
signal — an arch that can only match the n-gram floor isn't modeling
structure, regardless of its absolute loss.

---

## License

See repository metadata.

## Credits

Inspired by the use case explored in Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[autoresearch](https://github.com/karpathy/autoresearch) projects.
