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
~4.6 minutes at 1,500 steps over 20,000 generated sentences:

| probe                       | english preset | agglutinative preset |
|-----------------------------|---------------:|---------------------:|
| vocab size                  | 151            | 193                  |
| Zipf slope                  | −0.95          | −0.76                |
| trigram-baseline PP         | 22.5           | 50.2                 |
| **transformer val PP**      | **15.2**       | **28.7**             |
| held-out PP (new seed)      | 62.1           | 70.3                 |
| **structure ratio** (shuf/real PP) | **35.7×** | **22.9×**           |
| **role-swap correct**       | **79.6%**      | **72.0%**            |
| wall time (CPU, 1,500 steps)| 278 s          | 270 s                |

### How to read these

- **Trigram PP vs. transformer PP**: the gap shows how much
  beyond-local structure the transformer captured. Both presets show a
  large gap, meaning both languages actually test *structure learning*,
  not just local co-occurrence.
- **Structure ratio** = `PP(shuffled) / PP(real)`. A model that has
  learned grammar assigns much higher probability to real token orders
  than to shuffled ones. Both presets show >20× preference.
- **Role-swap**: for SVO sentences, we swap subject and object and ask
  whether the model still prefers the correct order. Chance is 50 %.
  Both presets clear 70 %, showing genuine positional / case-based role
  assignment was learned.

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

# generate corpora, train both presets, run probes
python -m eval.run_unified --sentences 20000 --steps 1500

# re-run just the probes on cached models
python -m eval.rerun_probes

# view side-by-side samples
python -m eval.samples
```

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

### Follow-on probes (not yet implemented)

- **Agreement across distance** (requires plural subjects in grammar)
- **Grounded QA**: given world state + partial sentence, predict correct
  completion using world metadata
- **Recursion-depth stress ladder**: PP at depths 1, 2, 3, 5

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
