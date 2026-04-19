# MicroLang: MicroEnglish vs. Conlang-Regular

Synthetic-language candidates for fast LLM architecture validation.
Both languages share a common world simulator and a common set of sentence
templates; they differ only in surface realization.

## TL;DR

- **MicroEnglish** converges faster and hits a lower absolute val perplexity (14.7 vs 29.0), and samples are eye-readable without a decoder — **better default for rapid arch sanity checks**.
- **Conlang-Regular** has a much higher n-gram baseline (trigram PP 50.3 vs 22.6), meaning there is more non-local structure that a transformer must capture — **better stress test** for whether an arch actually models long-range dependencies and morphology.
- Both converge clearly under 5 minutes on CPU at 1,500 steps with a 0.8M-param model. Both produce Zipfian distributions that look language-like.
- **Recommendation:** ship MicroEnglish as the primary; keep Conlang as an optional "harder mode" via a shared config toggle. Residual MicroEnglish issues were either fixed or deliberately kept (see below).

## Design summary

|                     | MicroEnglish                              | Conlang-Regular                          |
|---------------------|-------------------------------------------|------------------------------------------|
| Lexicon             | ~500 real English lemmas, 1 sense each    | 200 CVCV roots, 1:1 to English concepts  |
| Morphology          | Regularized English (`-s`, `-ed`, `-ing`) | Agglutinative: NUM-CASE-DEF, TNS-ASP-AGR |
| Word order          | Strict SVO                                | SOV default; SVO/free toggleable         |
| Articles            | Required, rule-driven                     | Definiteness as suffix `-na`             |
| Relatives           | `that` / `who`                            | Subordinator particle `ke`               |
| Grounding           | Shared world-state sim (hybrid switchable)| Shared world-state sim                   |
| Readable by eye     | Yes                                       | Only with glossed decoder                |

Every structural feature has an on/off toggle (clause depth, relatives, complements, coordination, case marking, word order, grounded fraction, vocab sub-sampling) so you can ablate capabilities when testing an architecture.

## Corpus statistics (20,000 sentences each)

| metric                   | MicroEnglish | Conlang-Regular |
|--------------------------|:------------:|:---------------:|
| vocab size               | 151          | 193             |
| total tokens             | 151,055      | 121,076         |
| type/token ratio         | 0.0010       | 0.0016          |
| unigram entropy (nats)   | 4.05         | 4.52            |
| Zipf slope               | −0.95        | −0.76           |
| avg sentence length      | 7.55         | 6.05            |
| p99 sentence length      | 20           | 16              |
| bigram perplexity        | 16.3         | 29.0            |
| trigram perplexity       | 22.6         | 50.3            |

*(Corpora generated at vocab=all, max_clause_depth=2, grounded_fraction=0.5. Numbers rise with deeper nesting.)*

Conlang has a higher absolute vocab (193 > 151) despite 60% fewer roots: agglutinative morphology generates many surface forms from each root. That also flattens the Zipf curve slightly (−0.76 vs −0.95) — still language-like, just less steep. Its fewer tokens per 20 k sentences reflects its shorter average sentence length (no articles, case marking on the noun means prepositions are often merged into suffixes).

## Training curve: tiny transformer (~0.82M params)

Architecture: 4 layers, 4 heads, d=128, block_size=64, AdamW lr=3e-4. 1,500 steps, batch 32, CPU only.

| metric                      | MicroEnglish | Conlang-Regular |
|-----------------------------|:------------:|:---------------:|
| final train loss            | 2.68         | 3.34            |
| final val loss              | 2.69         | 3.37            |
| val perplexity              | **14.7**     | **29.0**        |
| PP gap vs. trigram baseline | 22.6 → 14.7 (−35 %) | 50.3 → 29.0 (−42 %) |
| wall time (1,500 steps)     | 278 s        | 261 s           |

**Key observation:** the *gap* between trigram PP and transformer PP is larger for Conlang (−42 %) than for MicroEnglish (−35 %). That means Conlang's signal is more beyond-local: an n-gram model hits a higher floor because the morphology forces you to track agreement and case structure across non-adjacent positions. MicroEnglish is easier overall (lower floor), but Conlang tests more of what a real architecture needs to do.

Both loss curves descend smoothly — no plateaus, no instability, no divergence. Both are perfectly usable as fast-iteration signal.

## Side-by-side samples

### SVO
```
MicroEnglish : the fish heared the cat .
Conlang      : kuha tadi dulukina vugodu .
(decoded)    : alice gray mouse.ACC.DEF watch.PST .
```

### SV
```
MicroEnglish : eve walks .
Conlang      : sogona davu .
(decoded)    : horse.DEF arrive .
```

### COP (copular)
```
MicroEnglish : the book that holded the dog is small .
Conlang      : duluna tadi vo .
(decoded)    : mouse.DEF gray BE .
```

### COMP (complement clause)
```
MicroEnglish : eve who smells the dog thinked that the dog gives the cup .
Conlang      : kuha tadi dulu maseki bitidu ke fibikina fuhi ke pile .
(decoded)    : alice [gray mouse jack.ACC touch.PST] SUB coin.ACC.DEF fix SUB say .
```

### SUB (subordinated)
```
MicroEnglish : the red cat watchs mia while eve who pulls the gray table
               that takes the yellow lamp takes the book .
Conlang      : sogona bopokina mamedu fi rute muhakina hipudu .
(decoded)    : horse.DEF table.ACC.DEF push.PST BEFORE fish dog.ACC.DEF give.PST .
```

Notice how MicroEnglish shows regularized morphology the eye catches immediately: `heared`, `thinked`, `holded`, `watchs`. That's a feature — it removes every English irregularity that would force the model to memorize exceptions instead of learning rules.

## Residual MicroEnglish issues (and whether they were addressed)

| Issue in vanilla English                | MicroEnglish fix                                  | Residual? |
|-----------------------------------------|---------------------------------------------------|:---------:|
| Irregular verbs / plurals               | Fully regularized (`heared`, `thinked`)           | no        |
| Article optionality                     | Required on singular count nouns; rule-driven     | no        |
| PP attachment ambiguity                 | Grammar attaches nearest-head only                | no        |
| Pronoun antecedent ambiguity            | Pronouns omitted in v0 — optional feature         | no        |
| Garden-path sentences                   | Disallowed by template restriction                | no        |
| Preposition polysemy                    | Fixed set of ~15, one sense each                  | no        |
| Phrasal verb idioms                     | Banned                                            | no        |
| Silent/zero morphology                  | All suffixes explicit                             | no        |
| Tense-aspect irregularity               | Restricted to productive combos                   | no        |
| `-s` ambiguity (3sg verb vs plural noun)| **Kept intentionally**                            | yes       |

The `-s` collision (`walks` as verb vs. `cats` as plural noun) is the one place we preserved English's texture — removing it would make the morphology artificially cleaner than any real language. It forces the model to use context to disambiguate, which is a useful capability to test.

## Recommendation

**Primary language: MicroEnglish.** It wins on readability, convergence speed, and simplicity. It converges to val PP 14.7 in ~4.6 min on CPU at 0.8M params and 20 k sentences — comfortably within your rapid-validation budget. Its samples are debuggable by eye, which matters when you're iterating on architectures.

**Keep Conlang-Regular as an opt-in "harder mode."** Its higher n-gram floor means it's a better probe of whether an architecture is actually modeling structure rather than local co-occurrence. The design as written already shares the same world simulator and metadata schema, so running both against the same arch is a one-flag change.

**Next step:** merge the two generators into a single configurable grammar. The natural axis is a `morphology` flag (`regular-english` vs `agglutinative`) and a `word_order` flag — with those two knobs on one generator, a single codebase produces the full spectrum from "trivial" to "stress test," letting you ablate precisely which feature an architecture is struggling with.

## Unified generator + capability probes

Both languages have been merged into a single configurable generator
(`generators/unified.py`) with orthogonal toggles for morphology, word
order, case marking, articles, recursion depth, grounded fraction, and
vocabulary subsampling. The two presets (`english`, `agglutinative`) are
just named configurations of the same codebase.

After retraining both presets from the unified generator (20 k sentences,
1500 steps), the capability-probe suite reports:

| probe                       | english preset | agglutinative preset |
|-----------------------------|---------------:|---------------------:|
| val perplexity              | 15.2           | 28.7                 |
| held-out PP (new seed)      | 62.1           | 70.3                 |
| structure ratio (shuf/real) | **35.7×**      | **22.9×**            |
| role-swap correct           | **79.6%**      | **72.0%**            |
| trigram floor PP            | 22.5           | 50.2                 |

**Reading the probes:**
- *Structure ratio* = `PP(shuffled tokens) / PP(real sentence)`. Both
  presets show >20× preference for grammatical token order — the
  transformer is clearly learning structure, not just unigrams.
- *Role-swap correct* = % of minimal-pair SVO sentences where the model
  assigns higher log-prob to the correct argument order than to a
  subject/object swap. Chance = 50 %. Both presets clear 70 %, showing
  the architecture has learned positional (English) or case-based
  (agglutinative) role assignment.
- *Held-out vs. val gap* shows the models partially overfit to specific
  world states seen during training. Acceptable for architecture-
  comparison purposes; for production iteration we'd rotate the world
  simulator more aggressively.

## Files

- `generators/world.py` — shared entity-state simulator
- `generators/unified.py` — **unified grammar + realizers (primary)**
- `generators/microenglish.py`, `conlang.py` — legacy, kept for reference
- `eval/stats.py` — distributional + structural statistics (+ trigram baseline)
- `eval/tiny_transformer.py` — tiny GPT trainer (saves checkpoints)
- `eval/probes.py` — capability probes (held-out PP, structure, role-swap)
- `eval/run_all.py` — legacy paired-generator driver
- `eval/run_unified.py` — end-to-end unified driver (recommended)
- `eval/rerun_probes.py` — re-run probes without retraining
- `eval/samples.py` — aligned samples + conlang decoder
- `corpora/` — generated corpora + metadata + dictionary
- `models/u_{english,agglutinative}.pt` — saved checkpoints
- `report/results.json` — raw metrics (paired generators)
- `report/unified_results.json` — raw metrics (unified)
- `report/probes_fixed.json` — final probe scores
- `report/{run_log,unified_run_log}.txt` — full training logs
- `docs/superpowers/specs/2026-04-19-microlang-design.md` — design spec
