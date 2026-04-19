"""Capability probes for trained MicroLang models.

Three probes, each reporting a model's ability to capture non-trivial structure:

  1. HELD_OUT_PP   — perplexity on a fresh test corpus from the same grammar.
  2. STRUCTURE     — PP(real test) vs PP(shuffled test). Ratio > 1 means the
                     model prefers grammatical token orders to random ones.
  3. ROLE_SWAP     — for SVO sentences with distinct animate-vs-inanimate args,
                     does the model assign higher log-prob to the correct
                     argument order than to the swapped one?

All probes run on CPU in seconds given a cached model.
"""
from __future__ import annotations
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generators.unified import Grammar, LangConfig
from generators.world import COLORS, SIZES
from eval.tiny_transformer import TinyGPT


def load_model(path: Path, device="cpu") -> tuple[TinyGPT, dict, list[str]]:
    blob = torch.load(path, map_location=device, weights_only=False)
    stoi = blob["stoi"]
    itos = blob["itos"]
    cfg = blob["model_cfg"]
    model = TinyGPT(vocab_size=cfg["vocab_size"], d=cfg["d"], heads=cfg["heads"],
                    layers=cfg["layers"], block_size=cfg["block_size"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, stoi, itos


def sentence_nll(model: TinyGPT, stoi: dict, tokens: list[str]) -> float:
    """Total NLL of a token sequence under the model (sum over positions)."""
    ids = [stoi["<s>"]] + [stoi.get(t, stoi["<s>"]) for t in tokens]
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
    logp = F.log_softmax(logits[0], dim=-1)
    nll = -logp[torch.arange(len(y[0])), y[0]].sum().item()
    return nll


def corpus_pp(model, stoi, sents: list[list[str]]) -> float:
    total_nll = 0.0
    total_tok = 0
    for s in sents:
        nll = sentence_nll(model, stoi, s)
        total_nll += nll
        total_tok += len(s)
    return math.exp(total_nll / total_tok)


def probe_held_out(model, stoi, cfg: LangConfig, n=500) -> float:
    test_cfg = LangConfig(**{**cfg.__dict__, "seed": cfg.seed + 999})
    g = Grammar(test_cfg)
    sents = [toks for toks, _ in g.generate(n)]
    return corpus_pp(model, stoi, sents)


def probe_structure(model, stoi, cfg: LangConfig, n=500) -> dict:
    rng = random.Random(cfg.seed + 5000)
    test_cfg = LangConfig(**{**cfg.__dict__, "seed": cfg.seed + 5000})
    g = Grammar(test_cfg)
    sents = [toks for toks, _ in g.generate(n)]
    real_pp = corpus_pp(model, stoi, sents)
    shuf = []
    for s in sents:
        body = s[:-1] if s and s[-1] == "." else s[:]
        rng.shuffle(body)
        shuf.append(body + (["."] if s and s[-1] == "." else []))
    shuf_pp = corpus_pp(model, stoi, shuf)
    return {"real_pp": real_pp, "shuffled_pp": shuf_pp,
            "ratio": shuf_pp / real_pp}


def probe_role_swap(model, stoi, cfg: LangConfig, n=500) -> dict:
    """Build SVO-only sentences and compare to subject/object swapped variants."""
    test_cfg = LangConfig(**{**cfg.__dict__, "seed": cfg.seed + 8000,
                             "allow_complement": False, "allow_coordination": False,
                             "allow_relative": False})
    g = Grammar(test_cfg)
    from generators.unified import AbsClause
    wins = 0
    tries = 0
    total_correct_nll = 0.0
    total_swap_nll = 0.0
    while tries < n:
        # produce an SVO clause directly
        g.world.new()
        clause = g._mk_svo(tense=g.rng.choice(["present", "past"]))
        if clause.kind != "SVO":
            continue
        normal = g._realize(clause)
        # swap subject and object
        swapped_clause = AbsClause(kind="SVO", tense=clause.tense,
                                    subject=g._mk_np(clause.object.entity, "nom"),
                                    object=g._mk_np(clause.subject.entity, "acc", definite=True),
                                    verb=clause.verb)
        # rebuild with the original entities swapped
        swapped_clause.subject = clause.object
        swapped_clause.subject.role = "nom"
        swapped_clause.object = clause.subject
        swapped_clause.object.role = "acc"
        swapped = g._realize(swapped_clause)
        n_nll = sentence_nll(model, stoi, normal)
        s_nll = sentence_nll(model, stoi, swapped)
        total_correct_nll += n_nll
        total_swap_nll += s_nll
        if n_nll < s_nll:
            wins += 1
        tries += 1
    return {"n": tries, "correct_preferred_pct": wins / tries,
            "avg_correct_nll": total_correct_nll / tries,
            "avg_swapped_nll": total_swap_nll / tries}


def probe_agreement(model, stoi, cfg: LangConfig, n=500) -> dict:
    """Subject-verb agreement across a relative clause.

    Requires cfg.allow_plural_subjects=True. Builds minimal pairs where the
    only difference is the main verb's agreement form; the correct form
    agrees with the (possibly distant) subject head, not the nearer noun
    inside the relative clause.
    """
    if not cfg.allow_plural_subjects:
        return {"skipped": True, "reason": "cfg.allow_plural_subjects=False"}
    # Force plural subjects + relative clauses at test time
    test_cfg = LangConfig(**{**cfg.__dict__,
                             "seed": cfg.seed + 3000,
                             "allow_plural_subjects": True,
                             "plural_subject_prob": 1.0,
                             "allow_relative": True,
                             "allow_complement": False,
                             "allow_coordination": False,
                             "max_clause_depth": 1})
    g = Grammar(test_cfg)
    wins = 0; tries = 0
    total_c = 0.0; total_w = 0.0
    while tries < n:
        g.world.new()
        clause = g._mk_svo(tense="present")
        # require plural subject AND a relative modifier for a real test
        if not clause.subject.plural or clause.subject.modifier is None:
            continue
        # object inside rel clause must differ in number from subject (always
        # singular here since we only plural-ify subject NPs)
        correct = g._realize(clause)
        # build wrong-agreement variant: flip plural on subject used only at
        # verb realization. We rebuild with subject flagged singular for verb.
        R = g.realizer
        s = R.np(clause.subject)
        o = R.np(clause.object)
        v_wrong = R.verb(clause.verb, clause.tense, False)
        wrong = R.arrange(s, o, v_wrong) + ["."]
        if correct == wrong:
            continue  # agreement unmarked in this form (shouldn't happen here)
        c_nll = sentence_nll(model, stoi, correct)
        w_nll = sentence_nll(model, stoi, wrong)
        total_c += c_nll; total_w += w_nll
        if c_nll < w_nll:
            wins += 1
        tries += 1
    return {"n": tries, "correct_preferred_pct": wins / tries,
            "avg_correct_nll": total_c / tries,
            "avg_wrong_nll": total_w / tries}


def probe_grounded_qa(model, stoi, cfg: LangConfig, n=500) -> dict:
    """Internal-consistency probe on copular sentences.

    For a sentence "the [ADJ] [N] is [VAL]" where ADJ and VAL are the same
    property kind (e.g., color), the correct variant has ADJ == VAL
    (consistent) and the distractor has ADJ != VAL (contradicting itself).
    A model that has learned compositional semantics prefers the
    consistent one.
    """
    test_cfg = LangConfig(**{**cfg.__dict__,
                             "seed": cfg.seed + 4000,
                             "allow_plural_subjects": False,
                             "grounded_fraction": 1.0,
                             "max_clause_depth": 0})
    g = Grammar(test_cfg)
    wins = 0; tries = 0
    total_c = 0.0; total_w = 0.0
    attempts = 0
    while tries < n and attempts < n * 20:
        attempts += 1
        g.world.new()
        clause = g._mk_cop()
        # need an adjective and a property that's color or size
        if clause.property_kind not in ("color", "size"):
            continue
        # force subject NP to have an adjective OF THE SAME KIND
        ent = clause.subject.entity
        if ent.etype == "person":
            continue
        if clause.property_kind == "color" and ent.color:
            clause.subject.adjective = f"color:{ent.color}"
            correct_val = ent.color
        elif clause.property_kind == "size" and ent.size:
            clause.subject.adjective = f"size:{ent.size}"
            correct_val = ent.size
        else:
            continue
        # build the consistent version (ADJ == VAL)
        clause.property_value = correct_val
        consistent = g._realize(clause)
        # build the inconsistent version with a different value of same kind
        alt_pool = [c for c in (COLORS if clause.property_kind == "color" else SIZES)
                    if c != correct_val]
        bad_val = g.rng.choice(alt_pool)
        clause.property_value = bad_val
        inconsistent = g._realize(clause)
        c_nll = sentence_nll(model, stoi, consistent)
        w_nll = sentence_nll(model, stoi, inconsistent)
        total_c += c_nll; total_w += w_nll
        if c_nll < w_nll:
            wins += 1
        tries += 1
    if tries == 0:
        return {"skipped": True, "reason": "no eligible sentences generated"}
    return {"n": tries, "consistent_preferred_pct": wins / tries,
            "avg_consistent_nll": total_c / tries,
            "avg_inconsistent_nll": total_w / tries}


def probe_recursion_ladder(model, stoi, cfg: LangConfig,
                            depths=(0, 1, 2, 3), n=300) -> dict:
    """Held-out PP at multiple max_clause_depth settings.

    A flat or mildly increasing curve = arch handles recursion; a cliff
    = fails at some depth.
    """
    results = {}
    for d in depths:
        test_cfg = LangConfig(**{**cfg.__dict__,
                                 "seed": cfg.seed + 6000 + d,
                                 "max_clause_depth": d})
        g = Grammar(test_cfg)
        sents = [toks for toks, _ in g.generate(n)]
        results[f"depth_{d}"] = corpus_pp(model, stoi, sents)
    return results


def run_all_probes(model_path: Path, cfg: LangConfig, label: str,
                    enable_agreement: bool = False,
                    enable_grounded_qa: bool = False,
                    enable_recursion: bool = False) -> dict:
    model, stoi, itos = load_model(model_path)
    print(f"\n=== Probes: {label} ===")
    pp = probe_held_out(model, stoi, cfg)
    print(f"  held-out PP:       {pp:.3f}")
    s = probe_structure(model, stoi, cfg)
    print(f"  real PP:           {s['real_pp']:.3f}")
    print(f"  shuffled PP:       {s['shuffled_pp']:.3f}")
    print(f"  structure ratio:   {s['ratio']:.3f}x")
    r = probe_role_swap(model, stoi, cfg)
    print(f"  role-swap correct: {r['correct_preferred_pct']*100:.1f}% of {r['n']}")
    out = {"held_out_pp": pp, "structure": s, "role_swap": r}
    if enable_agreement:
        a = probe_agreement(model, stoi, cfg)
        if a.get("skipped"):
            print(f"  agreement:         skipped ({a['reason']})")
        else:
            print(f"  agreement correct: {a['correct_preferred_pct']*100:.1f}% of {a['n']}")
        out["agreement"] = a
    if enable_grounded_qa:
        q = probe_grounded_qa(model, stoi, cfg)
        if q.get("skipped"):
            print(f"  grounded-qa:       skipped ({q['reason']})")
        else:
            print(f"  grounded-qa consistent: {q['consistent_preferred_pct']*100:.1f}% of {q['n']}")
        out["grounded_qa"] = q
    if enable_recursion:
        ladder = probe_recursion_ladder(model, stoi, cfg)
        print(f"  recursion ladder:  " + "  ".join(f"{k}={v:.1f}" for k, v in ladder.items()))
        out["recursion_ladder"] = ladder
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--cfg-json", help="JSON of LangConfig fields")
    ap.add_argument("--label", default="model")
    args = ap.parse_args()
    cfg_dict = json.loads(args.cfg_json) if args.cfg_json else {}
    cfg = LangConfig(**cfg_dict)
    run_all_probes(Path(args.model), cfg, args.label)
