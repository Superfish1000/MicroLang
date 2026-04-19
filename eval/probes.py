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


def run_all_probes(model_path: Path, cfg: LangConfig, label: str) -> dict:
    model, stoi, itos = load_model(model_path)
    print(f"\n=== Probes: {label} ===")
    pp = probe_held_out(model, stoi, cfg)
    print(f"  held-out PP:       {pp:.3f}")
    s = probe_structure(model, stoi, cfg)
    print(f"  real PP:           {s['real_pp']:.3f}")
    print(f"  shuffled PP:       {s['shuffled_pp']:.3f}")
    print(f"  structure ratio:   {s['ratio']:.3f}x  (higher = more structure-aware)")
    r = probe_role_swap(model, stoi, cfg)
    print(f"  role-swap correct: {r['correct_preferred_pct']*100:.1f}% of {r['n']}")
    print(f"    avg NLL correct: {r['avg_correct_nll']:.3f}")
    print(f"    avg NLL swapped: {r['avg_swapped_nll']:.3f}")
    return {"held_out_pp": pp, "structure": s, "role_swap": r}


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
