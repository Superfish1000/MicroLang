"""Generate corpora for both languages and run the full comparison.

Writes corpora to corpora/, runs stats and training, dumps report/results.json.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generators.microenglish import MicroEnglish, MEConfig
from generators.conlang import Conlang, CLConfig
from eval.stats import compute, trigram_pp
from eval.tiny_transformer import train as train_model


def write_corpus(path: Path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        for toks, _ in sentences:
            f.write(" ".join(toks) + "\n")


def write_meta(path: Path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        for _, meta in sentences:
            f.write(json.dumps(meta) + "\n")


def stream_from_sentences(sentences):
    out = []
    for toks, _ in sentences:
        out.extend(toks)
    return out


def run_lang(name: str, sentences, out_dir: Path, train_steps: int):
    corp_path = out_dir / f"{name}.txt"
    meta_path = out_dir / f"{name}.meta.jsonl"
    write_corpus(corp_path, sentences)
    write_meta(meta_path, sentences)

    tps = [toks for toks, _ in sentences]
    stats = compute(tps)
    stats["trigram_perplexity"] = trigram_pp(tps)

    print(f"\n--- training tiny transformer on {name} ---")
    stream = stream_from_sentences(sentences)
    tr = train_model(stream, steps=train_steps, block=64, d=128, layers=4,
                     heads=4, lr=3e-4, log_every=max(1, train_steps // 15))
    return {"name": name, "stats": stats,
            "training": {k: v for k, v in tr.items()}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="report/results.json")
    args = ap.parse_args()

    corpora = ROOT / "corpora"
    corpora.mkdir(exist_ok=True)

    print(f"Generating {args.sentences} MicroEnglish sentences...")
    me = MicroEnglish(MEConfig(seed=1, grounded_fraction=0.5, max_clause_depth=2))
    me_sents = me.generate(args.sentences)

    print(f"Generating {args.sentences} Conlang-Regular sentences...")
    cl = Conlang(CLConfig(seed=1, grounded_fraction=0.5, max_clause_depth=2, word_order="SOV"))
    cl_sents = cl.generate(args.sentences)

    # dump dictionary so we can decode samples
    with open(ROOT / "corpora" / "conlang_dictionary.json", "w") as f:
        json.dump(cl.dictionary(), f, indent=2)

    results = {}
    for name, sents in [("microenglish", me_sents), ("conlang", cl_sents)]:
        results[name] = run_lang(name, sents, corpora, args.steps)

    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n\n=================== SUMMARY ===================")
    for name, r in results.items():
        s = r["stats"]
        t = r["training"]
        print(f"\n{name}")
        print(f"  vocab_size        {s['vocab_size']}")
        print(f"  n_tokens          {s['n_tokens']}")
        print(f"  type_token_ratio  {s['type_token_ratio']:.4f}")
        print(f"  avg_sent_len      {s['avg_sent_len']:.2f}")
        print(f"  zipf_slope        {s['zipf_slope']:.3f}")
        print(f"  bigram_pp         {s['bigram_perplexity']:.3f}")
        print(f"  trigram_pp        {s['trigram_perplexity']:.3f}")
        print(f"  transformer final_val_loss  {t['final_val']:.4f}")
        print(f"  transformer val_perplexity  {t['val_perplexity']:.3f}")
        print(f"  transformer n_params        {t['n_params']}")


if __name__ == "__main__":
    main()
