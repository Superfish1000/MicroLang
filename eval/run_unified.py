"""Run the unified generator at two presets (english / agglutinative),
train models, save them, and run capability probes.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generators.unified import Grammar, LangConfig
from eval.stats import compute, trigram_pp
from eval.tiny_transformer import train as train_model
from eval.probes import run_all_probes


PRESETS = {
    "english": LangConfig(morphology="english", word_order="SVO",
                           articles=True, case_marking=False,
                           grounded_fraction=0.5, max_clause_depth=2, seed=1),
    "agglutinative": LangConfig(morphology="agglutinative", word_order="SOV",
                                 articles=False, case_marking=True,
                                 grounded_fraction=0.5, max_clause_depth=2,
                                 seed=1, lexicon_seed=1),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    out_dir = ROOT / "corpora"
    out_dir.mkdir(exist_ok=True)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    report_dir = ROOT / "report"
    report_dir.mkdir(exist_ok=True)

    results = {}
    for name, cfg in PRESETS.items():
        print(f"\n\n########## PRESET: {name} ##########")
        g = Grammar(cfg)
        sents = g.generate(args.sentences)
        corp_path = out_dir / f"u_{name}.txt"
        with corp_path.open("w", encoding="utf-8") as f:
            for toks, _ in sents:
                f.write(" ".join(toks) + "\n")
        tps = [toks for toks, _ in sents]
        stats = compute(tps)
        stats["trigram_perplexity"] = trigram_pp(tps)
        print(f"Generated {len(sents)} sentences. vocab={stats['vocab_size']}")

        stream = [t for toks, _ in sents for t in toks]
        model_path = models_dir / f"u_{name}.pt"
        tr = train_model(stream, steps=args.steps, block=64, d=128, layers=4,
                         heads=4, lr=3e-4, log_every=max(1, args.steps // 10),
                         save_path=str(model_path))

        probes = run_all_probes(model_path, cfg, label=name)
        results[name] = {"stats": stats,
                         "training": {k: v for k, v in tr.items() if k != "history"},
                         "probes": probes}

    with (report_dir / "unified_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n================ UNIFIED SUMMARY ================")
    for name, r in results.items():
        s, t, p = r["stats"], r["training"], r["probes"]
        print(f"\n{name}")
        print(f"  vocab={s['vocab_size']}  tokens={s['n_tokens']}  zipf={s['zipf_slope']:.3f}")
        print(f"  trigram_pp={s['trigram_perplexity']:.2f}  transformer val_pp={t['val_perplexity']:.2f}")
        print(f"  probes: held_out_pp={p['held_out_pp']:.2f}"
              f"  structure_ratio={p['structure']['ratio']:.2f}x"
              f"  role_swap_correct={p['role_swap']['correct_preferred_pct']*100:.1f}%")


if __name__ == "__main__":
    main()
