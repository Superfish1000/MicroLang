"""Re-run probes on already-saved models (no retraining)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.probes import run_all_probes
from eval.run_unified import PRESETS


def main():
    models_dir = ROOT / "models"
    out = {}
    for name, cfg in PRESETS.items():
        model_path = models_dir / f"u_{name}.pt"
        if not model_path.exists():
            print(f"skip {name}: no model at {model_path}")
            continue
        out[name] = run_all_probes(model_path, cfg, label=name)

    with (ROOT / "report" / "probes_fixed.json").open("w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n\n========== PROBES (fixed) ==========")
    for name, p in out.items():
        print(f"\n{name}")
        print(f"  held_out_pp       {p['held_out_pp']:.2f}")
        print(f"  structure_ratio   {p['structure']['ratio']:.2f}x")
        print(f"  role_swap         {p['role_swap']['correct_preferred_pct']*100:.1f}%")


if __name__ == "__main__":
    main()
