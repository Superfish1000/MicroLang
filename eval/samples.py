"""Pull aligned sample sentences and a decoded Conlang view for the report."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def decode_conlang(tokens: list[str], dictionary: dict) -> list[str]:
    inv = {v: k for k, v in dictionary.items()}
    gloss = []
    suffixes = [("pa", "PL"), ("ki", "ACC"), ("li", "DAT"), ("ne", "LOC"),
                ("na", "DEF"), ("du", "PST"), ("mo", "FUT"), ("si", "PROG"),
                ("ta", "PL_AGR")]
    particles = {"ke": "SUB", "vo": "BE", "mi": "NOT", "sa": "AND",
                 "ra": "BECAUSE", "zu": "WHILE", "to": "AFTER",
                 "fi": "BEFORE", "pe": "TO"}
    for t in tokens:
        if t == ".":
            gloss.append(".")
            continue
        if t in particles:
            gloss.append(particles[t])
            continue
        # strip suffixes greedily from the right
        tags = []
        cur = t
        changed = True
        while changed:
            changed = False
            for suf, lab in suffixes:
                if len(cur) > len(suf) and cur.endswith(suf):
                    cur = cur[:-len(suf)]
                    tags.append(lab)
                    changed = True
                    break
        root_word = inv.get(cur, f"?{cur}")
        if tags:
            gloss.append(f"{root_word}.{'.'.join(reversed(tags))}")
        else:
            gloss.append(root_word)
    return gloss


def main():
    me_corp = (ROOT / "corpora" / "microenglish.txt").read_text(encoding="utf-8").splitlines()
    me_meta = [json.loads(l) for l in (ROOT / "corpora" / "microenglish.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    cl_corp = (ROOT / "corpora" / "conlang.txt").read_text(encoding="utf-8").splitlines()
    cl_meta = [json.loads(l) for l in (ROOT / "corpora" / "conlang.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    cl_dict = json.loads((ROOT / "corpora" / "conlang_dictionary.json").read_text(encoding="utf-8"))

    print("## Side-by-side samples\n")
    # Pick examples of each template type
    seen_templates = set()
    samples = []
    for i, meta in enumerate(me_meta):
        t = meta.get("template")
        if t in seen_templates:
            continue
        seen_templates.add(t)
        samples.append(i)
        if len(seen_templates) == 5:
            break
    for i in samples:
        print(f"### {me_meta[i].get('template')}")
        print(f"  MicroEnglish: {me_corp[i]}")
        # find a matching Conlang sentence of same template
        for j, m in enumerate(cl_meta):
            if m.get("template") == me_meta[i].get("template"):
                print(f"  Conlang:      {cl_corp[j]}")
                print(f"  decoded:      {' '.join(decode_conlang(cl_corp[j].split(), cl_dict))}")
                break
        print()


if __name__ == "__main__":
    main()
