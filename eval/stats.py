"""Distributional and structural statistics for a corpus.

Usage:  python -m eval.stats <corpus.txt> [--name NAME]
"""
from __future__ import annotations
import sys
import math
from collections import Counter


def compute(tokens_per_sentence: list[list[str]]) -> dict:
    all_toks = [t for s in tokens_per_sentence for t in s]
    N = len(all_toks)
    unigrams = Counter(all_toks)
    V = len(unigrams)
    # entropy (nats) and perplexity
    probs = [c / N for c in unigrams.values()]
    H = -sum(p * math.log(p) for p in probs)
    unigram_pp = math.exp(H)

    # Zipf slope (log rank vs log freq) via least squares
    freqs = sorted(unigrams.values(), reverse=True)
    ranks = list(range(1, len(freqs) + 1))
    logr = [math.log(r) for r in ranks]
    logf = [math.log(f) for f in freqs]
    mean_r = sum(logr) / len(logr)
    mean_f = sum(logf) / len(logf)
    num = sum((logr[i] - mean_r) * (logf[i] - mean_f) for i in range(len(logr)))
    den = sum((logr[i] - mean_r) ** 2 for i in range(len(logr)))
    zipf_slope = num / den if den else 0.0

    # sentence length stats
    lens = [len(s) for s in tokens_per_sentence]
    lens_sorted = sorted(lens)
    def pct(p):
        return lens_sorted[min(len(lens_sorted) - 1, int(p * len(lens_sorted)))]
    avg_len = sum(lens) / len(lens)

    # bigram perplexity (add-1 smoothed) on same corpus (in-sample)
    bigrams = Counter()
    context = Counter()
    for s in tokens_per_sentence:
        seq = ["<s>"] + s
        for i in range(len(seq) - 1):
            bigrams[(seq[i], seq[i + 1])] += 1
            context[seq[i]] += 1
    # eval in-sample avg log prob (smoothed)
    log_prob_sum = 0.0
    token_count = 0
    vocab_plus = V + 1  # include <s>
    for s in tokens_per_sentence:
        seq = ["<s>"] + s
        for i in range(len(seq) - 1):
            c = bigrams[(seq[i], seq[i + 1])]
            d = context[seq[i]]
            p = (c + 1) / (d + vocab_plus)
            log_prob_sum += math.log(p)
            token_count += 1
    bigram_pp = math.exp(-log_prob_sum / token_count)

    # branching / structure proxy: type-token ratio
    ttr = V / N

    return {
        "n_sentences": len(tokens_per_sentence),
        "n_tokens": N,
        "vocab_size": V,
        "type_token_ratio": ttr,
        "unigram_entropy_nats": H,
        "unigram_perplexity": unigram_pp,
        "zipf_slope": zipf_slope,
        "bigram_perplexity": bigram_pp,
        "avg_sent_len": avg_len,
        "sent_len_p50": pct(0.5),
        "sent_len_p90": pct(0.9),
        "sent_len_p99": pct(0.99),
    }


def load_corpus(path: str) -> list[list[str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line.split())
    return out


def trigram_pp(tokens_per_sentence: list[list[str]]) -> float:
    """Add-1 smoothed trigram PP computed in-sample (training-set floor)."""
    trig = Counter()
    bg = Counter()
    V = len({t for s in tokens_per_sentence for t in s}) + 2
    for s in tokens_per_sentence:
        seq = ["<s>", "<s>"] + s
        for i in range(len(seq) - 2):
            trig[(seq[i], seq[i + 1], seq[i + 2])] += 1
            bg[(seq[i], seq[i + 1])] += 1
    lp = 0.0
    n = 0
    for s in tokens_per_sentence:
        seq = ["<s>", "<s>"] + s
        for i in range(len(seq) - 2):
            c = trig[(seq[i], seq[i + 1], seq[i + 2])]
            d = bg[(seq[i], seq[i + 1])]
            p = (c + 1) / (d + V)
            lp += math.log(p)
            n += 1
    return math.exp(-lp / n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--name", default="corpus")
    args = ap.parse_args()
    tps = load_corpus(args.corpus)
    s = compute(tps)
    s["trigram_perplexity"] = trigram_pp(tps)
    print(f"=== {args.name} ===")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v}")
