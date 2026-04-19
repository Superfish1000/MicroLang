"""Tiny nanoGPT-style transformer for comparing languages.

Deliberately small (~0.5–1M params) so it trains on CPU in minutes.
Trains on one corpus, returns train/val loss curves.
"""
from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        T = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d=128, heads=4, layers=4, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, heads, block_size) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build_vocab(tokens: list[str]) -> tuple[dict, list]:
    uniq = sorted(set(tokens))
    itos = ["<pad>", "<s>"] + uniq
    stoi = {s: i for i, s in enumerate(itos)}
    return stoi, itos


def encode(stream: list[str], stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi[t] for t in stream], dtype=torch.long)


def train(stream: list[str], steps=2000, block=64, batch=32, d=128, heads=4,
          layers=4, lr=3e-4, val_frac=0.1, device="cpu", seed=0, log_every=100,
          save_path=None):
    torch.manual_seed(seed)
    stoi, itos = build_vocab(stream)
    data = encode(stream, stoi)
    n_val = int(len(data) * val_frac)
    train_data = data[:-n_val]
    val_data = data[-n_val:]

    model = TinyGPT(len(itos), d=d, heads=heads, layers=layers, block_size=block).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def get_batch(src):
        ix = torch.randint(0, len(src) - block - 1, (batch,))
        x = torch.stack([src[i:i + block] for i in ix]).to(device)
        y = torch.stack([src[i + 1:i + block + 1] for i in ix]).to(device)
        return x, y

    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data)
                _, vl = model(vx, vy)
            model.train()
            elapsed = time.time() - t0
            history.append({"step": step, "train_loss": loss.item(),
                            "val_loss": vl.item(), "elapsed": elapsed})
            print(f"step {step:5d}  train {loss.item():.4f}  val {vl.item():.4f}  ({elapsed:.1f}s)")

    if save_path is not None:
        torch.save({
            "state_dict": model.state_dict(),
            "stoi": stoi, "itos": itos,
            "model_cfg": {"vocab_size": len(itos), "d": d, "heads": heads,
                          "layers": layers, "block_size": block},
        }, save_path)

    return {"history": history, "n_params": n_params, "vocab_size": len(itos),
            "final_train": history[-1]["train_loss"], "final_val": history[-1]["val_loss"],
            "val_perplexity": math.exp(history[-1]["val_loss"])}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stream = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            stream.extend(line.strip().split())
    print(f"corpus tokens: {len(stream)}")
    result = train(stream, steps=args.steps, block=args.block, d=args.d,
                   layers=args.layers, heads=args.heads)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
