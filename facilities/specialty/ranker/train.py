"""Train only on the training split; never evaluate held-out scenarios here."""
import argparse
import collections
import random
from pathlib import Path

import torch
from .features import FeatureBuilder
from .io import read_jsonl, write_json
from .losses import ranking_loss
from .model import MatchingMLP


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    features = FeatureBuilder(args.data)
    groups = collections.defaultdict(dict)
    for row in read_jsonl(args.labels):
        sid, fid = row["scenario_id"], row["facility_id"]
        if sid not in features.scenarios or fid not in features.scenarios[sid]["candidate_ids"]:
            raise ValueError("Unknown scenario/candidate in labels")
        actual_split = features.scenarios[sid]["split"]
        if row["split"] != actual_split:
            raise ValueError("Label split differs from prepared scenario split")
        relevance = row["relevance"]
        if relevance is not None and (type(relevance) is not int or relevance not in range(4)):
            raise ValueError("Expected integer relevance in 0..3 or null")
        if fid in groups[sid]:
            raise ValueError("Duplicate label for scenario/facility")
        groups[sid][fid] = row
    training = []
    for sid, rows in groups.items():
        if features.scenarios[sid]["split"] != "train":
            continue
        judged = [r for r in rows.values() if r["relevance"] is not None]
        if len({r["relevance"] for r in judged}) < 2:
            continue
        facility_ids = [r["facility_id"] for r in judged]
        vectors = features.group(sid, facility_ids)
        y = torch.tensor([r["relevance"] for r in judged], dtype=torch.float32)
        training.append((sid, vectors, y))
    if not training:
        raise ValueError("No informative labeled training groups. Import real teacher responses first.")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty --out to preserve existing checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    model = MatchingMLP(**features.dimensions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    history = []
    for epoch in range(args.epochs):
        random.shuffle(training)
        total = 0.0
        for _, vectors, y in training:
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model(*vectors), y, mode=args.loss, k=args.k)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
        row = {"epoch": epoch + 1, "training_loss": total / len(training)}
        history.append(row)
        print("epoch=%d training_loss=%.6f" % (row["epoch"], row["training_loss"]))
    torch.save({"model_state": model.state_dict(), "config": model.config,
                "vocabulary": features.vocab, "seed": args.seed,
                "loss": args.loss, "k": args.k}, output / "model.pt")
    write_json(output / "training_run.json", {"seed": args.seed, "loss": args.loss, "k": args.k,
        "training_groups": len(training), "parameters": sum(p.numel() for p in model.parameters()),
        "history": history, "status": "Training only; no usefulness or clinical evaluation performed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--labels", default="data/labeled/labels.jsonl")
    parser.add_argument("--out", default="runs/first")
    parser.add_argument("--loss", choices=["pairwise", "lambda"], default="lambda")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.k < 1 or args.lr <= 0:
        parser.error("epochs, k and lr must be positive")
    train(args)


if __name__ == "__main__":
    main()
