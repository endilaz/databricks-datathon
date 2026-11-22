"""Training entry point for direct preferences; no NDCG grades are invented."""
import argparse
import collections
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from .features import FeatureBuilder
from .io import read_jsonl, write_json
from .model import MatchingMLP


def direct_pair_loss(a, b, target_a, tie_weight=0.25):
    """Logistic preference loss; ties target a zero score difference, at lower weight."""
    if a.shape != b.shape or a.shape != target_a.shape:
        raise ValueError("Expected equal score/target shapes")
    if not 0 <= tie_weight <= 1 or not torch.isin(target_a, target_a.new_tensor([0, 0.5, 1])).all():
        raise ValueError("Targets must be 0, .5, 1 and tie_weight must be in [0,1]")
    weight = torch.where(target_a == 0.5, tie_weight, 1.0)
    return (F.binary_cross_entropy_with_logits(a-b, target_a, reduction="none") * weight).sum() / weight.sum().clamp_min(1)


def load_groups(features, path):
    groups, seen = collections.defaultdict(list), set()
    for row in read_jsonl(path):
        sid = row["scenario_id"]
        if row["pair_id"] in seen:
            raise ValueError("Duplicate pair label")
        seen.add(row["pair_id"])
        if sid not in features.scenarios or row["split"] != features.scenarios[sid]["split"]:
            raise ValueError("Unknown scenario or incorrect split")
        if not {row["facility_a"], row["facility_b"]} <= set(features.scenarios[sid]["candidate_ids"]):
            raise ValueError("Unknown candidate in pair")
        if row["facility_a"] == row["facility_b"]:
            raise ValueError("Cannot compare a facility with itself")
        expected = {"A": 1.0, "B": 0.0, "T": 0.5, "U": None}[row["outcome"]]
        if row["target_a"] != expected:
            raise ValueError("Target differs from captured outcome")
        if row["split"] == "train" and expected is not None:
            groups[sid].append(row)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/pairwise_v2")
    parser.add_argument("--out", default="runs/direct_pairs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tie-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or not 0 <= args.tie_weight <= 1:
        parser.error("Invalid epochs/lr/tie-weight")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(1)
    root = Path(args.root)
    features = FeatureBuilder(root / "prepared")
    groups = load_groups(features, root / "labeled/pair_labels.jsonl")
    if not groups:
        raise ValueError("No labeled training groups")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    model = MatchingMLP(**features.dimensions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    cache = []
    for sid, rows in groups.items():
        ids = features.scenarios[sid]["candidate_ids"]
        index = {fid: i for i, fid in enumerate(ids)}
        cache.append((features.group(sid, ids),
            torch.tensor([index[r["facility_a"]] for r in rows]),
            torch.tensor([index[r["facility_b"]] for r in rows]),
            torch.tensor([r["target_a"] for r in rows])))
    history = []
    for epoch in range(args.epochs):
        random.shuffle(cache); total = 0.0
        for vectors, a, b, targets in cache:
            optimizer.zero_grad(set_to_none=True)
            scores = model(*vectors)
            loss = direct_pair_loss(scores[a], scores[b], targets, args.tie_weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += loss.item()
        history.append({"epoch": epoch+1, "training_loss": total/len(cache)})
        print(history[-1])
    torch.save({"model_state": model.state_dict(), "config": model.config,
                "vocabulary": features.vocab, "loss": "direct_pair_logistic", "seed": args.seed}, out / "model.pt")
    write_json(out / "training_run.json", {"training_groups": len(cache), "history": history,
        "tie_weight": args.tie_weight, "model_evaluation": "not_run", "clinical_validation": "not_run"})


if __name__ == "__main__":
    main()
