"""Evaluate captured AI preferences; default leaves the test split untouched."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from .features import FeatureBuilder
from .io import read_json, read_jsonl, write_json, write_jsonl
from .model import model_from_checkpoint
from .train_pairs import direct_pair_loss


def preference_credit(gap, target, epsilon=1e-7):
    if target not in (0.0, 1.0):
        raise ValueError("Strict preference target required")
    if abs(gap) <= epsilon:
        return 0.5
    return float((gap > 0) == (target == 1.0))


def evaluate(root, checkpoint_path, splits=("train", "validation"), tie_weight=0.25):
    torch.set_num_threads(1)
    root, checkpoint_path = Path(root), Path(checkpoint_path)
    before_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    features = FeatureBuilder(root / "prepared")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["vocabulary"] != features.vocab:
        raise ValueError("Checkpoint vocabulary does not match the dataset")
    model = model_from_checkpoint(checkpoint)
    model.eval()
    grouped, seen = collections.defaultdict(list), set()
    for row in read_jsonl(root / "labeled/pair_labels.jsonl"):
        if row["pair_id"] in seen:
            raise ValueError("Duplicate pair label")
        seen.add(row["pair_id"])
        sid = row["scenario_id"]
        if row["split"] != features.scenarios[sid]["split"]:
            raise ValueError("Incorrect split")
        if row["split"] in splits:
            grouped[sid].append(row)
    predictions, query_losses = [], collections.defaultdict(list)
    with torch.no_grad():
        for sid, rows in grouped.items():
            scenario = features.scenarios[sid]
            ids = scenario["candidate_ids"]
            index = {fid: i for i, fid in enumerate(ids)}
            vectors = features.group(sid, ids)
            scores = {"mlp": model(*vectors), "cosine": vectors[2][:, 0],
                      "coverage": vectors[2][:, 1], "idf_coverage": vectors[2][:, 2]}
            judged = [r for r in rows if r["target_a"] is not None]
            if judged:
                a = torch.tensor([index[r["facility_a"]] for r in judged])
                b = torch.tensor([index[r["facility_b"]] for r in judged])
                targets = torch.tensor([r["target_a"] for r in judged])
                query_losses[scenario["split"]].append(
                    direct_pair_loss(scores["mlp"][a], scores["mlp"][b], targets, tie_weight).item())
            for row in rows:
                target = row["target_a"]
                if target is None:
                    continue
                a, b = index[row["facility_a"]], index[row["facility_b"]]
                gaps = {name: (value[a]-value[b]).item() for name, value in scores.items()}
                prediction = {"pair_id": row["pair_id"], "scenario_id": sid,
                    "symptom": scenario["symptoms"][0]["name"], "category": scenario["symptoms"][0]["category"],
                    "split": scenario["split"], "outcome": row["outcome"], "target_a": target,
                    "facility_a": ids[a], "facility_b": ids[b],
                    "name_a": features.facilities[ids[a]]["name"],
                    "name_b": features.facilities[ids[b]]["name"],
                    "model_preference_a": torch.sigmoid(torch.tensor(gaps["mlp"])).item()}
                for name, gap in gaps.items():
                    prediction[name+"_score_gap"] = gap
                    prediction[name+"_credit"] = preference_credit(gap, target) if target != 0.5 else None
                predictions.append(prediction)
    report = {"checkpoint_sha256": before_hash,
        "dataset_label_sha256": hashlib.sha256((root / "labeled/pair_labels.jsonl").read_bytes()).hexdigest(),
        "splits_evaluated": list(splits), "test_split_evaluated": "test" in splits,
        "tie_weight_for_loss": tie_weight,
        "metric_definition": "Strict-pair ordering agreement; exact predicted ties receive half credit. Targets that are ties are reported separately.",
        "meaning": "Agreement with provisional AI labels, not clinical accuracy or verified service availability.",
        "model_modified": False, "metrics": {}}
    for split in splits:
        rows = [r for r in predictions if r["split"] == split]
        strict = [r for r in rows if r["target_a"] != 0.5]
        tied = [r for r in rows if r["target_a"] == 0.5]
        metrics = {"judged_pairs": len(rows), "strict_pairs": len(strict), "tied_pairs": len(tied),
            "strict_scenarios": len({r["scenario_id"] for r in strict}),
            "mean_query_weighted_logistic_loss": sum(query_losses[split])/len(query_losses[split]) if query_losses[split] else None,
            "tie_mean_abs_preference_deviation_from_half": sum(abs(r["model_preference_a"]-0.5) for r in tied)/len(tied) if tied else None}
        for name in ("mlp", "cosine", "coverage", "idf_coverage"):
            by_query = collections.defaultdict(list)
            for row in strict:
                by_query[row["scenario_id"]].append(row[name+"_credit"])
            metrics[name] = {
                "pair_agreement": sum(r[name+"_credit"] for r in strict)/len(strict) if strict else None,
                "macro_scenario_agreement": sum(sum(v)/len(v) for v in by_query.values())/len(by_query) if by_query else None,
                "predicted_ties_on_strict_pairs": sum(abs(r[name+"_score_gap"]) <= 1e-7 for r in strict)}
        report["metrics"][split] = metrics
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == before_hash
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/pairwise_v2")
    parser.add_argument("--checkpoint", default="runs/direct_pairs/model.pt")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    parser.add_argument("--out", default="runs/direct_pairs/evaluation")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous evaluations")
    run_path = Path(args.checkpoint).with_name("training_run.json")
    tie_weight = read_json(run_path).get("tie_weight", 0.25) if run_path.exists() else 0.25
    report, predictions = evaluate(args.root, args.checkpoint, args.splits, tie_weight)
    write_json(out / "metrics.json", report)
    write_jsonl(out / "pair_predictions.jsonl", predictions)
    if predictions:
        with (out / "pair_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
