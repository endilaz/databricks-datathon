"""Rank a prepared scenario with a trained checkpoint; no evaluation metrics."""
import argparse
import torch
from .features import FeatureBuilder
from .io import write_json
from .model import model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="runs/prediction.json")
    args = parser.parse_args()
    features = FeatureBuilder(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["vocabulary"] != features.vocab:
        raise ValueError("Checkpoint vocabulary differs from prepared data")
    model = model_from_checkpoint(checkpoint)
    model.eval()
    scenario = features.scenarios[args.scenario]
    ids = scenario["candidate_ids"]
    with torch.no_grad():
        vectors = features.group(args.scenario, ids)
        components = model.components(*vectors) if hasattr(model, "components") else None
        scores = (components["score"] if components is not None else model(*vectors)).tolist()
    needed = {k for symptom in scenario["symptoms"]
              for k in features.symptoms[features.s_index[symptom["symptom_id"]]]["specialties"]}
    results = []
    for i, (fid, score) in enumerate(zip(ids, scores)):
        facility = features.facilities[fid]
        matched = sorted(needed & set(facility["specialties"]))
        results.append({"facility_id": fid, "name": facility["name"], "score": score,
                        "care_level": None, "claim_status": "unverified",
                        "listed_specialty_matches": matched,
                        "reason_categories": ["LISTED_SPECIALTY_MATCH" if matched else "NO_LISTED_MATCH"]})
        if components is not None:
            results[-1]["score_components"] = {
                "weighted_coverage": components["weighted_coverage"][i].item(),
                "learned_correction": components["learned_correction"][i].item(),
                "positive_scale": components["positive_scale"].item()}
    results.sort(key=lambda r: r["score"], reverse=True)
    write_json(args.out, {"scenario_id": args.scenario,
        "meaning": "Provisional record-based fit, not a probability or validated clinical referral.",
        "reason_note": "Categories describe dictionary overlap, not a causal explanation of MLP scores.",
        "results": results})


if __name__ == "__main__":
    main()
