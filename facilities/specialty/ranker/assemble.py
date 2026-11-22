"""Expand explicit AI-authored decisions into teacher responses, without inferring labels.

Each grade and supporting specialty is authored in the annotation batches. This
command performs serialization and coverage checks, never clinical classification.
"""
import argparse
import hashlib
from pathlib import Path

from .io import read_json, read_jsonl, write_json, write_jsonl
from .labeling import TEACHER_INSTRUCTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--batches", default="data/annotation_batches")
    parser.add_argument("--out", default="data/teacher_responses.jsonl")
    args = parser.parse_args()
    requests = {r["scenario_id"]: r for r in read_jsonl(Path(args.data) / "label_requests.jsonl")}
    responses, seen = [], set()
    reasons = {0: "NO_LISTED_MATCH", 1: "INDIRECT_MATCH", 2: "LISTED_SPECIALTY_MATCH",
               3: "TARGETED_SPECIALTY_MATCH", None: "INSUFFICIENT_EVIDENCE"}
    batch_paths = sorted(Path(args.batches).glob("batch_*.json"))
    for batch_path in batch_paths:
        for annotation in read_json(batch_path):
            sid = annotation["scenario_id"]
            if sid not in requests or sid in seen:
                raise ValueError("Unknown/duplicate scenario annotation: " + sid)
            seen.add(sid)
            candidates = requests[sid]["candidates"]
            if len(annotation["decisions"]) != len(candidates):
                raise ValueError("Candidate decision count mismatch: " + sid)
            judgments = []
            for candidate, decision in zip(candidates, annotation["decisions"]):
                grade, support, note_key = decision
                note = annotation["notes"][note_key]
                categories = [reasons[grade]]
                if grade is None and "duplicate_id" in candidate["quality_flags"]:
                    categories.append("RECORD_CONFLICT")
                judgments.append({"candidate_id": candidate["candidate_id"], "relevance": grade,
                    "abstain_reason": note if grade is None else None,
                    "reason_categories": categories, "supporting_specialties": support,
                    "fit_note": note})
            responses.append({"scenario_id": sid, "judgments": judgments})
    if seen != set(requests):
        raise ValueError("Some requested scenarios lack annotations")
    responses.sort(key=lambda r: r["scenario_id"])
    write_jsonl(args.out, responses)
    write_json(Path(args.out).with_name("teacher_run.json"), {
        "teacher_id": "gpt-6.1-sol", "generation_method": "interactive_annotation_by_Codex_assistant",
        "external_api_calls": 0, "date": "2026-11-22", "task": "conditional_listed_specialty_fit",
        "input_sha256": hashlib.sha256((Path(args.data) / "label_requests.jsonl").read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(TEACHER_INSTRUCTIONS.encode()).hexdigest(),
        "batch_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in batch_paths},
        "scenarios": len(responses), "judgments": sum(len(r["judgments"]) for r in responses),
        "independently_reviewed": False, "facility_claims_verified": False,
        "note": "Explicit AI judgments, not dictionary-generated labels. Fixed captured outputs; rerunning assembly does not rerun an AI teacher."})
    print("Assembled %d explicitly annotated scenarios; no labels inferred by this script." % len(responses))


if __name__ == "__main__":
    main()
