"""Deterministic hospital subset and exact symptom entries from the user's JSON."""
import argparse
import collections
import csv
import hashlib
import itertools
import math
import random
from pathlib import Path

from .io import read_json, write_json, write_jsonl


def parse_list(value):
    import json
    parsed = json.loads(value or "[]")
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array")
    return sorted({v.strip() for v in parsed if isinstance(v, str) and v.strip()})


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_diverse(rows, count, rng):
    """Round-robin across priority, state and specialty-count strata."""
    buckets = collections.defaultdict(list)
    for row in rows:
        bucket = min(len(row["specialties"]) // 10, 3)
        buckets[(row["review_priority"], row["state"], bucket)].append(row)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(count, len(rows)):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                if len(selected) == min(count, len(rows)):
                    return selected
    return selected


def prepare(args):
    rng = random.Random(args.seed)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --out to preserve labels.")
    dictionary = read_json(args.dictionary)
    raw_vocab = [x["specialty"] for x in dictionary["specialty_union"]]
    vocab = [k for k in raw_vocab if k.strip() and not k.lstrip().startswith(("{", "["))]
    aliases = dictionary.get("specialty_aliases", {})
    if len(vocab) != len(set(vocab)):
        raise ValueError("Duplicate specialty vocabulary")
    symptoms = []
    for category, entries in dictionary["symptom_to_specialties"].items():
        for name, specialties in entries.items():
            symptoms.append({"id": "sym_%03d" % len(symptoms), "category": category,
                             "name": name, "specialties": specialties})
    known = set(vocab)
    if any(set(s["specialties"]) - known for s in symptoms):
        raise ValueError("Dictionary maps symptoms to unknown specialties")
    csv.field_size_limit(100_000_000)
    rows, seen_ids, seen_clusters = [], set(), set()
    excluded = collections.Counter()
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["facilityTypeId"] != "hospital":
                excluded["not_hospital"] += 1
                continue
            uid, cluster = row["unique_id"], row.get("cluster_id", "")
            if not uid or uid in seen_ids or (cluster and cluster in seen_clusters):
                excluded["duplicate_or_missing_identity"] += 1
                continue
            try:
                specialties = sorted({aliases.get(k, k) for k in parse_list(row["specialties"])})
                specialties = [k for k in specialties if k.strip() and not k.lstrip().startswith(("{", "["))]
                if not specialties:
                    raise ValueError("Empty specialties")
            except (ValueError, TypeError):
                excluded["invalid_or_empty_specialties"] += 1
                continue
            if set(specialties) - known:
                excluded["specialties_outside_dictionary"] += 1
                continue
            seen_ids.add(uid)
            if cluster:
                seen_clusters.add(cluster)
            flags = list(filter(None, row.get("quality_flags", "").split(";")))
            try:
                source_urls = parse_list(row.get("source_urls", "[]"))
            except (ValueError, TypeError):
                source_urls = []
                flags.append("invalid_source_urls_json")
            rows.append({"facility_id": uid, "cluster_id": cluster,
                         "name": row.get("normalized_name") or row["name"],
                         "state": row.get("normalized_state", ""),
                         "specialties": specialties, "quality_flags": flags,
                         "specialty_quality_flags": [f for f in flags if f in (
                             "repetitive_specialties_list", "duplicate_id", "repetitive_capability_list")],
                         "review_priority": row.get("review_priority", "unknown"),
                         "source_row_number": row.get("source_row_number", ""),
                         "care_level": None, "claim_status": "unverified",
                         "source_urls": source_urls})
    if len(rows) < args.facilities:
        raise ValueError("Not enough eligible hospital records")
    frequencies = collections.Counter(k for row in rows for k in row["specialties"])
    # Sampling uses dictionary relevance only to obtain a mix of candidates, never labels.
    selected = select_diverse(rows, args.facilities, rng)
    selected_by_id = {r["facility_id"]: r for r in selected}
    categories = sorted({s["category"] for s in symptoms})
    rng.shuffle(categories)
    anchors = []
    for category in itertools.cycle(categories):
        candidates = [s for s in symptoms if s["category"] == category and s not in anchors]
        if candidates:
            anchors.append(rng.choice(candidates))
        if len(anchors) == args.scenarios:
            break
        if len(anchors) == len(symptoms):
            raise ValueError("Too many scenarios requested")
    families = [[s] for s in anchors]
    # Split connected symptom families together. Exact symptoms cannot leak across splits.
    groups = []
    for idx, family in enumerate(families):
        ids = {s["id"] for s in family}
        overlapping = [g for g in groups if ids & g["symptoms"]]
        merged = {"families": {idx}, "symptoms": set(ids)}
        for group in overlapping:
            merged["families"].update(group["families"])
            merged["symptoms"].update(group["symptoms"])
            groups.remove(group)
        groups.append(merged)
    rng.shuffle(groups)
    if len(groups) < 3:
        raise ValueError("Need at least three disjoint symptom groups for a held-out split")
    family_split = {}
    held_out_count = max(1, int(round(len(groups) * 0.15)))
    for i, group in enumerate(groups):
        split = "test" if i < held_out_count else "validation" if i < 2 * held_out_count else "train"
        for family_idx in group["families"]:
            family_split[family_idx] = split
    scenarios, requests, examples = [], [], []
    for family_idx, family in enumerate(families):
        needed = {k for s in family for k in s["specialties"]}
        # Dictionary overlap selects a mix of candidates; it never supplies labels.
        ranked = sorted(selected, key=lambda r: (-len(needed & set(r["specialties"])), r["facility_id"]))
        top = ranked[:max(args.candidates, len(ranked) // 3)]
        if getattr(args, "candidate_mode", "mixed") == "contrastive":
            n = args.candidates // 3
            chosen = rng.sample(top, min(n, len(top)))
            bottom = sorted([r for r in selected if r not in chosen],
                            key=lambda r: (len(needed & set(r["specialties"])), len(r["specialties"]), r["facility_id"]))
            bottom = bottom[:max(n, len(selected) // 5)]
            chosen += rng.sample(bottom, n)
        else:
            chosen = rng.sample(top, min(args.candidates // 2, len(top)))
        rest = [r for r in selected if r not in chosen]
        chosen += rng.sample(rest, args.candidates - len(chosen))
        rng.shuffle(chosen)  # Candidate order must not reveal the sampling heuristic.
        sid = "scenario_%03d" % len(scenarios)
        scenario = {"scenario_id": sid, "family_id": "family_%02d" % family_idx,
                    "split": family_split[family_idx], "origin": "exact_dictionary_entry",
                    "clinical_review_status": "not_reviewed",
                    "symptoms": [{"symptom_id": s["id"], "name": s["name"],
                                  "category": s["category"]} for s in family],
                    "candidate_ids": [r["facility_id"] for r in chosen]}
        scenarios.append(scenario)
        # Blind aliases prevent reputation/name cues. URLs stay in the audit subset.
        candidates = [{"candidate_id": "candidate_%02d" % i,
                       "specialty_claims": r["specialties"],
                       "claim_status": r["claim_status"],
                       "quality_flags": r["specialty_quality_flags"], "care_level": None}
                      for i, r in enumerate(chosen)]
        payload = {"scenario_id": sid, "patient": {"symptoms": scenario["symptoms"]},
                   "candidates": candidates}
        requests.append(payload)
        for alias, facility in zip(candidates, chosen):
            examples.append({"scenario_id": sid, "candidate_id": alias["candidate_id"],
                             "facility_id": facility["facility_id"], "split": scenario["split"],
                             "label": None, "label_status": "awaiting_ai_label"})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "vocabulary.json", {"specialties": vocab, "symptoms": symptoms})
    # Frequency is counted once per eligible facility, never repeated array entries.
    write_json(out / "specialty_statistics.json", {"eligible_facilities": len(rows),
        "facility_count": dict(frequencies),
        "idf": {k: min(4.0, math.log(1 + len(rows) / (frequencies[k] + 1))) for k in vocab},
        "meaning": "Listing distinctiveness, not disease rarity or clinical importance."})
    write_jsonl(out / "facilities.jsonl", selected)
    fields = ["facility_id", "name", "state", "specialties", "quality_flags", "review_priority", "claim_status", "care_level", "source_row_number"]
    import json
    with (out / "facilities_subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in fields})
    write_jsonl(out / "scenarios.jsonl", scenarios)
    write_jsonl(out / "label_requests.jsonl", requests)
    write_jsonl(out / "labeling_subset.jsonl", examples)
    write_jsonl(out / "label_template.jsonl", [{"scenario_id": r["scenario_id"],
        "judgments": [{"candidate_id": c["candidate_id"], "relevance": None,
            "abstain_reason": "not_labeled", "reason_categories": [],
            "supporting_specialties": [], "fit_note": ""} for c in r["candidates"]]}
        for r in requests])
    report = {"seed": args.seed, "source_csv": str(Path(args.csv).resolve()),
        "source_csv_sha256": fingerprint(args.csv),
        "dictionary_source": str(Path(args.dictionary).resolve()),
        "dictionary_sha256": fingerprint(args.dictionary),
        "dropped_structurally_invalid_specialty_tokens": [k for k in raw_vocab if k not in vocab],
        "excluded": dict(excluded), "eligible_hospitals": len(rows),
        "subset_hospitals": len(selected_by_id), "scenario_families": len(families),
        "scenarios": len(scenarios), "judgments_requested": len(examples),
        "specialty_dimensions": len(vocab), "symptom_dimensions": len(symptoms),
        "scenario_split_counts": dict(collections.Counter(s["split"] for s in scenarios)),
        "subset_priority_counts": dict(collections.Counter(r["review_priority"] for r in selected)),
        "candidate_sampling": getattr(args, "candidate_mode", "mixed"),
        "status": "unlabeled; no model training or usefulness evaluation performed",
        "limits": ["All service claims remain unverified.", "No care-level labels invented.",
                   "Only supplied specialty aliases applied; no additional clinical merging.",
                   "Scenarios are exact single-symptom dictionary entries; no invented combinations.",
                   "No location, pressure index, patient outcomes or clinical triage training.",
                   "Facility identities may recur across splits; symptom IDs do not.",
                   "The subset is stratified for development, not nationally representative."]}
    write_json(out / "preparation_report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--out", default="data/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--facilities", type=int, default=120)
    parser.add_argument("--scenarios", type=int, default=28, help="Sample exact dictionary entries; add no symptom combinations")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--candidate-mode", choices=["mixed", "contrastive"], default="mixed")
    args = parser.parse_args()
    if args.facilities < args.candidates or args.candidates < 2 or args.scenarios < 3:
        parser.error("Need facilities >= candidates >= 2, and scenarios >= 3")
    prepare(args)


if __name__ == "__main__":
    main()
