import copy
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import torch
from ranker.features import FeatureBuilder
from ranker.io import read_json, read_jsonl, write_jsonl
from ranker.train_pairs import direct_pair_loss, load_groups

ROOT = Path(__file__).resolve().parents[1] / "data/pairwise_v2"


class ExpandedPairTests(unittest.TestCase):
    def test_full_coverage_and_no_repeated_wording_split_leakage(self):
        scenarios = read_jsonl(ROOT / "prepared/scenarios.jsonl")
        self.assertEqual(len(scenarios), 148)
        self.assertEqual(len({x["symptoms"][0]["symptom_id"] for x in scenarios}), 148)
        word_splits = {}
        for s in scenarios:
            name = " ".join(s["symptoms"][0]["name"].casefold().split())
            self.assertEqual(word_splits.setdefault(name, s["split"]), s["split"])
        counts = Counter(p["scenario_id"] for p in read_jsonl(ROOT / "pair_index.jsonl"))
        self.assertTrue(all(counts[s["scenario_id"]] == 12 for s in scenarios))

    def test_annotation_coverage_and_pair_directions(self):
        labels = read_jsonl(ROOT / "labeled/pair_labels.jsonl")
        self.assertEqual(len(labels), 1776)
        self.assertEqual(len({r["pair_id"] for r in labels}), len(labels))
        by_id = {r["pair_id"]: r for r in labels}
        for pair in read_jsonl(ROOT / "labeled/preferences.jsonl"):
            row = by_id[pair["pair_id"]]
            self.assertIn(row["outcome"], "AB")
            self.assertEqual(pair["preferred_facility_id"], row["facility_a"] if row["outcome"] == "A" else row["facility_b"])
        self.assertEqual(sum(r["adjudicated"] for r in labels), 2)

    def test_direct_loss_matches_logistic_and_tie_pushes_towards_equality(self):
        a = torch.tensor([0.0], requires_grad=True)
        loss = direct_pair_loss(a, torch.zeros(1), torch.ones(1))
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward(); self.assertLess(a.grad.item(), 0)
        a = torch.tensor([2.0], requires_grad=True)
        direct_pair_loss(a, torch.zeros(1), torch.tensor([0.5])).backward()
        self.assertGreater(a.grad.item(), 0)

    def test_loader_excludes_abstentions_and_heldout_labels(self):
        features = FeatureBuilder(ROOT / "prepared")
        groups = load_groups(features, ROOT / "labeled/pair_labels.jsonl")
        self.assertTrue(groups)
        self.assertTrue(all(features.scenarios[sid]["split"] == "train" for sid in groups))
        self.assertTrue(all(r["target_a"] is not None for rows in groups.values() for r in rows))
        row = copy.deepcopy(next(iter(groups.values()))[0]); row["split"] = "test"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"; write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "incorrect split"):
                load_groups(features, path)


if __name__ == "__main__":
    unittest.main()
