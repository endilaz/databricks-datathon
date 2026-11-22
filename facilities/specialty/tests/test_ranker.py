import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from ranker.features import FeatureBuilder
from ranker.io import read_json, read_jsonl, write_jsonl
from ranker.labeling import validate_labels
from ranker.losses import ranking_loss
from ranker.model import MatchingMLP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "prepared"


class RankingMathTests(unittest.TestCase):
    def test_preferred_pair_gradient(self):
        scores = torch.tensor([0.0, 0.0], requires_grad=True)
        loss = ranking_loss(scores, torch.tensor([3.0, 0.0]), "pairwise")
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(scores.grad[0].item(), 0)
        self.assertGreater(scores.grad[1].item(), 0)

    def test_lambda_weight_matches_independent_two_item_calculation(self):
        loss = ranking_loss(torch.zeros(2), torch.tensor([3.0, 0.0]), "lambda", k=2)
        # Swapping a grade-3 result from rank 1 to 2 changes NDCG by this amount.
        expected_delta = 1 - 1 / math.log2(3)
        self.assertAlmostEqual(loss.item(), expected_delta * math.log(2), places=6)

    def test_unjudged_candidates_are_not_negatives(self):
        a = ranking_loss(torch.tensor([0.4, 0.1]), torch.tensor([3.0, 0.0]))
        b = ranking_loss(torch.tensor([0.4, 0.1, 100.0]), torch.tensor([3.0, 0.0, -1.0]))
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_ties_produce_differentiable_zero(self):
        scores = torch.zeros(3, requires_grad=True)
        loss = ranking_loss(scores, torch.ones(3))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros(3)))

    def test_extreme_scores_are_finite(self):
        loss = ranking_loss(torch.tensor([-10000.0, 10000.0]), torch.tensor([3.0, 0.0]))
        self.assertTrue(torch.isfinite(loss))


class PreparedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = FeatureBuilder(DATA)

    def test_scenarios_preserve_dictionary(self):
        source_path = Path(read_json(DATA / "preparation_report.json")["dictionary_source"])
        source = read_json(PROJECT_ROOT / source_path)
        splits = {}
        for scenario in self.features.scenarios.values():
            self.assertEqual(len(scenario["symptoms"]), 1)
            symptom = scenario["symptoms"][0]
            self.assertIn(symptom["name"], source["symptom_to_specialties"][symptom["category"]])
            self.assertEqual(set(symptom), {"symptom_id", "name", "category"})
            self.assertNotIn(symptom["symptom_id"], splits)
            splits[symptom["symptom_id"]] = scenario["split"]

    def test_model_forward_and_backward_only(self):
        torch.manual_seed(5)
        scenario = next(iter(self.features.scenarios.values()))
        vectors = self.features.group(scenario["scenario_id"], scenario["candidate_ids"])
        model = MatchingMLP(**self.features.dimensions)
        scores = model(*vectors)
        self.assertEqual(scores.shape, (len(scenario["candidate_ids"]),))
        # Arbitrary fixture labels check numerical plumbing, not real clinical fit.
        fixture = torch.tensor([3.0] + [0.0] * (len(scores) - 1))
        ranking_loss(scores, fixture).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_symptom_encoding_is_one_value_per_symptom(self):
        scenario = next(iter(self.features.scenarios.values()))
        hospital = self.features.facilities[scenario["candidate_ids"][0]]
        original = self.features.encode(scenario, hospital)[0]
        fixture = copy.deepcopy(scenario)
        current_id = fixture["symptoms"][0]["symptom_id"]
        other = next(s for s in self.features.symptoms if s["id"] != current_id)
        fixture["symptoms"][0] = {"symptom_id": other["id"], "name": other["name"], "category": other["category"]}
        changed = self.features.encode(fixture, hospital)[0]
        self.assertEqual(original.sum().item(), 1)
        self.assertEqual(changed.sum().item(), 1)
        self.assertEqual((original - changed).abs().sum().item(), 2)

    def test_label_import_and_reject_invented_evidence(self):
        request = read_jsonl(DATA / "label_requests.jsonl")[0]
        response = {"scenario_id": request["scenario_id"], "judgments": []}
        for i, candidate in enumerate(request["candidates"]):
            response["judgments"].append({"candidate_id": candidate["candidate_id"],
                "relevance": 2 if i == 0 else 0, "abstain_reason": None,
                "reason_categories": ["LISTED_SPECIALTY_MATCH" if i == 0 else "NO_LISTED_MATCH"],
                "supporting_specialties": candidate["specialty_claims"][:1] if i == 0 else [],
                "fit_note": "UNIT TEST FIXTURE: shape validation only."})
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            write_jsonl(temp / "fixture.jsonl", [response])
            validate_labels(DATA, temp / "fixture.jsonl", temp / "accepted", "UNIT_TEST_FIXTURE")
            self.assertEqual(len(read_jsonl(temp / "accepted" / "preferences.jsonl")), 7)
            response["judgments"][0]["supporting_specialties"] = ["invented specialty"]
            write_jsonl(temp / "fixture.jsonl", [response])
            with self.assertRaisesRegex(ValueError, "unsupported specialty"):
                validate_labels(DATA, temp / "fixture.jsonl", temp / "rejected", "UNIT_TEST_FIXTURE")


if __name__ == "__main__":
    unittest.main()
