import torch
from .io import read_json, read_jsonl


class FeatureBuilder:
    """Same encoding for training and inference. Labels never enter features."""
    def __init__(self, data_dir):
        from pathlib import Path
        self.data_dir = Path(data_dir)
        self.vocab = read_json(self.data_dir / "vocabulary.json")
        self.specialties = self.vocab["specialties"]
        self.symptoms = self.vocab["symptoms"]
        self.k_index = {k: i for i, k in enumerate(self.specialties)}
        self.s_index = {s["id"]: i for i, s in enumerate(self.symptoms)}
        self.facilities = {r["facility_id"]: r for r in read_jsonl(self.data_dir / "facilities.jsonl")}
        self.scenarios = {r["scenario_id"]: r for r in read_jsonl(self.data_dir / "scenarios.jsonl")}
        stats = read_json(self.data_dir / "specialty_statistics.json")
        self.idf = torch.tensor([stats["idf"][k] for k in self.specialties])

    @property
    def dimensions(self):
        return {"patient_dim": len(self.symptoms), "hospital_dim": len(self.specialties),
                "match_dim": 5}

    def encode(self, scenario, facility):
        patient = torch.zeros(len(self.symptoms))
        needed = torch.zeros(len(self.specialties))
        for item in scenario["symptoms"]:
            idx = self.s_index[item["symptom_id"]]
            patient[idx] = 1
            for k in self.symptoms[idx]["specialties"]:
                needed[self.k_index[k]] = 1
        hospital = torch.zeros(len(self.specialties))
        for k in facility["specialties"]:
            hospital[self.k_index[k]] = 1
        overlap = needed * hospital
        cosine = overlap.sum() / (needed.norm() * hospital.norm()).clamp_min(1e-8)
        coverage = overlap.sum() / needed.sum().clamp_min(1)
        weighted = (overlap * self.idf).sum() / (needed * self.idf).sum().clamp_min(1e-8)
        # Only expose observable flags, never labels, names, reputation or source IDs.
        match = torch.stack([cosine, coverage, weighted,
                             torch.tensor(min(len(facility["specialty_quality_flags"]) / 3.0, 1.0)),
                             torch.tensor(float(bool(facility["specialty_quality_flags"])))])
        return patient, hospital, match

    def group(self, scenario_id, facility_ids):
        vectors = [self.encode(self.scenarios[scenario_id], self.facilities[fid]) for fid in facility_ids]
        return tuple(torch.stack([v[i] for v in vectors]) for i in range(3))
