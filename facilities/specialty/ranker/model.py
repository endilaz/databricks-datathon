import torch
from torch import nn
from torch.nn import functional as F
import math


class MatchingMLP(nn.Module):
    """Patient and facility projections + a joint matching MLP, trained end to end."""
    def __init__(self, patient_dim, hospital_dim, match_dim=5, embedding_dim=32):
        super().__init__()
        self.config = dict(patient_dim=patient_dim, hospital_dim=hospital_dim,
                           match_dim=match_dim, embedding_dim=embedding_dim)
        self.patient_encoder = nn.Sequential(nn.Linear(patient_dim, 64), nn.ReLU(),
                                             nn.Linear(64, embedding_dim), nn.ReLU())
        self.hospital_encoder = nn.Sequential(nn.Linear(hospital_dim, embedding_dim), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(4 * embedding_dim + match_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, patient, hospital, match):
        p = self.patient_encoder(patient)
        h = self.hospital_encoder(hospital)
        combined = torch.cat([p, h, p * h, torch.abs(p - h), match], dim=-1)
        return self.scorer(combined).squeeze(-1)


class FeatureMLP(nn.Module):
    """A small control: can five existing matching features explain the labels?"""
    def __init__(self, match_dim=5, hidden_dim=16):
        super().__init__()
        self.config = dict(match_dim=match_dim, hidden_dim=hidden_dim)
        self.scorer = nn.Sequential(nn.Linear(match_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, patient, hospital, match):
        return self.scorer(match).squeeze(-1)


class SemanticResidualMLP(nn.Module):
    """Learn a correction to coverage using shared, training-observed specialties.

    'Semantic' here means the supplied symptom-to-specialty map, not a pretrained
    language model or additional clinical knowledge. There is no symptom-ID embedding.
    """
    def __init__(self, active_indices, symptom_need_map, embedding_dim=16,
                 hidden_dim=32, residual_bound=0.5, dropout=0.1):
        super().__init__()
        self.config = dict(active_indices=active_indices, symptom_need_map=symptom_need_map,
                           embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           residual_bound=residual_bound, dropout=dropout)
        self.register_buffer("active_indices", torch.tensor(active_indices, dtype=torch.long))
        self.register_buffer("symptom_need_map", torch.tensor(symptom_need_map, dtype=torch.float32))
        self.specialty_embeddings = nn.Parameter(torch.randn(len(active_indices), embedding_dim) * 0.02)
        self.correction = nn.Sequential(nn.Linear(5*embedding_dim+5, hidden_dim), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(8.0))))
        self.residual_bound = residual_bound

    def components(self, patient, hospital, match):
        need = (patient @ self.symptom_need_map).clamp(0, 1)
        available = hospital[:, self.active_indices]
        overlap = need * available
        q = (need @ self.specialty_embeddings) / need.sum(-1, keepdim=True).clamp_min(1)
        h = (available @ self.specialty_embeddings) / available.sum(-1, keepdim=True).clamp_min(1).sqrt()
        m = (overlap @ self.specialty_embeddings) / overlap.sum(-1, keepdim=True).clamp_min(1)
        joint = torch.cat([q, h, m, q*h, torch.abs(q-h), match], dim=-1)
        correction = self.residual_bound * torch.tanh(self.correction(joint).squeeze(-1))
        baseline = match[:, 2]
        scale = F.softplus(self.log_scale)
        return {"weighted_coverage": baseline, "learned_correction": correction,
                "positive_scale": scale, "score": scale*(baseline+correction)}

    def forward(self, patient, hospital, match):
        return self.components(patient, hospital, match)["score"]


def make_semantic_config(features):
    # Feature support comes ONLY from training candidate hospitals, not held-out labels.
    train_ids = {fid for s in features.scenarios.values() if s["split"] == "train" for fid in s["candidate_ids"]}
    observed = {k for fid in train_ids for k in features.facilities[fid]["specialties"]}
    active = [i for i, k in enumerate(features.specialties) if k in observed]
    mapping = [[float(features.specialties[i] in s["specialties"]) for i in active] for s in features.symptoms]
    return dict(active_indices=active, symptom_need_map=mapping)


def model_from_checkpoint(checkpoint):
    architecture = checkpoint.get("architecture", "original")
    classes = {"original": MatchingMLP, "features_only": FeatureMLP,
               "semantic_residual": SemanticResidualMLP}
    if architecture not in classes:
        raise ValueError("Unknown architecture: " + architecture)
    model = classes[architecture](**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    return model
