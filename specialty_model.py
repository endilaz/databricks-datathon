"""Learned symptom-to-specialty fit: numpy inference of the trained specialty ranker.

The weights come from `feat/facilities-specialty-model` (see export_specialty_model.py).
Fit = IDF-weighted coverage of the symptom's specialties plus a bounded learned
correction. It reflects listed specialties only, trained on provisional AI labels.
"""
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / 'data' / 'specialty_model.json.gz'
# The same record flags the trainer exposed to the model (ranker/prepare.py).
SPECIALTY_FLAGS = ('repetitive_specialties_list', 'duplicate_id', 'repetitive_capability_list')
_erf = np.vectorize(math.erf, otypes=[float])


def _gelu(x):
    """Exact GELU, matching torch.nn.GELU's default."""
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


class SpecialtyModel:
    def __init__(self, bundle):
        self.source = bundle['source']
        self.specialties = bundle['specialties']
        self.index = {name: i for i, name in enumerate(self.specialties)}
        self.aliases = {}
        for raw, name in bundle['aliases'].items():
            if name in self.index:
                self.aliases.setdefault(raw, name)
                self.aliases.setdefault(raw.strip(), name)
        self.idf = np.asarray(bundle['idf'])
        self.symptoms = {(c, n): i for i, (c, n, _) in enumerate(bundle['symptoms'])}
        self.symptom_needs = [np.array([self.index[k] for k in s], dtype=int) for _, _, s in bundle['symptoms']]
        self.active = np.asarray(bundle['active_indices'], dtype=int)
        self.active_position = {int(k): j for j, k in enumerate(self.active)}
        self.need_positions = [np.asarray(p, dtype=int) for p in bundle['need_positions']]
        w = bundle['weights']
        self.embeddings = np.asarray(w['specialty_embeddings'])
        self.w1, self.b1 = np.asarray(w['w1']), np.asarray(w['b1'])
        self.w2, self.b2 = np.asarray(w['w2'])[0], float(w['b2'][0])
        self.scale = math.log1p(math.exp(w['log_scale']))
        self.residual_bound = bundle['residual_bound']
        self.references = bundle.get('references', [])
        self._facility_cache = {}

    def canonical(self, name):
        """Map an app specialty code to the model's vocabulary name, or None."""
        if name in self.index:
            return name
        return self.aliases.get(name) or self.aliases.get(name.strip())

    def _facility(self, specialties, flag_count):
        key = (tuple(specialties), flag_count)
        cached = self._facility_cache.get(key)
        if cached is None:
            indices = sorted({self.index[c] for c in map(self.canonical, specialties) if c})
            active = np.array([self.active_position[i] for i in indices if i in self.active_position], dtype=int)
            h = self.embeddings[active].sum(0) / math.sqrt(max(len(active), 1))
            flags = [min(flag_count / 3.0, 1.0), float(flag_count > 0)]
            cached = (frozenset(indices), active, h, flags)
            self._facility_cache[key] = cached
        return cached

    def fits(self, specialties_list, flag_counts, symptom_key):
        """Return (weighted_coverage, learned_correction) arrays for one symptom."""
        s = self.symptoms[symptom_key]
        need = self.symptom_needs[s]
        need_set = set(need.tolist())
        need_idf = self.idf[need].sum()
        need_active = self.need_positions[s]
        need_active_set = set(need_active.tolist())
        q = self.embeddings[need_active].sum(0) / max(len(need_active), 1)
        rows, coverage_values = [], []
        for specialties, flag_count in zip(specialties_list, flag_counts):
            offered, active, h, flags = self._facility(specialties, flag_count)
            overlap = [i for i in need_set if i in offered]
            n_offered = len(offered)
            cosine = len(overlap) / (math.sqrt(len(need_set)) * math.sqrt(n_offered)) if n_offered else 0.0
            coverage = len(overlap) / max(len(need_set), 1)
            weighted = float(self.idf[overlap].sum() / need_idf) if need_idf > 0 else 0.0
            shared = [j for j in active.tolist() if j in need_active_set]
            m = self.embeddings[shared].sum(0) / max(len(shared), 1)
            rows.append(np.concatenate([q, h, m, q * h, np.abs(q - h), [cosine, coverage, weighted, *flags]]))
            coverage_values.append(weighted)
        if not rows:
            return np.zeros(0), np.zeros(0)
        # macOS Accelerate can raise spurious matmul FP warnings; results are unaffected.
        with np.errstate(all='ignore'):
            hidden = _gelu(np.asarray(rows) @ self.w1.T + self.b1)
            correction = self.residual_bound * np.tanh(hidden @ self.w2 + self.b2)
        return np.asarray(coverage_values), correction

    def fit_scores(self, specialties_list, flag_counts, symptom_key):
        """Specialty fit in [0, 1]: coverage plus learned correction, clipped."""
        coverage, correction = self.fits(specialties_list, flag_counts, symptom_key)
        return np.clip(coverage + correction, 0.0, 1.0)


def specialty_flag_count(quality_flags):
    return sum(flag in SPECIALTY_FLAGS for flag in filter(None, quality_flags.split(';')))


def load_model(path=MODEL_PATH):
    """Load the exported model, or return None so ranking falls back to the rule.

    Set SPECIALTY_MODEL=off to force the exact-match rule (e.g. to compare rankings).
    """
    if os.environ.get('SPECIALTY_MODEL', '').strip().lower() in {'off', '0', 'false', 'rule'}:
        return None
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return SpecialtyModel(json.load(handle))
    except (OSError, ValueError, KeyError) as error:
        print(f'Specialty model unavailable ({error}); using the exact-match rule.')
        return None
