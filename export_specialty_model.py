"""Export the trained specialty ranker to a torch-free JSON bundle for the app.

The model lives in `facilities/specialty/` on the `feat/facilities-specialty-model`
branch. This build step needs PyTorch; the app itself only needs numpy.

    python export_specialty_model.py --component path/to/facilities/specialty
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = 'runs/architecture_refinement/semantic_residual_seed_44/model.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--component', required=True, help='Path to facilities/specialty')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Relative to --component')
    parser.add_argument('--data', default='data/pairwise_v2/prepared', help='Relative to --component')
    parser.add_argument('--out', default=str(ROOT / 'data' / 'specialty_model.json.gz'))
    args = parser.parse_args()

    component = Path(args.component).resolve()
    sys.path.insert(0, str(component))
    import torch
    from ranker.features import FeatureBuilder
    from ranker.model import model_from_checkpoint

    checkpoint_path = component / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'semantic_residual':
        raise ValueError('The app runtime implements only the semantic_residual architecture')
    features = FeatureBuilder(component / args.data)
    if checkpoint['vocabulary'] != features.vocab:
        raise ValueError('Checkpoint vocabulary differs from prepared data')
    model = model_from_checkpoint(checkpoint)
    model.eval()
    dictionary = json.loads((component / 'data/source/facility_specialty_symptom_dictionary_merged.json').read_text())

    config = checkpoint['config']
    active = config['active_indices']
    state = {k: v.detach().double().tolist() for k, v in model.state_dict().items()}
    need_positions = [[j for j, x in enumerate(row) if x] for row in config['symptom_need_map']]

    # Reference outputs from torch so the numpy runtime can be checked against them.
    rng = random.Random(0)
    facility_ids = sorted(features.facilities)
    references = []
    for _ in range(64):
        symptom = rng.randrange(len(features.symptoms))
        fid = rng.choice(facility_ids)
        scenario = {'symptoms': [{'symptom_id': features.symptoms[symptom]['id']}]}
        vectors = [v.unsqueeze(0) for v in features.encode(scenario, features.facilities[fid])]
        with torch.no_grad():
            parts = model.components(*vectors)
        references.append({
            'specialties': features.facilities[fid]['specialties'],
            'specialty_quality_flags': features.facilities[fid]['specialty_quality_flags'],
            'symptom_index': symptom,
            'weighted_coverage': parts['weighted_coverage'].item(),
            'learned_correction': parts['learned_correction'].item(),
        })

    bundle = {
        'source': {
            'branch': 'feat/facilities-specialty-model',
            'checkpoint': args.checkpoint,
            'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'architecture': checkpoint['architecture'],
            'meaning': 'Provisional record-based specialty fit trained on AI pairwise labels; '
                       'not a probability or validated clinical referral.',
        },
        'specialties': features.specialties,
        'idf': features.idf.double().tolist(),
        'aliases': dictionary['specialty_aliases'],
        'symptoms': [[s['category'], s['name'], s['specialties']] for s in features.symptoms],
        'active_indices': active,
        'need_positions': need_positions,
        'residual_bound': config['residual_bound'],
        'weights': {
            'specialty_embeddings': state['specialty_embeddings'],
            'w1': state['correction.0.weight'], 'b1': state['correction.0.bias'],
            'w2': state['correction.3.weight'], 'b2': state['correction.3.bias'],
            'log_scale': state['log_scale'],
        },
        'references': references,
    }
    out = Path(args.out)
    with gzip.open(out, 'wt', encoding='utf-8') as handle:
        json.dump(bundle, handle, separators=(',', ':'))
    print(f'Wrote {out} ({out.stat().st_size:,} bytes), {len(active)} active specialties')


if __name__ == '__main__':
    main()
