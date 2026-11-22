import unittest

import ranking
from ranking import model_fits, rank_facilities, specialty_points
from specialty_model import load_model

SYMPTOMS = {'General': {'Fever': ['familyMedicine', 'internalMedicine', 'infectiousDiseases']}}
FEVER = ('General', 'Fever')


def facility(fid, specialties, lat=18.6):
    return {'id': fid, 'name': fid, 'lat': lat, 'lon': 73.9, 'level': 'Secondary',
            'level_confidence': 'high', 'specialties': specialties, 'quality_flags': '',
            'source_urls': ['https://example.org'], 'cleaning_flags': []}


class SpecialtyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        if cls.model is None:
            raise unittest.SkipTest('data/specialty_model.json.gz is not available')

    def test_numpy_runtime_matches_torch_reference_outputs(self):
        keys = {i: key for key, i in self.model.symptoms.items()}
        for ref in self.model.references:
            coverage, correction = self.model.fits(
                [ref['specialties']], [len(ref['specialty_quality_flags'])], keys[ref['symptom_index']])
            self.assertAlmostEqual(coverage[0], ref['weighted_coverage'], places=6)
            self.assertAlmostEqual(correction[0], ref['learned_correction'], places=6)

    def test_app_codes_map_to_model_vocabulary(self):
        self.assertEqual(self.model.canonical('internalMedicine'), 'Internal Medicine')
        self.assertEqual(self.model.canonical('ent'), 'Otolaryngology')
        self.assertIsNone(self.model.canonical('notASpecialty'))

    def test_fits_are_bounded_and_matches_use_app_codes(self):
        fits = model_fits([facility('a', ['internalMedicine', 'infectiousDiseases']),
                           facility('b', ['dentistry'])], {FEVER: 'High'}, SYMPTOMS)
        for row in fits:
            self.assertTrue(0.0 <= row[FEVER][0] <= 1.0)
        self.assertEqual(fits[0][FEVER][1], 'internalMedicine')
        self.assertIsNone(fits[1][FEVER][1])
        self.assertGreater(fits[0][FEVER][0], fits[1][FEVER][0])

    def test_ranking_uses_model_and_falls_back_to_rule(self):
        facilities = [facility('a', ['familyMedicine']), facility('b', ['internalMedicine', 'infectiousDiseases'])]
        ranked = rank_facilities(facilities, 18.6, 73.9, {FEVER: 'Medium'}, SYMPTOMS)
        self.assertEqual({r['id'] for r in ranked}, {'a', 'b'})
        saved = ranking.SPECIALTY_MODEL
        try:
            self.assertIsNone(model_fits(facilities, {FEVER: 'Medium'}, SYMPTOMS, model=None))
            self.assertEqual(specialty_points(['familyMedicine'], {FEVER: 'Medium'}, SYMPTOMS),
                             (40.0, [('Fever', 'familyMedicine')]))
        finally:
            ranking.SPECIALTY_MODEL = saved


if __name__ == '__main__':
    unittest.main()
