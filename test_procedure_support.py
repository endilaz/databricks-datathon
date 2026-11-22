import json
import unittest

from procedure_support import extract_evidence
from ranking import rank_facilities


PHACO = 'Phacoemulsification'
MAINTENANCE_HD = 'Maintenance in-centre haemodialysis'


def support(procedures=(), equipment=(), capabilities=(), tags=(), cities=()):
    row = {
        'procedure': json.dumps(procedures), 'equipment': json.dumps(equipment),
        'capability': json.dumps(capabilities), 'description': '', 'description_continuation': '',
        'city_names_mentioned_in_text': json.dumps(cities), 'address_city_search': 'pune',
    }
    return extract_evidence(row, tags, [])[0]


class ProcedureSupportTests(unittest.TestCase):
    def test_equipment_only_lead_does_not_turn_into_a_procedure_claim(self):
        evidence = support(
            procedures=['Cataract surgery'], equipment=['Phacoemulsification Unit'],
            tags=['Cataract surgery'],
        )[PHACO]
        self.assertEqual(evidence['status'], 'Procedure variant needs confirmation')
        self.assertFalse(evidence['service'])
        self.assertLessEqual(evidence['points'], 10)

    def test_planned_phaco_equipment_does_not_create_a_variant_match(self):
        evidence = support(
            procedures=['Cataract surgery'], equipment=['Plans to install a phaco machine'],
            tags=['Cataract surgery'],
        )
        self.assertNotIn(PHACO, evidence)

    def test_service_claim_earns_points_with_no_equipment_or_capability_text(self):
        stated = support(procedures=['Performs phacoemulsification'])[PHACO]
        listed = support(procedures=['Phacoemulsification'])[PHACO]
        self.assertEqual(stated['points'], 30)
        self.assertEqual(listed['points'], 24)
        self.assertTrue(all(item['status'] == 'Unreported' for item in stated['equipment']))

    def test_negation_and_branch_mix(self):
        capability = ['No-stitch cataract surgery using phacoemulsification']
        self.assertTrue(support(capabilities=capability)[PHACO]['service'])
        mixed = support(capabilities=capability, cities=['mumbai'])[PHACO]
        self.assertEqual(mixed['status'], 'Possible branch mix')
        self.assertEqual(mixed['points'], 22)
        other_branch = support(
            procedures=['Performs phacoemulsification at the Mumbai branch'], cities=['mumbai'],
        )[PHACO]
        self.assertEqual(other_branch['status'], 'Procedure excerpt mentions another city')
        self.assertEqual(other_branch['points'], 10)
        negative = support(procedures=['We do not perform phacoemulsification.'])
        self.assertNotIn(PHACO, negative)
        conflicting = support(
            procedures=['We do not perform phacoemulsification.', 'Performs phacoemulsification'],
        )[PHACO]
        self.assertEqual(conflicting['status'], 'Conflicting procedure claims')
        self.assertEqual(conflicting['points'], 0)

    def test_fistula_maintenance_is_not_maintenance_dialysis(self):
        evidence = support(procedures=[
            'Vascular surgery procedures including hemodialysis access (AV fistula) creation and maintenance',
        ])
        self.assertNotIn(MAINTENANCE_HD, evidence)

    def test_equipment_contradiction_is_attached_to_the_equipment(self):
        evidence = support(
            procedures=['Performs phacoemulsification'],
            capabilities=['The phacoemulsification machine is unavailable'],
        )[PHACO]
        self.assertFalse(evidence['conflict'])
        self.assertEqual(evidence['equipment'][0]['status'], 'Conflicting or negative text')
        self.assertEqual(evidence['points'], 20)

    def test_manual_pd_does_not_require_a_cycler(self):
        evidence = support(
            procedures=['Offers continuous ambulatory peritoneal dialysis'],
            equipment=['Peritoneal dialysis supplies'],
        )
        manual = evidence['Manual peritoneal dialysis (CAPD)']
        self.assertEqual([item['label'] for item in manual['equipment']], ['Peritoneal dialysis supplies'])
        self.assertNotIn('Automated peritoneal dialysis (APD)', evidence)

    def test_evidence_can_outrank_distance_and_current_certificate_breaks_tie(self):
        weak = support(procedures=['Phacoemulsification'])[PHACO]
        strong = support(
            procedures=['Performs phacoemulsification'], equipment=['Phacoemulsification Unit'],
            capabilities=['Ophthalmic surgical team'],
        )[PHACO]
        base = {
            'lat': 18.6, 'lon': 73.9, 'level': 'Secondary', 'level_confidence': 'low',
            'specialties': [], 'quality_flags': '', 'source_urls': [],
        }
        candidates = [
            dict(base, name='A nearby listing', procedure_support={PHACO: weak}),
            dict(base, name='A documented facility', lat=19.0, procedure_support={PHACO: strong}),
            dict(base, name='B documented facility', lat=19.0, procedure_support={PHACO: strong},
                 accreditation={'valid_from': '2000-01-01', 'valid_until': '2100-01-01',
                                'procedure_family': 'Cataract surgery'}),
            dict(base, name='C documented facility', lat=19.0, procedure_support={PHACO: strong},
                 accreditation={'valid_from': '2000-01-01', 'valid_until': '2000-12-31',
                                'procedure_family': 'Cataract surgery'}),
        ]
        ranked = rank_facilities(iter(candidates), 18.6, 73.9, {}, {}, PHACO)
        self.assertEqual([item['name'] for item in ranked], [
            'B documented facility', 'A documented facility', 'C documented facility', 'A nearby listing',
        ])
        self.assertFalse(ranked[2]['active_certificate'])


if __name__ == '__main__':
    unittest.main()
