"""Transparent, provisional facility ranking. No medical triage is performed here."""
from datetime import date
from math import asin, cos, radians, sin, sqrt

from procedure_support import RULES, SUPPORT_MAX
from specialty_model import load_model, specialty_flag_count

SPECIALTY_MODEL = load_model()

RADIUS_MILES = 100.0
SEVERITY_WEIGHT = {'Low': 1, 'Medium': 2, 'High': 3}
LEVEL_POINTS = {
    'Low': {'Primary': 20, 'Secondary': 12, 'Tertiary': 6},
    'Medium': {'Primary': 8, 'Secondary': 20, 'Tertiary': 14},
    'High': {'Primary': 2, 'Secondary': 12, 'Tertiary': 20},
}

CLEANING_FLAG_EXPLANATIONS = {
    'source_array_length_mismatch': 'The source types and source IDs have different list lengths, so they cannot be reliably paired.',
    'multiple_city_names_in_text_review': 'The source text mentions multiple cities. Check which statements describe this branch.',
    'official_phone_shared_by_5_or_more_records': 'At least five records use this official phone number. Confirm that it reaches this branch.',
    'invalid_or_missing_pin': 'The original export PIN is missing or fails the six-digit format check.',
    'planned_service_language_review': 'The source text may describe a planned service. Check which services are currently available.',
    'invalid_page_update_date': 'The exported page update date could not be read as a full date.',
    'page_update_in_future': 'The page update date is later than the export audit date.',
    'description_continuation_recovered_review': 'Part of the exported description was recovered from split rows and is flagged for review.',
    'coordinates_outside_broad_india_bounds': 'The source coordinates fall outside the geographic bounds used by the cleaner.',
}

def distance_miles(lat1, lon1, lat2, lon2):
    a, b = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    h = sin(dlat/2)**2 + cos(a)*cos(b)*sin(dlon/2)**2
    return 3958.7613 * 2 * asin(min(1, sqrt(h)))

def rule_fit(facility_specialties, options):
    """Exact-code fallback: 1.0 for the first listed specialty, 0.75 for another."""
    offered = set(facility_specialties)
    hit = next((name for name in options if name in offered), None)
    return (1.0 if hit == options[0] else 0.75) if hit else 0.0, hit

def model_fits(facilities, selected, symptom_map, model=SPECIALTY_MODEL):
    """Per-facility {symptom key: (fit, matched code)} from the learned specialty model.

    Each symptom is scored on its own, as in training (single-symptom scenarios).
    Returns None when the model is unavailable, so callers fall back to rule_fit.
    """
    if model is None or not selected or any(key not in model.symptoms for key in selected):
        return None
    specialties = [f['specialties'] for f in facilities]
    flags = [specialty_flag_count(f.get('quality_flags', '')) for f in facilities]
    canonical = [{model.canonical(k) for k in s} for s in specialties]
    table = [{} for _ in facilities]
    for key in selected:
        options = symptom_map[key[0]][key[1]]
        scores = model.fit_scores(specialties, flags, key)
        for row, offered, score in zip(table, canonical, scores):
            hit = next((name for name in options if model.canonical(name) in offered), None)
            row[key] = (float(score), hit)
    return table

def specialty_points(facility_specialties, selected, symptom_map, fits=None):
    """Severity-weighted specialty fit, scaled to 40 points.

    `fits` maps each selected symptom to (fit in [0, 1], matched code) from the learned
    model; without it, the exact-code rule is used.
    """
    if not selected: return 40.0, []
    total_weight = sum(SEVERITY_WEIGHT[s] for s in selected.values())
    earned = 0.0; matches = []
    for (category, symptom), severity in selected.items():
        fit, hit = fits[(category, symptom)] if fits else rule_fit(facility_specialties, symptom_map[category][symptom])
        earned += SEVERITY_WEIGHT[severity] * fit
        if hit: matches.append((symptom, hit))
    return round(40 * earned / total_weight, 2), matches

def rank_facilities(facilities, lat, lon, selected, symptom_map, selected_procedure=None):
    maximum = max((SEVERITY_WEIGHT[x] for x in selected.values()), default=1)
    severity = next(k for k,v in SEVERITY_WEIGHT.items() if v==maximum)
    use_support = False
    if selected_procedure:
        facilities = list(facilities)
        if selected_procedure in RULES:
            use_support = any('procedure_support' in f for f in facilities)
        else:
            use_support = any(selected_procedure in f.get('procedure_support', {}) for f in facilities)
    procedure_family = RULES[selected_procedure][0] if selected_procedure in RULES else selected_procedure
    today = date.today().isoformat()
    base_weight = (100 - SUPPORT_MAX) / 100 if use_support else 1
    candidates = []
    for f in facilities:
        support = f.get('procedure_support', {}).get(selected_procedure) if use_support else None
        if use_support and support is None: continue
        if selected_procedure and not use_support and selected_procedure not in f.get('procedures', []): continue
        d = distance_miles(lat, lon, f['lat'], f['lon'])
        if d > RADIUS_MILES: continue
        candidates.append((f, support, d))
    fit_table = model_fits([f for f, _, _ in candidates], selected, symptom_map)
    ranked = []
    for position, (f, support, d) in enumerate(candidates):
        distance_score = round(40 * base_weight * (1 - d/RADIUS_MILES), 2)
        fits = fit_table[position] if fit_table else None
        spec_score, matches = specialty_points(f['specialties'], selected, symptom_map, fits)
        spec_score = round(spec_score * base_weight, 2)
        level_score = LEVEL_POINTS[severity][f['level']] if selected else 20
        level_score *= base_weight
        evidence_score = support['points'] if support else 0
        score = round(distance_score + spec_score + level_score + evidence_score, 2)
        certificate = f.get('accreditation')
        active_certificate = bool(certificate and certificate['valid_from'] <= today <= certificate['valid_until'])
        relevant_certificate = active_certificate and use_support and certificate['procedure_family'] == procedure_family
        weaknesses = ['We have not confirmed that this facility is open or currently offers the listed services.']
        if f['level_confidence'] != 'high': weaknesses.append('Its primary, secondary, or tertiary care level is an estimate based on limited information.')
        if selected and not matches: weaknesses.append('The available record does not list a specialty that clearly matches the symptoms you selected.')
        elif matches: weaknesses.append('The specialty list comes from web text and has not been checked with the facility.')
        else: weaknesses.append('You did not select symptoms, so we could not check whether its specialties fit your needs.')
        if selected_procedure: weaknesses.append('The procedure mention may be outdated or refer to another facility. Please call to confirm it.')
        if support and support['status'] in ('Possible branch mix', 'Procedure excerpt mentions another city'):
            weaknesses.append('Location references in the export make it unclear which branch the procedure evidence describes.')
        flag_explanations = {
            'repetitive_specialties_list': 'Some specialties are repeated in the record.',
            'invalid_email_format': 'The email address may be incorrect.',
            'invalid_pincode_format': 'The PIN code may be incorrect.',
            'missing_pincode': 'The record has no PIN code.',
            'repetitive_source_urls_list': 'Some source links are repeated.',
            'repetitive_procedure_list': 'Some procedure claims are repeated.',
            'duplicate_id': 'This record ID appears more than once in the source data.',
            'implausible_yearEstablished': 'The recorded founding year looks incorrect.',
            'repetitive_capability_list': 'Some listed services are repeated.',
        }
        for flag in filter(None, f['quality_flags'].split(';')):
            if flag in flag_explanations: weaknesses.append(flag_explanations[flag])
        weaknesses.extend(CLEANING_FLAG_EXPLANATIONS[flag] for flag in f.get('cleaning_flags', [])
                          if flag in CLEANING_FLAG_EXPLANATIONS)
        if not f['source_urls']: weaknesses.append('We do not have a source link for this record.')
        else: weaknesses.append('We have not checked whether the source links describe this exact facility.')
        procedure_quote = support['service'].get('quote', '') if support else f.get('procedure_evidence', {}).get(selected_procedure, '')
        ranked.append({**f,'distance_miles':round(d,1),'distance_score':distance_score,
                       'specialty_score':spec_score,'level_score':level_score,'score':score,
                       'matched_specialties':matches,'matched_procedure_evidence':procedure_quote,
                       'procedure_support_detail':support,'procedure_support_score':evidence_score,
                       'active_certificate':active_certificate,'relevant_certificate':relevant_certificate,
                       'weaknesses':weaknesses})
    if use_support:
        return sorted(ranked, key=lambda f: (
            f['procedure_support_detail']['status'] == 'Conflicting procedure claims',
            -f['score'], -f['procedure_support_score'], -f['relevant_certificate'],
            f['distance_miles'], f['name'].casefold(),
        ))
    return sorted(ranked,key=lambda f:(-f['score'],f['distance_miles'],f['name'].casefold()))
