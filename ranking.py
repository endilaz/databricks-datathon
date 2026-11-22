"""Transparent, provisional facility ranking. No medical triage is performed here."""
from math import asin, cos, radians, sin, sqrt

RADIUS_MILES = 100.0
SEVERITY_WEIGHT = {'Low': 1, 'Medium': 2, 'High': 3}
LEVEL_POINTS = {
    'Low': {'Primary': 20, 'Secondary': 12, 'Tertiary': 6},
    'Medium': {'Primary': 8, 'Secondary': 20, 'Tertiary': 14},
    'High': {'Primary': 2, 'Secondary': 12, 'Tertiary': 20},
}

def distance_miles(lat1, lon1, lat2, lon2):
    a, b = radians(lat1), radians(lat2)
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    h = sin(dlat/2)**2 + cos(a)*cos(b)*sin(dlon/2)**2
    return 3958.7613 * 2 * asin(min(1, sqrt(h)))

def specialty_points(facility_specialties, selected, symptom_map):
    """Interim exact-code matching. Replace this function with teammate's model."""
    if not selected: return 40.0, []
    offered = set(facility_specialties)
    total_weight = sum(SEVERITY_WEIGHT[s] for s in selected.values())
    earned = 0.0; matches = []
    for (category, symptom), severity in selected.items():
        options = symptom_map[category][symptom]
        hit = next((name for name in options if name in offered), None)
        if hit:
            strength = 1.0 if hit == options[0] else 0.75
            earned += SEVERITY_WEIGHT[severity] * strength
            matches.append((symptom, hit))
    return round(40 * earned / total_weight, 2), matches

def rank_facilities(facilities, lat, lon, selected, symptom_map, selected_procedure=None):
    maximum = max((SEVERITY_WEIGHT[x] for x in selected.values()), default=1)
    severity = next(k for k,v in SEVERITY_WEIGHT.items() if v==maximum)
    ranked = []
    for f in facilities:
        if selected_procedure and selected_procedure not in f.get('procedures', []): continue
        d = distance_miles(lat, lon, f['lat'], f['lon'])
        if d > RADIUS_MILES: continue
        distance_score = round(40 * (1 - d/RADIUS_MILES), 2)
        spec_score, matches = specialty_points(f['specialties'], selected, symptom_map)
        level_score = LEVEL_POINTS[severity][f['level']] if selected else 20
        score = round(distance_score + spec_score + level_score, 2)
        weaknesses = ['We have not confirmed that this facility is open or currently offers the listed services.']
        if f['level_confidence'] != 'high': weaknesses.append('Its primary, secondary, or tertiary care level is an estimate based on limited information.')
        if selected and not matches: weaknesses.append('The available record does not list a specialty that clearly matches the symptoms you selected.')
        elif matches: weaknesses.append('The specialty list comes from web text and has not been checked with the facility.')
        else: weaknesses.append('You did not select symptoms, so we could not check whether its specialties fit your needs.')
        if selected_procedure: weaknesses.append('The procedure mention may be outdated or refer to another facility. Please call to confirm it.')
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
        if not f['source_urls']: weaknesses.append('We do not have a source link for this record.')
        else: weaknesses.append('We have not checked whether the source links describe this exact facility.')
        ranked.append({**f,'distance_miles':round(d,1),'distance_score':distance_score,
                       'specialty_score':spec_score,'level_score':level_score,'score':score,
                       'matched_specialties':matches,'matched_procedure_evidence':f.get('procedure_evidence',{}).get(selected_procedure,'') if selected_procedure else '',
                       'weaknesses':weaknesses})
    return sorted(ranked,key=lambda f:(-f['score'],f['distance_miles'],f['name'].casefold()))
