"""Screen export excerpts against the cataract and dialysis research in the methodology."""

import json
import re


SERVICE_MAX = 30
EQUIPMENT_MAX = 12
CAPABILITY_MAX = 8
SUPPORT_MAX = SERVICE_MAX + EQUIPMENT_MAX + CAPABILITY_MAX
CATARACT_REFERENCE = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9023903/'
DIALYSIS_REFERENCE = 'https://nhsrcindia.org/pradhan-mantri-national-dialysis-program'

CATARACT = r'\bcataract (?:surgery|surgeries|operations?|extraction)\b'
PHACO = r'\bphaco(?:emulsification)?\b'
MSICS = r'\b(?:msics|manual small[- ]incision cataract surgery)\b'
DIALYSIS = r'\bdialysis\b'
HD = r'\bh[ae]emodialysis\b|\bhemodialysis\b'
PD = r'\b(?:peritoneal dialysis|capd|apd)\b'
CAPD = r'\b(?:capd|continuous ambulatory peritoneal dialysis|manual peritoneal dialysis)\b'
APD = r'\b(?:apd|automated peritoneal dialysis)\b'
PHACO_SYSTEM = rf'{PHACO}[\w /-]{{0,22}}\b(?:machine|unit|system|platform|equipment)\b'
HD_MACHINE = rf'(?:{HD}|(?<!peritoneal )\bdialysis\b)[\w /-]{{0,15}}\b(?:machines?|units? of equipment)\b'
CYCLER = r'\b(?:cyclers?|automated peritoneal dialysis machines?)\b'

EYE_EQUIPMENT = (
    ('Operating microscope', r'\b(?:operating|surgic(?:al|ery)|ophthalmic) microscopes?\b'),
    ('Sterilization equipment', r'\b(?:autoclaves?|plasma sterili[sz]ers?|instrument sterili[sz]ers?)\b'),
)
EYE_CAPABILITIES = (
    ('Ophthalmic operating theatre', r'\b(?:eye|ophthalmic|ophthalmology|cataract)\b.{0,42}\b(?:operating|operation) theat(?:re|er)s?\b|\b(?:operating|operation) theat(?:re|er)s?\b.{0,42}\b(?:eye|ophthalmic|ophthalmology|cataract)\b'),
    ('Ophthalmic surgical team', r'\b(?:cataract|ophthalmic|eye) surgeons?\b|\bophthalmic surgical team\b'),
    ('Instrument sterilization', r'\b(?:surgical )?instrument sterili[sz]ation\b|\bsterili[sz]ation of (?:surgical )?instruments\b'),
)
HD_EQUIPMENT = (
    ('Dialysis machines', HD_MACHINE),
    ('Dialysis water treatment', r'\b(?:reverse osmosis|ro water treatment)\b.{0,65}\b(?:water|dialysis|plant|system|treatment)\b|\bdialysis\b.{0,65}\b(?:reverse osmosis|ro plant)\b'),
)
HD_CAPABILITIES = (
    ('Dialysis nursing or technical team', r'\bdialysis (?:nurses?|nursing|technicians?|team|staff)\b|\b(?:nurses?|technicians?) trained in dialysis\b'),
    ('Dialysis infection-control arrangements', r'\bdialysis\b.{0,45}\b(?:infection control|isolation)\b|\b(?:infection control|isolation)\b.{0,45}\bdialysis\b'),
)
PD_EQUIPMENT = (
    ('Peritoneal dialysis supplies', r'\b(?:peritoneal dialysis|capd|apd|pd) (?:fluids?|solutions?|supplies|bags|kits)\b'),
)
PD_CAPABILITIES = (
    ('Peritoneal dialysis training or follow-up', r'\b(?:peritoneal dialysis|capd|apd|pd)\b.{0,65}\b(?:patient training|caregiver training|home training|follow[- ]up|home support|patient education)\b|\b(?:train(?:ing|s)?|educat(?:ion|es))\b.{0,65}\b(?:peritoneal dialysis|capd|apd)\b'),
    ('Peritoneal dialysis catheter placement', r'\b(?:peritoneal dialysis|capd|apd|pd|tenckhoff) catheters?\b.{0,35}\b(?:placement|insertion)\b|\b(?:placement|insertion)\b.{0,35}\b(?:peritoneal dialysis|capd|pd|tenckhoff) catheters?\b'),
)

# Reference criteria are used to screen for documented support, not to certify readiness.
RULES = {
    'Cataract surgery': ('Cataract surgery', f'{CATARACT}|{PHACO}|{MSICS}',
                         EYE_EQUIPMENT, EYE_CAPABILITIES, ''),
    'Phacoemulsification': ('Cataract surgery', PHACO,
                            (('Phaco system', PHACO_SYSTEM), *EYE_EQUIPMENT), EYE_CAPABILITIES, PHACO_SYSTEM),
    'Manual small-incision cataract surgery': ('Cataract surgery', MSICS,
                                                EYE_EQUIPMENT, EYE_CAPABILITIES, ''),
    'Dialysis': ('Dialysis', f'{DIALYSIS}|{HD}|{PD}', HD_EQUIPMENT, HD_CAPABILITIES, ''),
    'Haemodialysis (any setting)': ('Dialysis', HD, HD_EQUIPMENT, HD_CAPABILITIES, HD_MACHINE),
    'Maintenance in-centre haemodialysis': (
        'Dialysis', rf'(?:(?:maintenance|chronic|regular|outpatient|in[- ]cent(?:re|er))\b.{{0,50}}(?:{HD})|'
        rf'(?:{HD}).{{0,50}}\b(?:maintenance|chronic|regular|outpatient|in[- ]cent(?:re|er))\b)',
        HD_EQUIPMENT, HD_CAPABILITIES, HD_MACHINE,
    ),
    'Peritoneal dialysis': ('Dialysis', PD, PD_EQUIPMENT, PD_CAPABILITIES, ''),
    'Manual peritoneal dialysis (CAPD)': ('Dialysis', CAPD, PD_EQUIPMENT, PD_CAPABILITIES, ''),
    'Automated peritoneal dialysis (APD)': (
        'Dialysis', APD, (('Peritoneal dialysis cycler', CYCLER), *PD_EQUIPMENT), PD_CAPABILITIES, CYCLER,
    ),
}

NEGATIVE = re.compile(
    r'\b(?:not (?:currently )?(?:offered|available|provided|performed|operational)|unavailable|'
    r'no longer (?:offers?|provides?|performs?|available)|(?:do|does) not (?:offer|provide|perform|have)|'
    r'(?:service|unit|equipment|machine)s? (?:is |are )?(?:offline|closed|suspended))\b', re.I,
)
NON_OPERATIONAL = re.compile(
    r'\b(?:plans? to|planned|proposed|coming soon|not yet|will (?:offer|introduce|start|install)|'
    r'(?:refer|referred|refers|referral)s? (?:out|to|for)|off[- ]site|under construction|'
    r'clinical trials?|case stud(?:y|ies)|research (?:study|project)|'
    r'training (?:course|workshop|programme for surgeons)|fellowship|'
    r'(?:previously|formerly) (?:offered|provided|performed)|performed in (?:19|20)\d{2}|'
    r'(?:h[ae]?emodialysis|dialysis) (?:uses|works by|is a (?:process|procedure|treatment))|'
    r'across (?:our |its |the )?(?:branches|locations|network))\b', re.I,
)
ACTION = re.compile(
    r'\b(?:offers?|provides?|performs?|provided|performed|available|delivers?|uses?|using|in[- ]house|on[- ]site)\b', re.I,
)
UNRELATED_SERVICE_ITEM = re.compile(
    r'\b(?:consultation|counselling|screening|evaluation|assessment|pre[- ]operative|'
    r'post[- ]operative|follow[- ]up|training|machines?|equipment|cyclers?|supplies|catheters?|'
    r'(?:av |arteriovenous )fistula|vascular access|h[ae]?emodialysis access)\b', re.I,
)
PHACO_DEVICE_ITEM = re.compile(PHACO_SYSTEM, re.I)
ACUTE_OR_HOME = re.compile(r'\b(?:acute|emergency|bedside|icu|home)[- ](?:only[- ])?h[ae]?emodialysis\b', re.I)
NABH = re.compile(r'\bNABH\b', re.I)

# These patterns identify directory wording associated with a procedure. They do
# not describe required equipment or establish whether a facility can provide it.
CATALOG_EQUIPMENT_TERMS = {
    'Appendectomy': r'(?:appendectomy|appendicectomy) (?:kits?|instrument sets?)',
    'Arthroscopy': r'arthroscopes?|arthroscop(?:y|ic) (?:towers?|shavers?|instruments?)',
    'Biopsy': r'biopsy (?:needles?|guns?|forceps|devices?|punches?)',
    'Blood transfusion': r'blood (?:bag sealers?|warmers?)|transfusion (?:warmers?|pumps?|sets?)',
    'Cesarean section': r'(?:caesarean|cesarean|c-section) (?:kits?|instrument sets?|instruments?)',
    'Chemotherapy': r'chemotherapy (?:infusion )?(?:pumps?|preparation hoods?)',
    'Colonoscopy': r'colonoscopes?|colonoscopy (?:towers?|processors?)',
    'Coronary angiography': r'coronary angiograph(?:y|ic) (?:catheters?|units?|systems?)',
    'Coronary angioplasty': r'coronary (?:angioplasty balloons?|balloon catheters?|stents?)',
    'Coronary bypass surgery': r'(?:coronary bypass|CABG) (?:instrument sets?|instruments?)',
    'CT scan': r'(?:(?<!PET[- /])CT|computed tomography) scanners?',
    'Dental bridge': r'(?:dental bridge|bridge fabrication) (?:milling machines?|milling systems?)',
    'Dental crown': r'(?:dental crown|crown fabrication) (?:milling machines?|milling systems?)',
    'Dental filling': r'(?:composite filling|dental composite) curing lights?',
    'Dental implant': r'(?:dental implant|implantology) (?:drills?|motors?|surgical kits?)',
    'Dentures': r'denture (?:flasks?|presses|processing units?)',
    'Endoscopy': r'endoscopes?|endoscopy (?:towers?|processors?)',
    'Gallbladder removal': r'cholecystectomy (?:kits?|instruments?|instrument sets?)',
    'Hemorrhoid surgery': r'h[ae]morrhoid(?:ectomy)? (?:staplers?|banding devices?|instrument sets?)',
    'Hip replacement': r'(?:total hip|hip replacement) (?:prosthes[ei]s|implants?|arthroplasty instruments?)',
    'Hysterectomy': r'hysterectomy (?:manipulators?|instrument sets?)',
    'Hysteroscopy': r'hysteroscopes?|hysteroscopy (?:towers?|fluid systems?)',
    'In vitro fertilization (IVF)': r'(?:IVF|in vitro fertilization) (?:embryo )?incubators?',
    'Intracytoplasmic sperm injection (ICSI)': r'(?:ICSI|intracytoplasmic sperm injection) (?:micromanipulators?|micropipettes?|microinjection systems?)',
    'Intrauterine insemination (IUI)': r'(?:IUI|intrauterine insemination) catheters?',
    'Joint replacement': r'joint (?:replacement|arthroplasty) (?:implants?|instrument sets?)',
    'Kidney transplant': r'(?:kidney|renal) (?:transplant|perfusion) (?:instrument sets?|pumps?|machines?)',
    'Knee replacement': r'(?:total knee|knee replacement) (?:prosthes[ei]s|implants?|arthroplasty instruments?)',
    'LASIK eye surgery': r'LASIK (?:lasers?|platforms?|systems?)',
    'Lithotripsy': r'lithotripters?|lithotripsy (?:lasers?|probes?)',
    'Liver transplant': r'liver (?:transplant|perfusion) (?:instrument sets?|pumps?|machines?)',
    'MRI scan': r'(?:MRI|magnetic resonance imaging) (?:scanners?|machines?)',
    'Pacemaker implantation': r'pacemaker (?:implantation kits?|programmers?)',
    'Prostate removal': r'prostatectomy (?:kits?|instruments?|instrument sets?)',
    'Radiation therapy': r'(?:medical )?linear accelerators?|radiotherapy (?:units?|machines?)',
    'Retinal detachment surgery': r'retinal detachment (?:repair )?(?:instrument sets?|instruments?)',
    'Root canal treatment': r'(?:rotary endodontic|endodontic rotary|root canal) (?:files?|motors?|systems?|instruments?)|(?:electronic )?apex locators?',
    'Spinal fusion': r'spinal fusion (?:cages?|implants?|instrument sets?)',
    'Teeth cleaning': r'(?:ultrasonic|dental) scalers?|ultrasonic scaling (?:units?|systems?)',
    'Teeth whitening': r'(?:tooth|teeth|dental) (?:whitening|bleaching) (?:lamps?|lights?|lasers?)',
    'Tonsillectomy': r'tonsillectomy (?:snares?|instrument sets?)',
    'Tooth extraction': r'(?:tooth|dental) extraction (?:forceps|elevators?)',
    'Ultrasound scan': r'ultrasound (?:scanners?|machines?|imaging (?:systems?|equipment))',
    'Ureteroscopy': r'ureteroscopes?|ureteroscopy (?:towers?|instruments?)',
    'Varicose vein treatment': r'(?:varicose vein|endovenous) (?:ablation )?(?:lasers?|laser systems?|ablation catheters?)',
    'Wisdom tooth extraction': r'(?:wisdom tooth|third molar) extraction (?:forceps|elevators?)',
    'X-ray imaging': r'x[- ]?ray (?:imaging )?(?:machines?|systems?|equipment|units?)',
}
CATALOG_UNCONFIRMED_CONTEXT = re.compile(
    r'\b(?:stock (?:image|photo)|(?:image|photo|illustration) (?:depicts?|shows?)|outsourc(?:ed|ing))\b', re.I,
)
CATALOG_PROCEDURE_ALIASES = {
    'Appendectomy': r'appendicectomy',
    'Cesarean section': r'caesarean section|c-section',
    'Coronary bypass surgery': r'coronary artery bypass grafting|CABG',
    'CT scan': r'(?<!PET[- /])CT scanning|computed tomography',
    'Gallbladder removal': r'cholecystectomy',
    'Hemorrhoid surgery': r'ha?emorrhoidectomy',
    'In vitro fertilization (IVF)': r'IVF|in vitro fertilisation',
    'Intracytoplasmic sperm injection (ICSI)': r'ICSI',
    'Intrauterine insemination (IUI)': r'IUI',
    'Joint replacement': r'joint arthroplasty',
    'LASIK eye surgery': r'LASIK',
    'MRI scan': r'MRI|magnetic resonance imaging',
    'Prostate removal': r'prostatectomy',
    'Radiation therapy': r'radiotherapy',
    'Root canal treatment': r'root canals?|RCT',
    'Ultrasound scan': r'ultrasonography|ultrasound',
    'X-ray imaging': r'x[- ]?rays?|radiography',
}


def catalog_support_patterns(procedures):
    devices = r'(?:equipment|machines?|devices?|systems?|kits?|instruments?|instrument sets?)'
    capability = r'(?:teams?|specialists?|trained staff|operating theatres?|procedure suites?|laborator(?:y|ies)|labs?)'
    patterns = {}
    for name in procedures:
        if name in RULES:
            continue
        term = re.escape(name.split(' (', 1)[0])
        if name == 'CT scan':
            term = r'(?<!PET[- /])' + term
        terms = '|'.join(filter(None, (term, CATALOG_PROCEDURE_ALIASES.get(name))))
        service = rf'\b(?:{terms})\b'
        phrases = [rf'{service}\s+{devices}', rf'{devices}\s+for\s+{service}']
        if alias := CATALOG_EQUIPMENT_TERMS.get(name):
            phrases.append(alias)
        equipment = r'\b(?:' + '|'.join(phrases) + r')\b'
        capability_pattern = rf'(?:{service}\s+{capability}\b|\b{capability}\s+(?:for|trained in)\s+{service})'
        patterns[name] = (service, equipment, capability_pattern)
    return patterns


def _entries(row):
    for field in ('procedure', 'capability', 'equipment', 'description', 'description_continuation'):
        values = json.loads(row[field]) if field in ('procedure', 'capability', 'equipment') else [row[field]]
        for index, text in enumerate(values):
            if isinstance(text, str) and text.strip():
                yield {'field': field, 'index': index if field in ('procedure', 'capability', 'equipment') else None,
                       'quote': text.strip()}


def _matching(entries, pattern):
    return [entry for entry in entries if re.search(pattern, entry['quote'], re.I)]


def _excerpt(entry, pattern):
    if not entry or len(entry['quote']) <= 600:
        return entry or {}
    quote = entry['quote']
    match = re.search(pattern, quote, re.I)
    if not match:
        return {**entry, 'quote': quote[:600]}
    start = max(0, match.start() - 180)
    end = min(len(quote), max(match.end() + 180, start + 300))
    return {**entry, 'quote': quote[start:end]}


def _non_operational(quote, pattern):
    return NON_OPERATIONAL.search(quote) or re.search(
        rf'\bno\s+(?:in[- ]house\s+|on[- ]site\s+)?(?:{pattern})', quote, re.I,
    )


def _negative(quote, pattern):
    return any(
        NEGATIVE.search(clause) and re.search(pattern, clause, re.I)
        for clause in re.split(r'[.!?;\n]+', quote)
    ) or re.search(
        rf'\bno\s+(?:in[- ]house\s+|on[- ]site\s+)?(?:{pattern})', quote, re.I,
    )


def _requirements(entries, requirements, *, uncertain_context=None):
    results = []
    for label, pattern in requirements:
        hits = _matching(entries, pattern)
        negative = next((entry for entry in hits if _negative(entry['quote'], pattern)), None)
        support = next((
            entry for entry in hits
            if _reports_requirement(entry, pattern)
            and not (uncertain_context and uncertain_context.search(entry['quote']))
        ), None)
        entry = negative or support or (hits[0] if hits else {})
        status = ('Conflicting or negative text' if negative else 'Reported in export' if support
                  else 'Ambiguous or non-current text' if hits else 'Unreported')
        results.append({'label': label, 'status': status, **_excerpt(entry, pattern)})
    return results


def _action_near(quote, pattern):
    return any(ACTION.search(clause) and re.search(pattern, clause, re.I)
               for clause in re.split(r'[.!?;\n]+', quote))


def _reports_requirement(entry, pattern):
    quote = entry['quote']
    return (not _non_operational(quote, pattern) and not _negative(quote, pattern)
            and (entry['field'] in ('equipment', 'capability') or _action_near(quote, pattern)))


def _evaluate(entries, name, existing_tag, mixed_branch, other_cities):
    family, pattern, equipment_rules, capability_rules, indirect = RULES[name]
    hits = _matching(
        (entry for entry in entries if entry['field'] != 'equipment'), pattern,
    )
    service_hits = [
        entry for entry in hits
        if not UNRELATED_SERVICE_ITEM.search(entry['quote'])
        and not PHACO_DEVICE_ITEM.search(entry['quote'])
        and (name != 'Maintenance in-centre haemodialysis' or not ACUTE_OR_HOME.search(entry['quote']))
    ]
    negative = next((entry for entry in service_hits if _negative(entry['quote'], pattern)), None)
    positive = [entry for entry in service_hits if not _non_operational(entry['quote'], pattern)
                and not _negative(entry['quote'], pattern)]
    stated = next((entry for entry in positive if _action_near(entry['quote'], pattern)), None)
    listed = next((entry for entry in positive if entry['field'] in ('procedure', 'capability')), None)
    service = stated or listed
    indirect_hits = existing_tag and indirect and any(
        _reports_requirement(entry, indirect) for entry in _matching(entries, indirect)
    )
    if (negative and not service) or not (service or indirect_hits or (name == family and existing_tag)):
        return None
    equipment = _requirements(entries, equipment_rules)
    capabilities = _requirements(entries, capability_rules)
    return _score_support(
        service, negative, equipment, capabilities, pattern,
        service_points=SERVICE_MAX if stated else 24 if listed else 0,
        mixed_branch=mixed_branch, other_cities=other_cities,
        variant=name != family, indirect_hits=indirect_hits,
    )


def _score_support(service, negative, equipment, capabilities, pattern, *, service_points,
                   mixed_branch, other_cities, variant=False, indirect_hits=False):
    equipment_hits = sum(item['status'] == 'Reported in export' for item in equipment)
    capability_hits = sum(item['status'] == 'Reported in export' for item in capabilities)
    points = service_points
    if equipment:
        points += EQUIPMENT_MAX * equipment_hits / len(equipment)
    if capabilities:
        points += CAPABILITY_MAX * capability_hits / len(capabilities)
    supporting_conflict = any(
        item['status'] == 'Conflicting or negative text' for item in (*equipment, *capabilities)
    )
    if supporting_conflict:
        points = min(points, 20)
    if negative:
        status, points = 'Conflicting procedure claims', 0
    elif not service:
        status = 'Procedure variant needs confirmation' if variant else 'Procedure needs confirmation'
        points = 10 if indirect_hits else min(points, 10)
    elif mixed_branch:
        other_city_in_claim = other_cities and other_cities.search(service['quote'])
        status = 'Procedure excerpt mentions another city' if other_city_in_claim else 'Possible branch mix'
        points = min(points, 10 if other_city_in_claim else 22)
    elif supporting_conflict:
        status = 'Conflicting equipment or capability evidence'
    else:
        status = 'Procedure and supporting evidence reported' if equipment_hits or capability_hits else 'Procedure reported'
    return {
        'status': status, 'points': round(points, 2), 'service': _excerpt(service, pattern),
        'conflict': _excerpt(negative, pattern), 'equipment': equipment, 'capabilities': capabilities,
    }


def _catalog_support(entries, quote, patterns, mixed_branch, other_cities):
    service_pattern, equipment_pattern, capability_pattern = patterns
    service = next(entry for entry in entries if entry['field'] == 'procedure' and entry['quote'] == quote.strip())
    service_entries = [service]
    service_entries.extend(
        entry for entry in _matching((item for item in entries if item['field'] == 'procedure'), service_pattern)
        if entry is not service and not UNRELATED_SERVICE_ITEM.search(entry['quote'])
    )
    negative = next((entry for entry in service_entries if _negative(entry['quote'], service_pattern)), None)
    if _non_operational(service['quote'], service_pattern) or CATALOG_UNCONFIRMED_CONTEXT.search(service['quote']):
        service = None
    stated = service and _action_near(service['quote'], service_pattern)
    equipment = _requirements(
        [entry for entry in entries if entry['field'] == 'equipment'],
        [('Related equipment', equipment_pattern)], uncertain_context=CATALOG_UNCONFIRMED_CONTEXT,
    )
    capabilities = _requirements(
        [entry for entry in entries if entry['field'] == 'capability'],
        [('Related facility capability', capability_pattern)], uncertain_context=CATALOG_UNCONFIRMED_CONTEXT,
    )
    return _score_support(
        service, negative, equipment, capabilities, service_pattern,
        service_points=SERVICE_MAX if stated else 24 if service else 0,
        mixed_branch=mixed_branch, other_cities=other_cities,
    )


def extract_evidence(row, tags, flags, *, catalog_patterns=None, catalog_quotes=None):
    entries = list(_entries(row))
    clinical_text = ' '.join(item['quote'] for item in entries)
    mixed_branch = 'multiple_city_names_in_text_review' in flags
    cities = json.loads(row['city_names_mentioned_in_text'])
    local_city = row['address_city_search']
    if cities and local_city not in cities:
        mixed_branch = True
    other_city_names = [re.escape(city) for city in cities if city and city != local_city]
    other_cities = re.compile(r'\b(?:' + '|'.join(other_city_names) + r')\b', re.I) if other_city_names else None

    support = {}
    for name, (family, pattern, _, _, _) in RULES.items():
        if family in tags or re.search(pattern, clinical_text, re.I):
            result = _evaluate(entries, name, family in tags, mixed_branch, other_cities)
            if result is not None:
                support[name] = result
    # Broad dialysis searches use one documented modality's checklist at a time.
    if 'Dialysis' in support and support['Dialysis']['status'] != 'Conflicting procedure claims':
        modalities = [name for name in RULES if name != 'Dialysis' and RULES[name][0] == 'Dialysis'
                      and name in support and support[name]['service']]
        best = max(modalities, key=lambda name: support[name]['points'], default=None)
        if best and support[best]['points'] > support['Dialysis']['points']:
            support['Dialysis'] = {**support[best], 'checklist': best}
    if catalog_patterns:
        for name in tags:
            if name in catalog_patterns:
                support[name] = _catalog_support(
                    entries, catalog_quotes[name], catalog_patterns[name], mixed_branch, other_cities,
                )
    mention = next((item for item in entries if NABH.search(item['quote'])), {})
    return support, _excerpt(mention, NABH.pattern)
