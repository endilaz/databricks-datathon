import csv
import gzip
import io
import json
import re
from pathlib import Path

import streamlit as st

from procedure_support import (
    CAPABILITY_MAX, CATARACT_REFERENCE, DIALYSIS_REFERENCE, EQUIPMENT_MAX, RULES, SERVICE_MAX, SUPPORT_MAX,
)
from ranking import LEVEL_POINTS, SEVERITY_WEIGHT, rank_facilities


BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
LANGUAGES = json.loads((DATA / 'locales' / 'languages.json').read_text(encoding='utf-8'))
LANGUAGE_LABELS = {item['code']: item['label'] for item in LANGUAGES}
language = st.session_state.get('language', 'en-IN')
if language not in LANGUAGE_LABELS:
    language = 'en-IN'
locale = json.loads((DATA / 'locales' / f'{language}.json').read_text(encoding='utf-8'))


def t(source, **values):
    translated = locale.get(source, source)
    return translated.format(**values) if values else translated


def readable_specialty(value):
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', value).replace('_', ' ').split()
    return ' '.join(word.lower() if word.lower() in {'and', 'or', 'of'} else word.capitalize() for word in words)


st.set_page_config(page_title=t('Find the right care'), page_icon='🏥', layout='wide')


@st.cache_data
def load_data(bundle_version):
    with gzip.open(DATA / 'facilities.json.gz', 'rt', encoding='utf-8') as file:
        facilities = json.load(file)
    pins = json.loads((DATA / 'pin_centroids.json').read_text(encoding='utf-8'))
    symptoms = json.loads((DATA / 'symptoms.json').read_text(encoding='utf-8'))
    procedures = json.loads((DATA / 'procedures.json').read_text(encoding='utf-8'))['procedures']
    cleaning = json.loads((DATA / 'cleaning_summary.json').read_text(encoding='utf-8'))
    return facilities, pins, symptoms, procedures, cleaning


def show_cleaning_methodology(cleaning):
    st.caption(t(
        'Search {searchable} facility records from the Virtue Foundation directory. '
        'We cleaned the data and checked map coordinates and procedure excerpts against the source.',
        searchable=f"{cleaning['searchable_records']:,}",
    ))
    with st.expander(t('Data cleaning and verification'), expanded=False):
        st.caption(t('Source export {file} · Audit as of {date}',
                     file=cleaning['source_file'], date=cleaning['as_of']))
        metrics = st.columns(4)
        metrics[0].metric(t('Cleaned export records'), f"{cleaning['cleaned_records']:,}")
        metrics[1].metric(t('Split records repaired'), f"{cleaning['repaired_records']:,}")
        metrics[2].metric(t('Exact duplicates removed'), f"{cleaning['duplicates_removed']:,}")
        metrics[3].metric(t('Records with review notes'), f"{cleaning['flagged_records']:,}")

        st.markdown(f"**{t('How we cleaned the export')}**")
        st.write(t(
            'The CSV parser read {rows} rows and identified {entries} facility entries. '
            'For split entries, we used JSON structure and coordinate agreement to recover the trailing fields. '
            'We kept the original source lines and flagged the recovered text for review.',
            rows=f"{cleaning['parsed_rows']:,}", entries=f"{cleaning['facility_entries']:,}",
        ))
        st.write(t(
            'Removing exact duplicate rows prevents repeated exports from being counted twice. '
            'Removing blank and repeated list items reduces clutter in the published data. '
            'The original export is retained so reviewers can trace changes back to the source.'
        ))

        st.markdown(f"**{t('Checks completed for this app build')}**")
        st.write(t(
            'The build checks unique record IDs and reconciles {issues} review flags with the audit and review queue. '
            'For searchable records, it checks broad India bounds, compares numeric coordinates with the exported map point, '
            'and checks that procedure excerpts appear in the cleaned source text.',
            issues=f"{cleaning['review_items']:,}",
        ))
        flags = cleaning['flag_counts']
        st.write(t(
            'Common review signals include mismatched source ID and source type lists ({sources}), '
            'multiple cities mentioned in one record ({cities}), shared phone numbers ({phones}), '
            'and missing or incorrectly formatted PINs ({pins}). '
            'Flags point to records worth reviewing and may have valid explanations.',
            sources=f"{flags.get('source_array_length_mismatch', 0):,}",
            cities=f"{flags.get('multiple_city_names_in_text_review', 0):,}",
            phones=f"{flags.get('official_phone_shared_by_5_or_more_records', 0):,}",
            pins=f"{flags.get('invalid_or_missing_pin', 0):,}",
        ))
        st.caption(t(
            'The results include practical visit notes where useful. '
            'The CSV download includes the full review notes and source line numbers.',
        ))
        st.markdown(f"**{t('Using the directory to plan a visit')}**")
        st.write(t(
            'PIN distances are estimates based on facility locations. Source links go back to directory material '
            'that can cover several branches. For advice on what care you need, speak with a clinician.',
        ))
        st.caption(t('Source export SHA-256'))
        st.code(cleaning['source_sha256'], language=None)


def accreditation_label(facility):
    certificate = facility.get('accreditation')
    if certificate:
        status = t('Within certificate dates') if facility['active_certificate'] else t('Outside certificate dates')
        return f"{certificate['issuer']} {certificate['certificate']} · {status}"
    if facility.get('nabh_mention'):
        return t('NABH mentioned in export · unverified')
    return t('Unknown')


def visit_note(facility, procedure):
    support = facility['procedure_support_detail'] or {}
    status = support.get('status')
    if status in ('Conflicting procedure claims', 'Conflicting equipment or capability evidence'):
        return t('The directory gives mixed information about this service or its equipment. Ask what is currently available.')
    if status == 'Procedure excerpt mentions another city':
        return t('The procedure description names another city. Confirm availability at this branch.')
    flags = facility.get('cleaning_flags', [])
    if status == 'Possible branch mix' or (procedure and 'multiple_city_names_in_text_review' in flags):
        return t('The sources mention multiple locations. Confirm which branch offers this service.')
    if status in ('Procedure needs confirmation', 'Procedure variant needs confirmation'):
        return t('Ask whether this branch offers {procedure}.', procedure=t(procedure))
    if procedure and not support and 'planned_service_language_review' in flags:
        return t('The record mentions planned services. Ask whether this procedure is available now.')
    if 'invalid_or_missing_pin' in flags or 'coordinates_outside_broad_india_bounds' in flags:
        return t('Confirm the street address before travelling.')
    if 'official_phone_shared_by_5_or_more_records' in flags:
        return t('This number may reach a central desk. Ask for the listed branch.')
    return ''


def evidence_text(item):
    if not item.get('quote'):
        return ''
    locator = item['field']
    if item['index'] is not None:
        locator += f"[{item['index']}]"
    return f"{locator} · {item['quote']}"


def checklist_text(items):
    return '\n'.join(
        f"{t(item['label'])} · {t(item['status'])}"
        + (f" · {evidence_text(item)}" if item.get('quote') else '')
        for item in items
    )


def show_checklist_excerpts(label, items):
    items_with_excerpts = [item for item in items if item.get('quote')]
    if not items_with_excerpts:
        return
    st.markdown(f"**{t(label)}**")
    availability_notes = {
        'Conflicting or negative text': 'The directory suggests it may be unavailable',
        'Ambiguous or non-current text': 'Current availability is unclear',
    }
    for item in items_with_excerpts:
        item_label = t(item['label'])
        if note := availability_notes.get(item['status']):
            item_label = f'{item_label} · {t(note)}'
        st.caption(item_label)
        st.text(f'“{item["quote"]}”')


def show_procedure_evidence(support):
    if checklist := support.get('checklist'):
        st.caption(t('Checklist for {checklist}', checklist=t(checklist)))
    service_quote = support['service'].get('quote')
    conflict_quote = support['conflict'].get('quote')
    if service_quote or conflict_quote:
        st.markdown(f"**{t('Procedure evidence')}**")
        if service_quote:
            st.text(f'“{service_quote}”')
        if conflict_quote:
            st.caption(t('The directory also reports a conflict'))
            st.text(f'“{conflict_quote}”')

    show_checklist_excerpts('Equipment evidence', support['equipment'])
    if not any(item.get('quote') for item in support['equipment']):
        st.caption(t('The directory has no matching equipment mention for this procedure.'))
    show_checklist_excerpts('Capability evidence', support['capabilities'])


def select_procedure(name):
    st.session_state['procedure'] = name


bundle_version = tuple((DATA / name).stat().st_mtime_ns for name in ('facilities.json.gz', 'cleaning_summary.json'))
facilities, pins, symptoms, procedures, cleaning = load_data(bundle_version)
title_column, language_column = st.columns([4, 1])
with title_column:
    st.title(t('Find the right care'))
with language_column:
    st.selectbox(t('Language'), list(LANGUAGE_LABELS), format_func=lambda code: LANGUAGE_LABELS[code], key='language')

show_cleaning_methodology(cleaning)

left, right = st.columns([1, 2])
with left:
    st.subheader(t('Your location'))
    pin = st.text_input(t('Six-digit PIN'), max_chars=6, placeholder=t('e.g. 110001'), key='pin')
    with st.expander(t('PIN not found? Enter coordinates instead')):
        manual = st.checkbox(t('Use coordinates'), key='manual_coordinates')
        lat = st.number_input(t('Latitude'), min_value=6.0, max_value=38.0, value=20.0, step=0.001, format='%.5f', disabled=not manual, key='latitude')
        lon = st.number_input(t('Longitude'), min_value=68.0, max_value=98.0, value=78.0, step=0.001, format='%.5f', disabled=not manual, key='longitude')
    if manual:
        location = (lat, lon)
        st.caption(t('Using entered coordinates.'))
    elif pin and re.fullmatch(r'[1-9][0-9]{5}', pin) and pin in pins:
        point = pins[pin]
        location = (point[0], point[1])
        st.caption(t('Distances are estimated from facility locations in this PIN.'))
    else:
        location = None
        if pin:
            st.warning(t('This PIN is not in the local index. Enter coordinates to continue.'))

with right:
    st.subheader(t('Symptoms (optional)'))
    categories = st.multiselect(t('Choose symptom categories'), list(symptoms), format_func=t, placeholder=t('Select categories if needed'), key='categories')
    selected = {}
    for category in categories:
        chosen = st.multiselect(t(category), list(symptoms[category]), format_func=t, key='symptoms_' + category)
        for symptom in chosen:
            severity = st.select_slider(
                t('{symptom} severity', symptom=t(symptom)),
                options=['Low', 'Medium', 'High'],
                value='Low',
                format_func=t,
                key='severity_' + category + '_' + symptom,
            )
            selected[(category, symptom)] = severity
    if selected:
        maximum = max(selected.values(), key=SEVERITY_WEIGHT.__getitem__)
        level_points = LEVEL_POINTS[maximum]
        favored_level = max(level_points, key=level_points.__getitem__)
        st.caption(t('Maximum reported severity: {severity}', severity=t(maximum)))

    st.subheader(t('Optional procedure'))
    support_counts = cleaning.get('procedure_support_counts', {})
    procedure_counts = {}
    for item in procedures:
        name = item['name']
        procedure_counts[name] = support_counts.get(name, item['facility_count'])
        if support_counts:
            procedure_counts.update({
                variant: support_counts.get(variant, 0)
                for variant, rule in RULES.items() if variant != name and rule[0] == name
            })
    selected_procedure = st.selectbox(
        t('Procedure you need'),
        ['No specific procedure'] + list(procedure_counts),
        format_func=t,
        help=t('Results consider the procedure claim, related equipment, and supporting capabilities reported in the directory.'),
        key='procedure',
    )
    if support_counts:
        st.caption(t('Compare procedure and equipment details for cataract surgery or dialysis.'))
        cataract_button, dialysis_button = st.columns(2)
        cataract_button.button(
            t('Try phacoemulsification'), on_click=select_procedure, args=('Phacoemulsification',),
        )
        dialysis_button.button(
            t('Try dialysis'), on_click=select_procedure, args=('Dialysis',),
        )
    if selected_procedure == 'No specific procedure':
        selected_procedure = None
    use_support = selected_procedure in support_counts
    if use_support:
        st.caption(t(
            '{count} records in the directory have information related to this procedure.',
            count=f'{procedure_counts[selected_procedure]:,}',
        ))
        if selected_procedure in RULES:
            family = RULES[selected_procedure][0]
            reference = CATARACT_REFERENCE if family == 'Cataract surgery' else DIALYSIS_REFERENCE
            st.link_button(t('Read the checklist research'), reference)
    elif selected_procedure:
        st.caption(t(
            '{count} records in the directory list this procedure.',
            count=f'{procedure_counts[selected_procedure]:,}',
        ))

search = st.button(t('Find facilities'), type='primary', disabled=not location)
if search:
    st.session_state['show_results'] = True
if not st.session_state.get('show_results') or not location:
    st.stop()

ranked = rank_facilities(facilities, location[0], location[1], selected, symptoms, selected_procedure)
if use_support:
    st.subheader(t('{count} candidates within 100 miles for {procedure}', count=f'{len(ranked):,}', procedure=t(selected_procedure)))
elif selected_procedure:
    st.subheader(t('{count} facilities within 100 miles with a {procedure} claim', count=f'{len(ranked):,}', procedure=t(selected_procedure)))
else:
    st.subheader(t('{count} facilities within 100 miles', count=f'{len(ranked):,}'))
if not ranked:
    st.info(t(
        'The directory found no matches within 100 miles. Try a nearby PIN or choose a broader procedure.',
    ) if selected_procedure else t(
        'The directory found no facilities within 100 miles. Try a nearby PIN or enter coordinates.',
    ))
    st.stop()

st.caption(t('Before visiting, call the facility to confirm the service, branch, and appointment availability.'))
with st.expander(t('How results are ranked')):
    if use_support:
        st.write(t(
            'Up to 50 points come from procedure evidence, 20 from distance, 20 from specialty match, and 10 from estimated care level. '
            'Conflicting procedure claims rank last. A relevant branch certificate within its published dates can break a tie.'
        ))
        st.write(t(
            'A procedure claim can earn up to {service} points on its own. Documented equipment adds up to {equipment} points '
            'and capabilities add up to {capability}. An item is unreported when the checklist finds no matching source text.',
            service=SERVICE_MAX, equipment=EQUIPMENT_MAX, capability=CAPABILITY_MAX,
        ))
    else:
        st.write(t(
            'Up to 40 points come from distance, 40 from matching the listed specialties to your search, and 20 from estimated care level. '
            'Without selected symptoms, the closest matches appear first.',
        ))

why_column = t('Why')
table = []
for index, facility in enumerate(ranked, 1):
    why = [t('This facility is about {distance} miles from the location you entered.', distance=facility['distance_miles'])]
    if selected:
        matches = facility['matched_specialties']
        examples = matches[:3]
        for symptom, specialty in examples:
            why.append(t(
                'The record lists {specialty}, which may help assess {symptom}.',
                specialty=t(readable_specialty(specialty)), symptom=t(symptom),
            ))
        extra = len(matches) - len(examples)
        if extra:
            why.append(t('The record also lists possible matches for {count} other selected symptoms.', count=extra))
        if not examples:
            why.append(t('Its listed specialties do not clearly match the symptoms you selected.'))
        why.append(t(
            'For symptoms marked {severity}, this app favors {favored} care. This facility is listed as {level} care.',
            severity=t(maximum), favored=t(favored_level.lower()), level=t(facility['level'].lower()),
        ))
    elif not use_support:
        why.append(t('You did not choose symptoms, so the available facilities are ordered by distance.'))
    support = facility['procedure_support_detail']
    if support and support['status'] == 'Procedure and supporting evidence reported':
        why.append(t('The directory lists {procedure} and supporting details.', procedure=t(selected_procedure)))
    elif selected_procedure and (support is None or support['status'] == 'Procedure reported'):
        why.append(t('The directory lists {procedure}.', procedure=t(selected_procedure)))
    weaknesses_text = ' '.join(t(item) for item in facility['weaknesses'])
    table.append({
        t('Rank'): index,
        t('Facility'): facility['name'],
        t('Score / 100'): facility['score'],
        t('Miles'): facility['distance_miles'],
        **({
            t('Procedure evidence / 50'): facility['procedure_support_score'],
            t('Evidence status'): t(support['status']),
        } if support else {}),
        t('Accreditation'): accreditation_label(facility),
        t('Level'): t(facility['level'].lower()),
        why_column: ' '.join(why),
        t('Before you visit'): visit_note(facility, selected_procedure),
        t('Weaknesses'): weaknesses_text,
        t('Address'): facility['address'],
        t('Phone'): facility['phone'],
        t('Email'): facility['email'],
        **({
            t('Checklist used'): t(support.get('checklist', selected_procedure)),
            t('Equipment evidence'): checklist_text(support['equipment']),
            t('Capability evidence'): checklist_text(support['capabilities']),
            t('Conflicting procedure excerpt'): evidence_text(support['conflict']),
        } if support else {}),
        t('Accreditation scope'): facility.get('accreditation', {}).get('scope', ''),
        t('Accreditation valid through'): facility.get('accreditation', {}).get('valid_until', ''),
        t('Accreditation source'): facility.get('accreditation', {}).get('source', ''),
        t('Procedure claim text (original English)'): facility['matched_procedure_evidence'],
        t('Evidence excerpt (original English)'): facility['evidence_excerpt'],
        t('Associated source URLs'): '; '.join(facility['source_urls'][:3]),
        t('Source record start lines'): ', '.join(map(str, facility['source_line_starts'])),
        t('ID'): facility['id'],
    })
visible_columns = [t('Rank'), t('Facility'), t('Miles'), t('Phone')]
visit_column = t('Before you visit')
if any(row[visit_column] for row in table):
    visible_columns.append(visit_column)
visible_columns.append(t('Score / 100'))
if use_support:
    visible_columns.append(t('Procedure evidence / 50'))
visible_columns.append(why_column)
accreditation_column = t('Accreditation')
if any(row[accreditation_column] != t('Unknown') for row in table):
    visible_columns.append(accreditation_column)
visible_columns.append(t('Address'))
visible_rows = [{column: row[column] for column in visible_columns} for row in table]
for row in visible_rows:
    if row.get(accreditation_column) == t('Unknown'):
        row[accreditation_column] = ''
st.dataframe(visible_rows, hide_index=True, use_container_width=True, height=420)
buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
writer.writeheader()
writer.writerows(table)
st.download_button(
    t('Download all ranked results (CSV)'), buffer.getvalue().encode('utf-8-sig'), 'ranked_facilities.csv', 'text/csv',
    help=t('Includes the full evidence checklist, source references, and audit notes.'),
)

st.subheader(t('Evidence for leading results'))
for index, facility in enumerate(ranked[:10], 1):
    heading = t('{rank}. {name} · {score}/100 · {distance} miles', rank=index, name=facility['name'], score=facility['score'], distance=facility['distance_miles'])
    with st.expander(heading):
        st.write(f"**{t('Address')}** · {facility['address'] or t('Not available')}")
        if facility['phone']:
            st.write(f"**{t('Phone')}** · {facility['phone']}")
        if facility['email']:
            st.write(f"**{t('Email')}** · {facility['email']}")
        note = table[index-1][t('Before you visit')]
        if note:
            st.write(f"**{t('Before you visit')}** · {note}")
        st.write(f"**{why_column}** · {table[index-1][why_column]}")
        metrics = st.columns(4 if use_support else 3)
        metrics[0].metric(t('Distance'), f"{facility['distance_score']:g}/{20 if use_support else 40}")
        metrics[1].metric(t('Specialties'), f"{facility['specialty_score']:g}/{20 if use_support else 40}")
        metrics[2].metric(t('Level fit'), f"{facility['level_score']:g}/{10 if use_support else 20}")
        if use_support:
            metrics[3].metric(t('Procedure evidence'), f"{facility['procedure_support_score']:g}/{SUPPORT_MAX}")
            show_procedure_evidence(facility['procedure_support_detail'])
        if selected_procedure and not use_support and facility['matched_procedure_evidence']:
            st.write(f"**{t('Matching extracted procedure text (original English):')}** {facility['matched_procedure_evidence']}")
        if facility.get('accreditation') or facility.get('nabh_mention'):
            st.markdown(f"**{t('Accreditation')}**")
            st.write(accreditation_label(facility))
        if certificate := facility.get('accreditation'):
            st.caption(t(
                'Scope {scope} · {address} · {start} to {end} · Certificate reviewed {checked}',
                scope=certificate['scope'], address=certificate['certificate_address'],
                start=certificate['valid_from'], end=certificate['valid_until'], checked=certificate['checked_on'],
            ))
            st.link_button(t('Open accreditation certificate'), certificate['source'])
        elif mention := facility.get('nabh_mention'):
            st.text(f'“{mention["quote"]}”')
        if facility['evidence_excerpt']:
            st.write(f"**{t('Facility text excerpt (original English):')}** {facility['evidence_excerpt']}")
        if facility['source_urls']:
            st.write(f"**{t('Directory source links')}**")
            for number, url in enumerate(facility['source_urls'][:3], 1):
                if url.startswith(('https://', 'http://')):
                    st.link_button(t('Source {number}', number=number), url)
        st.caption(t(
            'Source record starts at line(s) {lines} in {file}',
            lines=table[index-1][t('Source record start lines')], file=cleaning['source_file'],
        ))
        st.caption(f"{t('Record ID:')} {facility['id']}")
