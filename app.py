import csv
import gzip
import io
import json
import re
from pathlib import Path

import streamlit as st

from ranking import rank_facilities


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
def load_data():
    with gzip.open(DATA / 'facilities.json.gz', 'rt', encoding='utf-8') as file:
        facilities = json.load(file)
    pins = json.loads((DATA / 'pin_centroids.json').read_text(encoding='utf-8'))
    symptoms = json.loads((DATA / 'symptoms.json').read_text(encoding='utf-8'))
    procedures = json.loads((DATA / 'procedures.json').read_text(encoding='utf-8'))['procedures']
    return facilities, pins, symptoms, procedures


facilities, pins, symptoms, procedures = load_data()
title_column, language_column = st.columns([4, 1])
with title_column:
    st.title(t('Find the right care'))
    st.caption(t('India facility finder · prototype using extracted facility claims'))
with language_column:
    st.selectbox(t('Language'), list(LANGUAGE_LABELS), format_func=lambda code: LANGUAGE_LABELS[code], key='language')

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
        st.caption(t('PIN location estimated from {count} facility records, not an official postal centroid.', count=point[2]))
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
        maximum = max(selected.values(), key=lambda value: {'Low': 1, 'Medium': 2, 'High': 3}[value])
        st.caption(t('Maximum reported severity: {severity}', severity=t(maximum)))

    st.subheader(t('Optional procedure'))
    procedure_counts = {item['name']: item['facility_count'] for item in procedures}
    selected_procedure = st.selectbox(
        t('Procedure you need'),
        ['No specific procedure'] + list(procedure_counts),
        format_func=t,
        help=t('Selecting a procedure excludes every facility without a matching extracted procedure claim.'),
        key='procedure',
    )
    if selected_procedure == 'No specific procedure':
        selected_procedure = None
    else:
        st.caption(t(
            '{count} facilities in this dataset have a matching extracted claim. Confirm the service directly before visiting.',
            count=f'{procedure_counts[selected_procedure]:,}',
        ))

st.info(t('This is a prototype. Facility details come from web text and may be incomplete or wrong. Call the facility to confirm its services before visiting. This app does not provide medical advice.'))
search = st.button(t('Find facilities'), type='primary', disabled=not location)
if search:
    st.session_state['show_results'] = True
if not st.session_state.get('show_results') or not location:
    st.stop()

ranked = rank_facilities(facilities, location[0], location[1], selected, symptoms, selected_procedure)
if selected_procedure:
    st.subheader(t('{count} facilities within 100 miles with a {procedure} claim', count=f'{len(ranked):,}', procedure=t(selected_procedure)))
else:
    st.subheader(t('{count} facilities within 100 miles', count=f'{len(ranked):,}'))
if not ranked:
    if selected_procedure:
        st.warning(t('No eligible facilities with the selected extracted procedure claim were found within 100 miles.'))
    else:
        st.warning(t('No eligible facilities were found within 100 miles. This does not mean care is unavailable. Seek local clinical advice if needed.'))
    st.stop()
st.caption(t('Score = distance (0–40) + provisional specialty match (0–40) + level fit (0–20). Higher is ranked first. Facility evidence and weaknesses appear below.'))

table = []
for index, facility in enumerate(ranked, 1):
    why = [t('This facility is about {distance} miles from the location you entered.', distance=facility['distance_miles'])]
    if selected:
        examples = facility['matched_specialties'][:3]
        for symptom, specialty in examples:
            why.append(t(
                'The record lists {specialty}, which may help assess {symptom}.',
                specialty=t(readable_specialty(specialty)),
                symptom=t(symptom),
            ))
        extra = len(facility['matched_specialties']) - len(examples)
        if extra:
            why.append(t('The record also lists possible matches for {count} other selected symptoms.', count=extra))
        if not examples:
            why.append(t('Its listed specialties do not clearly match the symptoms you selected.'))
        maximum = max(selected.values(), key=lambda value: {'Low': 1, 'Medium': 2, 'High': 3}[value])
        favored = {'Low': 'primary', 'Medium': 'secondary', 'High': 'tertiary'}[maximum]
        why.append(t(
            'For symptoms marked {severity}, this app favors {favored} care. This facility is listed as {level} care.',
            severity=t(maximum), favored=t(favored), level=t(facility['level'].lower()),
        ))
    else:
        why.append(t('You did not choose symptoms, so the available facilities are ordered by distance.'))
    if selected_procedure:
        why.append(t('The record mentions {procedure}, the procedure you selected.', procedure=t(selected_procedure)))
    why_text = ' '.join(why)
    weaknesses_text = ' '.join(t(item) for item in facility['weaknesses'])
    table.append({
        t('Rank'): index,
        t('Facility'): facility['name'],
        t('Score / 100'): facility['score'],
        t('Miles'): facility['distance_miles'],
        t('Level'): t(facility['level'].lower()),
        t('Why'): why_text,
        t('Weaknesses'): weaknesses_text,
        t('Address'): facility['address'],
        t('Phone'): facility['phone'],
        t('Email'): facility['email'],
        t('Procedure claim text (original English)'): facility['matched_procedure_evidence'],
        t('Evidence excerpt (original English)'): facility['evidence_excerpt'],
        t('Associated source URLs'): '; '.join(facility['source_urls'][:3]),
        t('ID'): facility['id'],
    })
st.dataframe(table, hide_index=True, use_container_width=True, height=420)
buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
writer.writeheader()
writer.writerows(table)
st.download_button(t('Download all ranked results (CSV)'), buffer.getvalue().encode('utf-8-sig'), 'ranked_facilities.csv', 'text/csv')

st.subheader(t('Evidence for leading results'))
for index, facility in enumerate(ranked[:10], 1):
    heading = t('{rank}. {name} · {score}/100 · {distance} miles', rank=index, name=facility['name'], score=facility['score'], distance=facility['distance_miles'])
    with st.expander(heading):
        first, second, third = st.columns(3)
        first.metric(t('Distance'), f"{facility['distance_score']}/40")
        second.metric(t('Specialties'), f"{facility['specialty_score']}/40")
        third.metric(t('Level fit'), f"{facility['level_score']}/20")
        st.write(f"**{t('Why')}:** {table[index-1][t('Why')]}")
        st.write(f"**{t('Weaknesses')}:**")
        for weakness in facility['weaknesses']:
            st.write('• ' + t(weakness))
        st.write(f"**{t('Address')}:** {facility['address'] or t('Not available')}")
        st.write(f"**{t('Phone')}:** {facility['phone'] or t('Not available')}  |  **{t('Email')}:** {facility['email'] or t('Not available')}")
        if selected_procedure:
            st.write(f"**{t('Matching extracted procedure text (original English):')}** {facility['matched_procedure_evidence']}")
        st.write(f"**{t('Facility text excerpt (original English):')}** {facility['evidence_excerpt'] or t('No excerpt available')}")
        if facility['source_urls']:
            st.write(f"**{t('Associated source URLs (not individually verified):')}**")
            for number, url in enumerate(facility['source_urls'][:3], 1):
                if url.startswith(('https://', 'http://')):
                    st.link_button(t('Source {number}', number=number), url)
        st.caption(f"{t('Record ID:')} {facility['id']}")
