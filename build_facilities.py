"""Build the app's facility data from the cleaned export and existing app labels."""

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from procedure_support import catalog_support_patterns, extract_evidence


BASE = Path(__file__).resolve().parent
CONTROL_CHARACTERS = re.compile(r'[\x00-\x1f\x7f]')
CARE_LEVELS = {'Primary', 'Secondary', 'Tertiary'}


def build(cleaned, output):
    audit = json.loads((cleaned / 'cleaning_audit.json').read_text(encoding='utf-8'))
    summary = audit['summary']
    source = cleaned / 'facilities_clean.csv'
    with gzip.open(BASE / 'build_inputs/facility_enrichment.json.gz', 'rt', encoding='utf-8') as file:
        seed = json.load(file)
    enrichment = {record['id']: record for record in seed['records']}
    if len(enrichment) != len(seed['records']):
        raise ValueError('Duplicate IDs in the existing app enrichment')

    accreditations = json.loads((BASE / 'build_inputs/accreditation.json').read_text(encoding='utf-8'))
    if accreditations.keys() - enrichment.keys():
        raise ValueError('An accreditation record has an unknown app facility ID')
    catalog = json.loads((BASE / 'data/procedures.json').read_text(encoding='utf-8'))['procedures']
    catalog_patterns = catalog_support_patterns(item['name'] for item in catalog)
    records = {}
    export_ids = set()
    review_counts = Counter()
    review_items = set()
    procedure_counts = Counter()
    support_counts = Counter()
    csv.field_size_limit(20_000_000)
    with source.open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            facility_id = row['unique_id']
            if not facility_id or facility_id in export_ids:
                raise ValueError(f'Missing or repeated cleaned facility ID: {facility_id!r}')
            export_ids.add(facility_id)
            flags = json.loads(row['quality_flags'])
            source_lines = json.loads(row['source_line_starts'])
            if not source_lines or any(type(line) is not int or line < 2 for line in source_lines):
                raise ValueError(f'Invalid source line pointers for {facility_id}')
            review_counts.update(flags)
            review_items.update((facility_id, flag, tuple(source_lines)) for flag in flags)
            if facility_id not in enrichment:
                continue

            record = enrichment[facility_id].copy()
            lat = float(row['latitude_for_distance'])
            lon = float(row['longitude_for_distance'])
            point = json.loads(row['coordinates'])
            if not (6 <= lat <= 38 and 68 <= lon <= 98
                    and (lat, lon) == (float(row['latitude']), float(row['longitude']))
                    and point == {'type': 'Point', 'coordinates': [lon, lat]}):
                raise ValueError(f'Inconsistent search coordinates for {facility_id}')
            if record['level'] not in CARE_LEVELS:
                raise ValueError(f'Unknown care level for {facility_id}')
            procedure_text = {item.strip() for item in json.loads(row['procedure'])}
            quotes = record['procedure_evidence']
            if set(record['procedures']) != quotes.keys() or any(
                quote.strip() not in procedure_text for quote in quotes.values()
            ):
                raise ValueError(f'Procedure evidence does not match the cleaned record for {facility_id}')
            procedure_counts.update(record['procedures'])
            support, nabh_mention = extract_evidence(
                row, record['procedures'], flags, catalog_patterns=catalog_patterns, catalog_quotes=quotes,
            )
            support_counts.update(support.keys())
            certificate = accreditations.get(facility_id)
            if certificate:
                if (certificate['name'] != row['name'] or certificate['pin'] != row['pin_clean']
                        or certificate['address_fragment'].casefold() not in record['address'].casefold()
                        or date.fromisoformat(certificate['valid_from']) > date.fromisoformat(certificate['valid_until'])):
                    raise ValueError(f'Accreditation source does not match the expected branch for {facility_id}')
                record['accreditation'] = certificate
            if nabh_mention:
                record['nabh_mention'] = nabh_mention
            record.update(
                name=' '.join(CONTROL_CHARACTERS.sub('', row['name']).split()),
                lat=lat,
                lon=lon,
                phone=row['officialPhone'],
                email=row['email'],
                specialties=json.loads(row['specialties']),
                source_urls=json.loads(row['source_urls'])[:5],
                source_line_starts=source_lines,
                cleaning_flags=flags,
                procedure_support=support,
            )
            records[facility_id] = record

    if records.keys() != enrichment.keys():
        raise ValueError(f'Missing {len(enrichment.keys() - records.keys())} existing app IDs in the cleaned export')
    if len(export_ids) != summary['distinct_facilities'] or review_counts != summary['review_flag_counts']:
        raise ValueError('Cleaned CSV totals do not match the cleaning audit')
    with (cleaned / 'review_queue.csv').open(encoding='utf-8', newline='') as file:
        queue_items = [
            (row['unique_id'], row['quality_flag'], tuple(json.loads(row['source_line_starts'])))
            for row in csv.DictReader(file)
        ]
    if len(queue_items) != review_counts.total() or set(queue_items) != review_items:
        raise ValueError('Review queue entries do not match the cleaned records and their source lines')

    if procedure_counts != {item['name']: item['facility_count'] for item in catalog}:
        raise ValueError('Procedure tags do not match the app catalog counts')

    manifest = {
        'source_file': Path(summary['input']).name,
        'source_sha256': summary['input_sha256'],
        'cleaned_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'enrichment_origin_sha256': seed['original_bundle_sha256'],
        'as_of': summary['as_of'],
        'parsed_rows': summary['parsed_rows'],
        'facility_entries': summary['facility_entries'],
        'cleaned_records': len(export_ids),
        'repaired_records': summary['repaired_facilities'],
        'duplicates_removed': summary['exact_duplicates_removed'],
        'review_items': len(queue_items),
        'flagged_records': len({item[0] for item in review_items}),
        'flag_counts': dict(sorted(review_counts.items())),
        'searchable_records': len(records),
        'flagged_search_records': sum(bool(record['cleaning_flags']) for record in records.values()),
        'procedure_support_counts': dict(sorted(support_counts.items())),
        'accreditation_records': len(accreditations),
    }
    payload = json.dumps([records[record_id] for record_id in enrichment], ensure_ascii=False, separators=(',', ':'))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        bundle = staging / 'facilities.json.gz'
        bundle.write_bytes(gzip.compress(payload.encode('utf-8'), mtime=0))
        report = staging / 'cleaning_summary.json'
        report.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        bundle.replace(output / bundle.name)
        report.replace(output / report.name)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cleaned', type=Path, default=BASE.parent / 'cleaned')
    parser.add_argument('--output', type=Path, default=BASE / 'data')
    args = parser.parse_args()
    result = build(args.cleaned, args.output)
    print(f"Packaged {result['searchable_records']:,} search records from {result['cleaned_records']:,} cleaned records.")
