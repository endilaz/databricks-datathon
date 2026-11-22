# Find the right care — Databricks App prototype

This Streamlit app ranks facilities within 100 miles of a PIN-derived location (or entered coordinates). It uses the provided Virtue Foundation export and can run as a Databricks App on Free Edition. It does not diagnose symptoms or verify current service availability.

## Languages

The top-right language menu currently includes English (default), Assamese, Bengali, Bodo, Dogri, Gujarati, and Hindi, each labeled in its own language and English. The app bundles translated interface text, symptom and procedure menus, result explanations, and weakness descriptions. It does not need a translation API key at runtime. Proper names, addresses, links, and original extracted source quotations remain in their source form so users can verify the underlying record.

Sarvam Translate supports all 22 scheduled Indian languages, but the provided API account returned `No credits available` after these six language packs were completed. The remaining languages are deliberately absent from the menu until their translations can be generated and reviewed. Machine-translated health terms should receive native-speaker review before use in patient-facing care navigation.

## Run

1. Keep `app.py`, `ranking.py`, `app.yaml`, `requirements.txt`, and `data/` together.
2. Install `requirements.txt`, then run `streamlit run app.py` locally, or upload/deploy the folder as a Databricks App. The included `app.yaml` uses Databricks' documented Streamlit command.
3. Enter a six-digit PIN or coordinates. Symptoms and a named procedure are both optional; select either, both, or neither.

## Optional procedure filter

The cleaned CSV's `procedure` column contains 156,221 entries and about 89,000 distinct exact strings. Many entries are full sentences or unrelated web-text fragments. This prototype uses a conservative catalog of 52 named procedures and groups clear wording variants (for example, `root canal`, `root canal treatment`, and `RCT`). The catalog covers 4,904 of the 7,882 loaded facility records. These counts describe extracted claims, not verified services.

The 52-term catalog is supplemented by specific options for cataract and dialysis. For those families, the builder screens cleaned procedure, capability, equipment, and description excerpts using the documented methodology. The app shows the matching text, the cleaned field and array position, and whether each checklist item is reported, unreported, ambiguous, or contradicted. For other procedures, a matching catalog tag is required and the app shows its matching excerpt.

Procedure matching deliberately skips obvious negations and research/training text, but the source may still include claims for another facility. The list is incomplete and a missing match does not prove that a facility cannot perform a procedure. Verify the procedure, current availability, and appropriate referral directly with the facility before using this for real care decisions.

## Scoring

Only facilities with internally consistent candidate coordinates and a provisional primary, secondary, or tertiary label are included. The 100-mile radius is strict.

- Distance, 0–40: `40 × (1 − distance_miles/100)`.
- Specialty, 0–40: interim exact-code rule. Each selected symptom has weight 1/2/3 for Low/Medium/High; a facility gets full credit for the first mapped specialty, 75% for another mapped specialty, and zero otherwise. The weighted mean is multiplied by 40. Replace `specialty_points()` in `ranking.py` with the teammate's model while preserving the 0–40 output.
- Level, 0–20: based on the maximum reported severity:

| Maximum severity | Primary | Secondary | Tertiary |
| --- | ---: | ---: | ---: |
| Low | 20 | 12 | 6 |
| Medium | 8 | 20 | 14 |
| High | 2 | 12 | 20 |

For cataract and dialysis queries, the score is **50 points for procedure evidence, 20 for distance, 20 for specialty fit, and 10 for level fit**. A stated service earns 30 evidence points and a procedure listing earns 24. Documented equipment can add up to 12 and capabilities up to 8. Unreported equipment or capabilities leave the service points available. A broad dialysis search can use one documented modality's checklist.

A city conflict elsewhere in the record caps evidence at 22 points. If the matching claim itself names another city, the cap is 10. Equipment-only leads are capped at 10. Explicitly negative or conflicting equipment or capability evidence caps the score at 20. Conflicting service claims score zero on evidence and sort after other candidates. These are ranking rules for the prototype's source evidence. Current availability and clinical readiness require independent checks.

Without symptoms, specialty and level components award equal points to every candidate. Distance orders ordinary searches without symptoms. Procedure evidence and distance both influence reviewed cataract and dialysis searches.

The score is a project rubric, not a clinical probability. High self-reported severity alone does not prove tertiary care is needed.

## Evidence and limitations

Search results prioritize contact details, distance, and the reason a facility matched. A short **Before you visit** note appears when a conflicting service claim, branch ambiguity, address issue, or shared phone number could affect a visit. The leading result cards show supporting excerpts and directory links. The CSV includes the full evidence checklist, audit notes, source identifiers, and line numbers.

The cleaning checks cover internal consistency and traceability. Current availability and the branch providing a service should be confirmed directly with the facility.

The **Data cleaning and verification** panel reports the source export date and checksum, 30 repairs, 11 exact duplicate removals, and 7,681 review items across 6,643 of the 9,989 cleaned records. Result details and downloads include source CSV line numbers.

Accreditation is shown as a branch-matched certificate on file, an unverified NABH mention in the export, or unknown. The two manually matched certificates include scope, validity dates and issuing-source links. A currently dated certificate with relevant scope breaks ties between equal scores with equal procedure evidence. The app uses published dates and has no live revocation feed. Certificate scope supports the stated service category and still leaves specific technique and availability to be checked.

The PIN index is a median of candidate facility coordinates, **not** an official PIN centroid. It covers 2,797 PINs in the supplied records, so it does not yet cover all of India. Add an authoritative India Post PIN reference for national coverage. The source data contains no dependable real-time opening hours, emergency readiness, or service availability.

## Rebuild the app data

From the full datathon workspace, run `python3.12 india_care_app/build_facilities.py`. The builder reads `cleaned/facilities_clean.csv`, `cleaned/cleaning_audit.json`, and `cleaned/review_queue.csv`. It joins the cleaned data to the saved app enrichment by facility ID, then regenerates `data/facilities.json.gz` and `data/cleaning_summary.json`. The deployment ZIP contains these generated files and runs without the raw CSV.

The existing enrichment supplies postal display fields, provisional care levels, procedure tags and excerpts. The builder checks that every enriched record has a unique cleaned ID, in-bounds and internally consistent coordinates, and procedure excerpts that match the cleaned procedure entries after trimming whitespace. It reconciles the full export's review queue and the existing procedure catalog counts. It also applies `procedure_support.py` and validates each curated accreditation record against the branch name, address and PIN before packaging. The 2,107 other cleaned records are outside the current search subset.

Sources: [project brief](https://docs.google.com/document/d/e/2PACX-1vQxhDNQmgZhPgjG0M6LCb_UXNV8CJryqBNQQuqWfIg_dm40ZU1AYAXzZKViC6haFoCmLNiRVfUSBMjf/pub), [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/tutorial-streamlit), [Sarvam Translate](https://docs.sarvam.ai/api/api-guides-tutorials/models/sarvam-translate).
