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

The cleaned CSV's `procedure` column contains 163,791 nonempty entries and about 89,000 distinct exact strings. Many entries are full sentences or unrelated web-text fragments. This prototype uses a conservative catalog of 52 named procedures and groups clear wording variants (for example, `root canal`, `root canal treatment`, and `RCT`). The catalog covers 4,904 of the 7,882 loaded facility records. These counts describe extracted claims, not verified services.

When a procedure is selected, `rank_facilities()` excludes facilities without a matching catalog tag **before** calculating the ranking. It shows the exact extracted text used for each match. The procedure is a strict eligibility filter and does not add points to the 100-point score. If no procedure is selected, ranking works as before.

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

If no symptoms are selected, every facility receives 40/40 for specialty fit and 20/20 for level fit because there is no symptom or severity to compare. After any procedure filter, distance determines the order. The score and its component metrics remain visible, but the plain-language “Why” explanation contains no scores or arrow notation.

The score is a project rubric, not a clinical probability. High self-reported severity alone does not prove tertiary care is needed.

## Evidence and limitations

Each result shows its score breakdown, matched extracted specialty names, matching procedure text when selected, source-text excerpt, associated source URLs, address, phone, email, and data weaknesses. Source URLs and text have not been individually verified; some source records merge different facilities. The app therefore labels claims as provisional and should not be used for live patient routing until facilities and clinical rules are validated.

The PIN index is a median of candidate facility coordinates, **not** an official PIN centroid. It covers 2,797 PINs in the supplied records, so it does not yet cover all of India. Add an authoritative India Post PIN reference for national coverage. The source data contains no dependable real-time opening hours, emergency readiness, or service availability.

Sources: [project brief](https://docs.google.com/document/d/e/2PACX-1vQxhDNQmgZhPgjG0M6LCb_UXNV8CJryqBNQQuqWfIg_dm40ZU1AYAXzZKViC6haFoCmLNiRVfUSBMjf/pub), [Databricks Apps](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/tutorial-streamlit), [Sarvam Translate](https://docs.sarvam.ai/api/api-guides-tutorials/models/sarvam-translate).
