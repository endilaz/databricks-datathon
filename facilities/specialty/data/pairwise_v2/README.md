# Expanded direct pairwise training dataset

This is an expanded, provisionally AI-labeled specialty-matching dataset. It uses
all 148 symptom entries exactly as supplied. There are no invented symptom
combinations, patient details or severity inputs. The original 28-symptom graded
dataset is preserved separately as a baseline.

## What is in this batch

- 400 hospital records sampled from 5,610 eligible deduplicated hospital records.
- All 148 symptom entries across 14 categories; 142 distinct normalized wordings.
- Twelve candidates per symptom, drawn from stronger dictionary overlap, weaker overlap,
  and random records. Sampling uses dictionary overlap; labels are separate AI judgments.
- 385 of the 400 sampled hospitals actually occur in the candidate sets.
- Twelve direct comparisons per symptom: 1,776 primary judgments.
- 894 strict preferences, 866 ties and 16 identity-related abstentions.
- 104 training, 21 validation and 23 test scenarios. Repeated symptom wording stays in
  one split across categories. Facility identities can recur across splits.
- 626 strict training preferences, 127 validation preferences and 141 test preferences.
- 96 training scenarios contain a strict preference, compared with 18 in the baseline.

The pool is sampled across reported state, review priority and specialty-count strata
with seed 42. These are source attributes, not verified geography or quality ratings.
Specialty lists are deduplicated and use the user's alias map. Missing listings mean
unlisted, not proven absent. All facility claims remain unverified.

## How the direct labels were made

The AI in the Codex chat (`gpt-6.1-sol`) read the scenario and complete specialty
claims for the blinded candidates. A lossless character codebook shortened common
specialty names during display; other names were displayed verbatim. Candidate names
were not used for judging. There were no external API calls.

For twelve shuffled candidates, the comparison graph is a connected cycle:
0 vs 1, 1 vs 2, ..., 10 vs 11, 11 vs 0. Each candidate participates twice. This gives
a connected set of comparisons without claiming to observe every possible pair or
to recover an exact total ranking. Close ties are preserved.

For each pair the AI explicitly chose A, B, T (tie) or U (abstain). These labels
are not computed from the earlier 0–3 grades, dictionary weights, specialty frequency,
or a model score. They describe conditional listed-specialty relevance, not hospital
quality, live service availability, clinical necessity or a referral instruction.

The captured annotation strings contain twelve explicit outcomes in the fixed pair
order. The importer expands them to JSONL and CSV. Rationale is captured per symptom
scenario; a separate medical rationale and cited specialty list were not authored for
each pair. The readable CSV includes both full specialty lists for audit.

## Order spot checks and correction

Twenty-four comparisons were re-presented with A/B reversed to the same interactive
annotator. These are not independent or blinded checks, and their agreement is not
an accuracy estimate. One disagreement led to reinspection of a skin-sore record:
Dermatology was not actually listed by candidate 6. Two adjacent comparisons were
corrected to ties. Initial decisions and the correction are retained in batch 04.

This is weak supervision with no independent clinical review. Ties are informative
about lack of a justified preference; they were not broken to increase sample counts.

## Files

- `prepared/`: facilities, source symptom vocabulary, candidate scenarios, frequency
  features, source hashes and sampling report. Generic pointwise response templates
  from the shared preparation helper are unused in this direct-pairwise protocol.
- `pair_index.jsonl`: identifiers and A/B assignments for all 1,776 pairs.
- `pair_prompts.jsonl`: full pair inputs and the comparison rubric.
- `annotations/batch_*.json`: explicit captured AI decisions and scenario notes.
- `display_codebook.json`: the lossless display abbreviation key.
- `teacher_run.json`: provenance, input hashes and annotation limitations.
- `label_examples.json`: readable comparison examples, including the corrected case.
- `labeled/pair_labels.csv`: readable labels, symptom names and both hospital records' specialty lists.
- `labeled/pair_labels.jsonl`: training targets and traceable pair identifiers.
- `labeled/preferences.jsonl`: strict preferences only, without inferred relevance grades.
- `labeled/order_checks.jsonl`: initial and reversed judgments, plus final outcomes.
- `labeled/label_report.json`: counts, coverage and status.

## Training support

Run from the project directory, after installing the existing requirements:

```bash
.venv/bin/python -m ranker.train_pairs --root data/pairwise_v2 --epochs 10 --out runs/pairwise_quickstart
```

The original MLP and refined architecture have been trained on these labels;
see the component's [training and verification guide](../../README.md) and
[architecture comparison](../../ARCHITECTURE.md). The command above scores each
candidate with the shared matching MLP and applies a logistic preference loss:

`BCEWithLogits(score_A - score_B, target_A)`

Targets are 1 for prefer A, 0 for prefer B, and 0.5 for a tie. Abstentions are excluded.
A tie encourages equal scores. `--tie-weight` defaults to 0.25 as an initial adjustable
hyperparameter, not a confidence estimate. Use 1 for equal tie weighting or 0 to omit
ties. Losses are averaged within each scenario, then training steps traverse scenarios.
Validation/test labels are not used for training. No absolute 0–40 scale is learned.

Direct preferences alone do not supply graded gains for NDCG-weighted LambdaLoss.
The original graded baseline remains usable with `ranker.train --loss lambda`.
An exact overall hospital order is not fabricated from sparse pairwise wins.

## Reproduction

Run from `facilities/specialty`. The exact dictionary is bundled under `data/source`.
To rebuild the subset, obtain the original CSV listed in
[`data/source/README.md`](../source/README.md) and replace the example absolute path
below with its location. It is not needed for training from the bundled data.
Use a new empty output root:

```bash
.venv/bin/python -m ranker.prepare \
  --csv /absolute/path/to/facilities_cleaned_full_specialties_standardized.csv \
  --dictionary data/source/facility_specialty_symptom_dictionary_merged.json \
  --out data/rebuilt_pairs/prepared --facilities 400 --scenarios 148 \
  --candidates 12 --candidate-mode contrastive
.venv/bin/python -m ranker.pairwise prepare --root data/rebuilt_pairs
```

That reproduces the inputs. Copying the captured annotation files and running
`ranker.pairwise import` reproduces the labels; it does not rerun an AI teacher.

All 148 symptom IDs are represented in the dataset, but 104 entries are in the training
split. The others are held out. Their symptom-specific one-hot weights will not be
trained; dictionary-based match features can still transfer. This split therefore
tests a difficult symptom generalization setting and does not establish competence
on unseen symptoms. It also cannot test interactions between multiple symptoms.

Existing validation results measure agreement with the AI teacher, rather than
independently verified hospital suitability. The test split remains unevaluated.
More AI labels do not substitute for an independent reviewed test set.
