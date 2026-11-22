# Source inputs

The bundled `facility_specialty_symptom_dictionary_merged.json` is the exact
dictionary used to prepare both datasets. SHA-256:

```text
6281dad9e3561d2c833b719f179e939cc6033a07c454c64d2de21563690f902d
```

The original raw input was `facilities_cleaned_full_specialties_standardized.csv`.
It is not bundled in this component. SHA-256:

```text
9491d4d3c388b3148936d2ed76c3d470c8aab4884e26013ae62d79d5db9db638
```

The CSV had 10,000 rows and 61 columns; filtering yielded 5,610 eligible unique
hospital records. The included subsets contain the facility features needed for
training, evaluation and example predictions. Downloading the raw CSV is only
necessary to regenerate those subsets. The root app's compressed facility data
is a separate processed artifact and cannot reproduce the original sampling.

`source_csv` in preparation reports names this unbundled input. `dictionary_source`
is relative to `facilities/specialty`, so reports work on another computer.
Original input hashes, sampled facility identities, captured labels and model
weights are preserved. Source claims have not been independently verified.
