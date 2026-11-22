# Symptom–hospital specialty matching

Train a model to score a hospital's **listed specialty fit** for a symptom, then rank
the candidate hospitals. The current model combines weighted coverage with a learned
correction. The labels are provisional AI judgments; facility claims are unverified.

**Already trained:** the selected refined checkpoint is
`runs/architecture_refinement/semantic_residual_seed_44/model.pt`.
You can verify it or generate predictions immediately after setup.

- [Setup](#setup)
- [Train a model](#train-a-model)
- [Find saved models](#find-saved-models)
- [Verify a model](#verify-a-model)
- [Generate a ranked prediction](#generate-a-ranked-prediction)
- [Dataset and architecture](#dataset-and-architecture)

## Setup

From the repository root, open this component's directory:

```bash
cd facilities/specialty
```

If `.venv` is already installed and working, reuse it. For a new environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The commands below use macOS/Linux paths and line continuations. On Windows,
replace `.venv/bin/python` with `.venv\Scripts\python.exe` and put each multiline
command on one line, removing the trailing backslashes. The current trainers run on **CPU**; installing a
CUDA version of PyTorch alone does not move this code to the GPU.

## Train a model

### Recommended: train the refined architecture with validation monitoring

This trains the refined model and two controls using one seed. Each run evaluates
validation loss after every epoch and saves its best checkpoint.

```bash
.venv/bin/python -m ranker.refine_architecture \
  --root data/pairwise_v2 \
  --original-checkpoint runs/direct_pairs/model.pt \
  --seeds 42 \
  --max-epochs 25 \
  --patience 4 \
  --out runs/refinement_run_02
```

Your new refined model will be saved at:

```text
runs/refinement_run_02/semantic_residual_seed_42/model.pt
```

Its epoch history and validation results will be saved at:

```text
runs/refinement_run_02/semantic_residual_seed_42/training_run.json
```

How this command behaves:

- It trains **three architectures**: `original`, `features_only`, and `semantic_residual`.
  This command currently has no option to train only the refined architecture.
- `--seeds 42` runs each architecture once. Use `--seeds 42 43 44` to reproduce the
  three-seed comparison: nine training runs in total.
- `--max-epochs 25` sets the maximum. Early stopping starts after at least 10 epochs,
  with patience 4 and a minimum validation-loss improvement of 0.0001.
- Each `model.pt` contains the weights from that run's **lowest validation-loss epoch**,
  which may be earlier than the final epoch printed.
- `comparison.json` in the output root summarizes all runs. Compare validation losses
  there when choosing between runs; each run has already selected its best epoch.
- The original checkpoint is a reference for diagnostics. The new models start
  **from scratch**; this command does not resume or extend the old checkpoint.

`runs/direct_pairs/model.pt` already exists in this project. On a fresh checkout where
it is missing, create that reference checkpoint with the baseline command below first.

### Baseline: train the original MLP on direct comparisons

```bash
.venv/bin/python -m ranker.train_pairs \
  --root data/pairwise_v2 \
  --epochs 10 \
  --out runs/direct_pairs_run_02
```

This saves a **final-epoch** checkpoint and training-loss history. It does not perform
per-epoch validation or early stopping. If using this newly created checkpoint as
the reference above, set:

```text
--original-checkpoint runs/direct_pairs_run_02/model.pt
```

Both trainers start a fresh run. They refuse to overwrite a nonempty output folder.
Choose a new `--out` name for each experiment. Increasing `--epochs` in a new command
does not resume an existing model.

### Legacy: LambdaLoss with the original graded labels

The smaller, original dataset has 0–3 relevance grades and supports NDCG-weighted
LambdaRank training:

```bash
.venv/bin/python -m ranker.train \
  --data data/prepared \
  --labels data/labeled/labels.jsonl \
  --loss lambda --k 5 --epochs 10 \
  --out runs/graded_lambda_run_02
```

The expanded `data/pairwise_v2` dataset has **direct pair preferences and ties**.
Use the refined trainer or `train_pairs` for it. Those use pairwise logistic loss;
they do not invent relevance grades for LambdaLoss.

## Find saved models

| Location | Contents |
|---|---|
| `runs/direct_pairs/model.pt` | Your original MLP after 10 epochs |
| `runs/architecture_refinement/semantic_residual_seed_44/model.pt` | Currently selected refined checkpoint, best epoch 3 |
| `runs/architecture_refinement/semantic_residual_seed_42/model.pt` | Refined model from seed 42 |
| `runs/architecture_refinement/semantic_residual_seed_43/model.pt` | Refined model from seed 43 |
| `runs/architecture_refinement/comparison.json` | Existing nine-run comparison and selection details |
| `runs/refinement_run_02/<architecture>_seed_42/model.pt` | Checkpoints produced by the recommended training command above |
| Beside each model: `training_run.json` | Training settings, loss history and, for refined runs, validation history and best epoch |

A `.pt` file contains the saved model weights and configuration. Use its path with
`--checkpoint` when verifying or predicting. The original training run and nine-run
architecture comparison, including their checkpoints, are included in this repository.
New experiment outputs under `runs/` are ignored by default; explicitly add a new
run if you want to share it.

## Verify a model

There are three separate checks: software checks, held-out label agreement, and
inspection of ranked outputs. None of these independently verifies hospital services.

### 1. Check that the implementation works

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The current suite contains **16 tests** covering ranking-loss mathematics, gradient
behavior, label/split checks, model loading and the refined score decomposition.
These tests check the implementation, not recommendation quality.

### 2. Measure agreement on training and validation data

Verify the selected refined checkpoint:

```bash
.venv/bin/python -m ranker.evaluate_pairs \
  --root data/pairwise_v2 \
  --checkpoint runs/architecture_refinement/semantic_residual_seed_44/model.pt \
  --splits train validation \
  --out runs/verify_selected_01
```

To verify your new run, replace only the checkpoint and output paths:

```bash
.venv/bin/python -m ranker.evaluate_pairs \
  --root data/pairwise_v2 \
  --checkpoint runs/refinement_run_02/semantic_residual_seed_42/model.pt \
  --splits train validation \
  --out runs/verify_refinement_run_02
```

The evaluator prints the metrics and creates:

| Output file | What to inspect |
|---|---|
| `metrics.json` | Model agreement, validation loss and coverage baselines |
| `pair_predictions.csv` | Individual A/B comparisons, AI labels, model score differences and baseline results |
| `pair_predictions.jsonl` | The same comparisons in machine-readable form |

Evaluation does not update the model's weights. It also refuses to overwrite a
nonempty output folder, so use a fresh `--out` path when repeating it.

In `metrics.json`, look at:

- `metrics.validation.mlp.pair_agreement`: agreement with strict AI preferences.
- `metrics.validation.idf_coverage.pair_agreement`: the weighted-coverage baseline.
- `metrics.validation.mean_query_weighted_logistic_loss`: lower is better for this
  objective; this includes teacher-tied pairs at the configured tie weight.
- `metrics.validation.tie_mean_abs_preference_deviation_from_half`: how far the
  model separates pairs the teacher considered equal; lower is better.

Strict agreement excludes teacher ties and abstentions. If a model or baseline
predicts an exact tie on a strict comparison, it receives half credit. The current
validation set has 127 strict comparisons across 20 scenarios; it also has 121
teacher-tied comparisons. These measure agreement with provisional AI labels,
not independent clinical accuracy.

### 3. Reserve the test split for the final selected model

Training commands do not evaluate the test split. After choosing the architecture
and training settings, run a final test explicitly:

```bash
.venv/bin/python -m ranker.evaluate_pairs \
  --root data/pairwise_v2 \
  --checkpoint runs/architecture_refinement/semantic_residual_seed_44/model.pt \
  --splits test \
  --out runs/final_test_selected_01
```

The test set was still untouched when this README was updated. Repeated symptom
wording stays in one split; hospital identities can recur across splits. This is
a held-out-symptom evaluation, not an unseen-hospital evaluation.

## Generate a ranked prediction

Run one prepared symptom scenario through the selected model:

```bash
.venv/bin/python -m ranker.predict \
  --data data/pairwise_v2/prepared \
  --checkpoint runs/architecture_refinement/semantic_residual_seed_44/model.pt \
  --scenario scenario_002 \
  --out runs/prediction_selected_01.json
```

`scenario_002` is **Unusual sleepiness** in the **Common child concerns** category.
Other scenario IDs and their symptoms are listed in
`data/pairwise_v2/prepared/scenarios.jsonl`. This command ranks that scenario's
prepared candidate hospitals; it is not yet a free-text or dropdown application.

The output JSON contains ranked hospitals, raw scores, listed specialty matches
and reason categories. The refined model also provides `score_components`:

```text
score = positive_scale × (weighted_coverage + learned_correction)
```

The score is unbounded and has not been mapped to the eventual 0–40 scale. Match
categories describe supplied records; they are not causal explanations of every
neural adjustment. Severity, location and care-level modeling are outside this version.

## Dataset and architecture

- [Full model layouts, refinement rationale and experiment results](ARCHITECTURE.md)
- [Expanded pairwise dataset, labeling method and reproduction commands](data/pairwise_v2/README.md)
- [Readable expanded training comparisons](data/pairwise_v2/labeled/pair_labels.csv)
- [Captured labeling examples](data/pairwise_v2/label_examples.json)

The expanded dataset covers 148 symptom entries and a 400-hospital pool, with
1,776 direct judgments: 894 preferences, 866 ties and 16 abstentions. Every scenario
contains one symptom; learned interactions between multiple symptoms remain untested.
The original graded batch remains under `data/prepared` and `data/labeled`.

## Repository contents and integration

This folder is a self-contained specialty-model component. The existing Streamlit
app at the repository root still uses its own specialty scoring rule; uploading this
component does not change app behavior. Its raw ranking scores still need a defined
0–40 conversion before they can replace the app's specialty points.

| Path | Purpose |
|---|---|
| `ranker/` | Feature encoding, models, ranking losses, training, evaluation and prediction |
| `tests/` | Implementation and prepared-data checks |
| `data/pairwise_v2/` | Expanded candidate subset, AI prompts, annotations and pairwise labels |
| `data/prepared/`, `data/labeled/` | Original smaller graded dataset |
| `data/source/` | Exact symptom/specialty dictionary and original CSV provenance |
| `runs/` | Saved checkpoints, training logs and existing validation results |
| `PACKAGE_MANIFEST.json` | SHA-256 checksums of the packaged files |

The full original facility CSV is not needed to train, verify or predict using
the bundled subsets. It is not included here; its filename and SHA-256 are recorded
in `data/source/README.md`. Only rebuilding the subsets requires that original CSV.
Preparation reports use paths relative to this component directory.
Preparation and label-import reports record their stage's historical status;
current training and validation results are under `runs/architecture_refinement/`.

Check the packaged files have not changed, from `facilities/specialty`:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("PACKAGE_MANIFEST.json").read_text())
for entry in manifest["files"]:
    path = Path(entry["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise SystemExit(f"Checksum mismatch: {path}")
print(f"Verified {len(manifest['files'])} packaged files")
PY
```
