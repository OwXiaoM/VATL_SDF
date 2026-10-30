# Corrected SDF-only population split and individual validation

The authoritative file is
`/leonardo/home/userexternal/xzhang01/data/v1/subject_ids_label_t1_fast.yaml`,
node `mra_atlas_synthseg_label_t1_fast.subject_ids`:
**497 train / 55 val / 0 test**, with no overlap. The original YAML is unchanged.
The 55 subjects are a validation cohort, not a separate final test cohort.

Previous checkpoints trained on all 552 subjects. Preserve those runs for
historical inspection; they are rejected by this validation pipeline. No new
training or GPU evaluation jobs have been submitted.

## Train from scratch

`configs/ablations/02_sdf_only.yaml` is the corrected baseline. It explicitly
selects the YAML `train` list, checks exact split sizes and complete metadata
coverage, and keeps `loss.mra_weight: 0.0`. Its architecture, SDF loss, optimizer,
sampling, seed, schedule and stopping epoch are unchanged from the original 02
configuration. The run name is now `sdf_only_split_corrected`.

The loader rejects missing/null splits, duplicates, overlapping groups, missing
IDs, wrong counts and attempts to use validation subjects for population fitting.
Population training follows train YAML order. `require_fresh_start: true` and
`resume: null` reject any checkpoint initialization/resume by default.

Check the declared population set without training, then submit manually:

```bash
source slurm/environment.sh
python -c 'from mra_sdf_only.config import load_config; from mra_sdf_only.dataset import read_metadata; print(len(read_metadata(load_config("configs/ablations/02_sdf_only.yaml"))))'
sbatch slurm/train_sdf_only_split.sh
```

The preflight must print `497`. Training uses four GPUs and the original
24-hour resource limit, in a new run directory. Existing checkpoints and outputs
are untouched. If a later continuation is necessary, explicitly disable the
fresh-start guard **only for a checkpoint from this corrected run**; resume
validation still requires exact training IDs/order and immutable configuration.
Never initialize from a previous 552-subject checkpoint.

## Fixed validation subjects

Evaluation uses the first N entries of the **YAML val list**, not TSV order,
sorting or random sampling. N=3 is the prefix of N=5:

1. IXI403-Guys-0965
2. IXI037-Guys-0704
3. IXI050-Guys-0711
4. IXI426-IOP-1011
5. IXI118-Guys-0764

`validation_cohort.json` freezes this selection, all 55 validation IDs, metadata
and split hashes, FOV and evaluation settings. It is a selection manifest only:
no inference or performance measurement has been performed with the new model.
Both 3/5 selected-ID files are also saved under `analysis/split_correction/`.

## Evaluate the new checkpoint

The evaluation config defaults to `.subject_ids.val`. Its checkpoint is deliberately
unset: supply the **new 497-subject** checkpoint explicitly. Population training
IDs must exactly match the declared train set and must not overlap any val IDs.

After retraining, replace the checkpoint/output paths below:

```bash
source slurm/environment.sh
eval_checkpoint=/absolute/path/NEW_sdf_only_split_corrected/checkpoint_step_XXXXXXXX.pt

python -m mra_sdf_only.test_individual --checkpoint "$eval_checkpoint" \
  --num-test-subjects 5 --prepare-only \
  --cohort-manifest configs/evaluation/validation_cohort.json \
  --output /absolute/path/new_validation_preflight5

sbatch slurm/evaluate_sdf_only.sh 3 /absolute/path/new_validation3 \
  --checkpoint "$eval_checkpoint" --cohort-manifest configs/evaluation/validation_cohort.json
sbatch slurm/evaluate_sdf_only.sh 5 /absolute/path/new_validation5 \
  --checkpoint "$eval_checkpoint" --cohort-manifest configs/evaluation/validation_cohort.json
```

Outputs require new directories. CLI `--num-test-subjects` is retained for
compatibility; its selected population is explicitly **validation**, recorded
in the split key and manifest. Use the same manifest/settings for future
SDF-only models. No commands are submitted automatically.

Two separate protocols remain unchanged: `zero_shot_atlas` is an age/sex-conditioned
population atlas with no GT fitting; `test_time_sdf_fit` fits only new subject
latent/rigid variables using deterministic observation blocks, with the pretrained
model frozen. Primary scores use disjoint held-out spatial blocks within each
validation subject. Whole-FOV topology/geometry scores remain descriptive because
they include adaptation regions. CSV/JSON scores, summaries, native-grid SDF/masks
and comparison images are exported separately by protocol. See
[PROTOCOL.md](PROTOCOL.md) for unchanged inference and metric definitions.

## Checks

```bash
source slurm/environment.sh
python -m unittest -v tests.test_splits tests.test_individual tests.test_pipeline
bash -n slurm/train_sdf_only_split.sh slurm/evaluate_sdf_only.sh
```

Tests verify the real 497/55 split and YAML ordering, contamination rejection,
null/overlap/count failures, fresh-start guards, unchanged training hyperparameters,
and the individual pipeline's N=3/N=5 reproducibility, frozen parameters and metrics.
GPU fitting remains seeded rather than bitwise deterministic (`grid_sample`
backward); strict CPU regression checks are available. Real corrected-model
performance is pending retraining.
