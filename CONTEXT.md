# Conversation context and restart point

Last updated: 2026-10-30.

## Current decision (overrides historical baseline comparisons below)

**02_sdf_only is the baseline for every subsequent experiment.** Do not continue
investigating intensity supervision or comparing intensity-related ablations.
Current objective: enforce the authoritative **497 train / 55 validation** split,
retrain SDF-only from scratch, and evaluate the first 3/5 validation subjects in
YAML order. Split handling is corrected and validated; launch commands are ready.
No jobs were submitted. Existing 552-subject checkpoints remain preserved and
cannot initialize this run or provide held-out validation results. A new external
cohort is not required to use the existing 55 validation subjects after proper
train-only retraining. See `configs/evaluation/README.md`.

## User requests and completed work

1. User reported that training/testing had finished, MRA looked very smooth,
   and vessel masks looked acceptable but fragile. They asked for explanations.
2. Inspected the run settings, training/inference/loss/model code, saved atlas,
   and logs; presented hypotheses and preliminary measurements below.
3. User requested controlled scripts to train ablations one by one and an
   SDF-only supervision setting. Created six full configs, a Slurm launcher,
   diagnostic export script, documentation, and a gradient regression test.
4. User asked to save the conversation as agent/context Markdown files so a
   future session can resume. This file is a structured handoff, not a verbatim
   transcript. `AGENTS.md` points future agents here.

## Original run

```
/leonardo_scratch/large/userexternal/xzhang01/Cinema_output/mra_sdf_only/20261024_072629_915052_baseline_61956329
```

- Saved config: `config_final.yaml`; logs: `losses.jsonl`, sampling logs.
- Metadata: `training_metadata.tsv`.
- Latest saved atlas found: `atlases/epoch_0189`.
- Latest checkpoint found: `checkpoint_step_00262200.pt`.
- Logs ended at epoch index 193 / step 267690, despite configured 1,000 epochs.
  Why training stopped was not investigated; do not assume convergence or failure.
- Atlas: age 20, sex M; 1 mm isotropic grid, shape (240,240,120).
- MRA/SDF weights 1.0/0.1; SDF normalized by 5 mm truncation; pointwise L1.
- 250,000 samples per task per subject; SDF surface/interior/outside fractions
  0.5/0.25/0.25; surface band 2 mm; pools intentionally overlap.
- Latent shape [256,3,3,3]; rigid transforms; four experts, top two selected.
- Metadata source:
  `/leonardo/home/userexternal/xzhang01/data/v1/ixi_metadata_label_t1_fast_sdf.tsv`.

## Findings and interpretation

Established from code:

- MRA and SDF have separate heads on shared features. The vessel mask is obtained
  directly as predicted SDF < 0, not by segmenting the reconstructed MRA.
- Atlas construction decodes an age-weighted mean latent grid. It does not simply
  average subject binary masks or subject SDF predictions. Age weighting selects
  subjects without filtering by sex; sex enters the decoder condition.
- Subject alignment is rigid, not deformable vessel correspondence.
- SDF supervision has no explicit connectivity, centerline, or topology loss.
- MoE uses hard top-k selection and softmax weights over selected experts.

Measurements for latest saved atlas (`epoch_0189`):

| SDF threshold (mm) | Selected voxel count |
| --- | ---: |
| < 0 | 18,332 |
| < 0.1 | 18,787 |
| < 0.25 | 19,631 |
| < 0.5 | 21,542 |

- Exported mask agrees exactly with SDF < 0 (zero mismatched voxels).
- Atlas weights have approximately 34.336 effective subjects, defined as
  1 / sum(weights squared).
- Recent logged normalized SDF L1 was about 0.0355, equivalent to about 0.18 mm
  for the sampled training targets. This is not measured surface displacement
  and not a held-out error.
- An exploratory 26-neighbor component check on `epoch_0009` reported 377
  components, 21,312 vessel voxels, largest component 8,761 voxels (41.1%).
  Later-epoch component/reference checks were interrupted because they ran
  slowly. No latest-atlas connectivity improvement from threshold offsets was
  established. Do not confuse volume sensitivity with demonstrated reconnection.
- No visual image review or quantitative subject-reference evaluation was
  completed in this conversation.

Leading hypotheses, not established causes:

1. Anatomical variation and imperfect correspondence weaken thin branches in
   the decoded mean latent; average latent anatomy need not remain connected.
2. Small SDF sign errors erase narrow bridges when thresholded at zero.
3. The 1 mm grid undersamples thin/oblique vessels.
4. Pointwise SDF regression can fit average distances without preserving topology;
   stronger SDF supervision may help but loss weights alone do not establish
   gradient dominance (gradient diagnostics were disabled).
5. Hard expert selection may introduce local discontinuities.
6. Limited subject latent spatial capacity or incomplete training may contribute.

## Implemented controlled experiments

Full configs in `configs/ablations/`:

| Config stem | Only experimental change from matched baseline |
| --- | --- |
| `00_baseline` | None |
| `01_sdf_weight_1` | loss.sdf_weight: 0.1 -> 1.0 |
| `02_sdf_only` | loss.mra_weight: 1.0 -> 0.0 |
| `03_all_experts` | model.moe_k: 2 -> 4 |
| `04_surface_band_1mm` | sampling.sdf.surface_band_mm: 2 -> 1 |
| `05_latent_grid_5` | model.latent_shape: [256,3,3,3] -> [256,5,5,5] |

All have distinct run names and a common `training.stop_after_epoch: 199`
(200 completed epochs). They retain `training.epochs: 1000` and the original
1,000-epoch scheduler horizon. Other baseline settings are preserved.

New files:

- `slurm/train_ablation_4gpu.sh`: four-GPU, 24-hour Slurm launcher accepting a
  config stem followed by trainer flags, e.g. `--set key=value ...`.
- `slurm/diagnose_vessels.sh`: inside a GPU allocation, exports one training
  subject on its native grid and the same atlas at 1 mm and 0.5 mm.
- `configs/ablations/README.md`: commands, comparisons, resume instructions,
  limitations, and interpretation.
- Added `test_sdf_only_has_no_intensity_supervision` to `tests/test_pipeline.py`.

No production training/loss/model code needed changes. No full training jobs
were submitted. No ablation results exist from this session.

SDF-only semantics:

- MRA intensity contributes zero gradient; SDF weight remains 0.1 to isolate
  removal of MRA supervision; MoE weight stays 0.01.
- Existing MRA loading/sampling and two heads remain. MRA coordinates still
  participate in MoE balancing, and MRA files are required.
- Ignore MRA reconstructions/metrics from this variant: its MRA head is untrained.
- Optional practical follow-up sets SDF weight to 1.0 too, but that makes two
  changes relative to the baseline, so label it separately.

## Validation already completed

- `bash -n` passed for both new shell scripts.
- Loaded all six configs with `load_config`; recursively verified that each
  differs from matched baseline only in its intended key plus run name.
- Passed unit tests:
  - `test_sdf_only_has_no_intensity_supervision`
  - `test_two_heads_and_loss_routing`
- Passed two-step synthetic CPU training/checkpoint checks for SDF-only and
  all-expert routing using a tiny model/dataset.
- SDF-only check confirmed zero MRA weighted loss and positive SDF weighted loss.
- Full-size GPU/DDP execution and full training outcomes are not verified.

## Restart here

First ask for or inspect any new user-provided job/result paths. Do not rerun
completed checks or submit jobs merely because this context exists.

Recommended first diagnostic, before retraining: compare a native individual
reconstruction with the reference SDF/mask, and inspect the atlas at finer
spacing. Good individual fitting with poor atlas connectivity points toward
atlas construction/alignment rather than only training capacity.

Inside a GPU allocation:

```bash
source slurm/environment.sh
bash slurm/diagnose_vessels.sh /absolute/path/to/checkpoint.pt /absolute/path/to/new_diagnostics IXI002-Guys-0828
```

The 0.5 mm grid has eight times as many voxels; account for memory/runtime.
Native subject and atlas grids differ, so do not compare voxel by voxel.

Training commands (user intended to run one at a time, waiting between jobs):

```bash
sbatch slurm/train_ablation_4gpu.sh 00_baseline
sbatch slurm/train_ablation_4gpu.sh 01_sdf_weight_1
sbatch slurm/train_ablation_4gpu.sh 02_sdf_only
sbatch slurm/train_ablation_4gpu.sh 03_all_experts
sbatch slurm/train_ablation_4gpu.sh 04_surface_band_1mm
sbatch slurm/train_ablation_4gpu.sh 05_latent_grid_5
```

Optional stronger SDF-only run:

```bash
sbatch slurm/train_ablation_4gpu.sh 02_sdf_only --set loss.sdf_weight=1.0 run_name=sdf_only_weight1
```

If a 24-hour allocation expires, resume the same variant from its own latest
checkpoint using `--set resume=/absolute/path/to/checkpoint.pt`. Exact resume
requires the same GPU world size and unchanged immutable config sections.
For a full 1,000-epoch run, use `training.stop_after_epoch=null`, consistently
across compared runs. See README for details.

Compare equal completed epochs and matched spacing, not wall time. Assess
individual zero-level Dice, vessel volume, component sizes/counts relative to
reference, and visible branch recovery. Fewer components alone can reflect
false bridges. Threshold offsets are diagnostic, not proof of anatomical
recovery. Repeat promising results with matched additional seeds.

Potential next experiments after these results: atlas alignment/construction
changes if subject fitting is good; separately controlled sign/centerline or
topology supervision if individual fitting remains fragile. These have not
been implemented.

## GitHub publication (2026-10-30)

User requested uploading the code to `OwXiaoM/VATL_SDF` (private). Initialized
a local Git repository with branch `main`. Logs and Python caches are excluded
by `.gitignore`. GitHub app access supports publication; SSH authentication on
this host failed because no local key or agent was available. No training jobs
were submitted and original run outputs were preserved.

## Ablation analysis (2026-10-30)

User reported experiments had run and requested conclusions. Found only variants
00, 01, and 02 in the configured output root. All three Slurm jobs timed out
at about 24 hours (62280714, 62280711, 62280701), ending at epoch indices
194/194/193. Compared epoch 189 atlases and epochs 185–189 training losses.
Configs are controlled, metadata and atlas provenance match.

- Baseline: SDF raw L1 0.03892; 18,842 vessel voxels; 214 components
  (26-neighbor); largest component 58.6%.
- SDF weight 1: L1 0.06266; 18,298 voxels; 282 components; largest 68.2%.
  Mixed topology evidence; higher training errors; still improving over time.
- SDF-only: L1 0.09543; 21,179 voxels; 478 components; largest 48.6%.
  Removing MRA did not help at this budget. This does not establish mechanism.
- Positive thresholds through +0.5 mm did not produce a general connectivity fix.
- All exported masks match SDF < 0; all atlas fields finite.
- No native individual-reference evaluation, held-out evaluation, or finer-grid
  atlas export was performed. No jobs launched. Other variants not found.

Full report: `analysis/ablations_20261030/REPORT.md`; raw reproducible measurements
in `metrics.json` and `measure.py`; mask MIP figure `mask_projections.png`.
Keep baseline as reference; recommend native-subject/reference and finer-grid
diagnostics before attributing remaining fragmentation to a specific mechanism.

## Detailed connectivity audit (2026-10-30; supersedes broad geometry ranking above)

User challenged the inference that baseline preserves major vessels better.
Audited `measure.py` (explicit 6 and 26 adjacency) and `evaluate.py` (default
6 adjacency), with a diagonal-voxel synthetic check. Original 26-neighbor
counts are reproduced; no arithmetic error explains the visual differences.
Training SDF L1 is sampled subject fitting against clipped targets, not atlas
surface error. The earlier statement that removing MRA "did not help" is too
broad if applied to anatomical/major-vessel preservation.

Reused all three epoch-189 SDF<0 masks on verified identical grids. Generated
full 6/26 component distributions, cutoff counts, absolute sizes, CSV positions,
fixed central/peripheral ROI statistics, full-mask Lee skeletons, fixed-anchor
reachability, and cross-prediction spatial correspondence. No training or
production evaluation code changed; no inference, retraining, or jobs submitted.

- Absolute largest components (6 / 26): baseline 10,518 / 11,047; higher SDF
  weight 12,212 / 12,488; SDF-only 9,970 / 10,288.
- Central40 (|x|,|y|<=40 mm, all z) volumes: baseline 12,805; higher weight
  12,167; SDF-only 14,595. SDF-only has sizeable central differences, not only
  tiny peripheral speckles; global >=100 component central occupancy is
  12,258 / 11,424 / 13,593 voxels.
- SDF-only second component (2,399 voxels) overlaps baseline largest by 1,115
  voxels but is disconnected from SDF-only largest under both definitions.
  Nearest voxel centers between those components are 7.35 mm apart.
- All three pass coarse inferior-to-superior and left-to-right anchor tests;
  this is not named-vessel patency. Boundary-touching component volumes differ
  substantially (26-neighbor: 4,739 / 95 / 994 voxels).
- No common registered atlas reference or named-vessel tracing exists in saved
  outputs. Additional structures cannot be called true recovery or false
  positives from prediction disagreement alone. No overall anatomical winner
  is established. Preserve baseline as experimental reference, not proven best.

Report and reproducible artifacts:
`analysis/ablations_20261030/connectivity_audit/REPORT.md`.
Original report now links to this interpretation update.

## Individual SDF evaluation pipeline (2026-10-30)

Dataset/split audit:

- Existing baseline checkpoint:
  `/leonardo_scratch/large/userexternal/xzhang01/Cinema_output/mra_sdf_only/20261027_144056_557977_ablation_02_sdf_only_62280701/checkpoint_step_00262200.pt`.
  Step 262200; position epoch 190, batch 0, inner 0 (190 completed epochs).
- Checkpoint training metadata contains all 552 available SDF metadata IDs.
- `data/v1/subject_ids_label_t1_fast.yaml`, key
  `mra_atlas_synthseg_label_t1_fast.subject_ids`, has train=497, val=55, test=0.
  All 497 train and all 55 val IDs overlap checkpoint training. Other inspected
  v0/v1 subject YAMLs likewise have empty test lists; all inspected metadata
  IDs overlap checkpoint training. Training loader with subject_ids=null used
  the TSV without applying the legacy split YAML.
- Therefore no real held-out N=3/N=5 cohort can be selected. Do not reinterpret
  validation/training subjects as held-out. Need a genuinely disjoint test-ID
  declaration plus accessible native SDF TSV with age and sex.
- Saved audit: `analysis/individual_sdf_baseline/split_audit.json`; actual N=3
  and N=5 prepare-only attempts rejected the empty list, with run.json records
  under `preflight_n3` and `preflight_n5`. No real-subject inference was run.

Implementation:

- `mra_sdf_only/test_individual.py`: explicit split validation, metadata-order
  prefix selection, saved cohort manifest/hashes, separate zero_shot_atlas and
  test_time_sdf_fit protocols, native-grid SDF/mask exports and comparison images.
  Existing native_grid, weighted_latent, condition_vector and rigid_transform
  are reused. No intensity data are loaded or evaluated.
- Zero-shot: age-weighted training mean latent; known age/sex; identity rigid;
  no test-reference fitting. This is a conditioned population atlas.
- Test-time fitting: new external latent/rigid variables; whole pretrained model
  including training latent/rigid tables frozen and hashed before/after. The
  latent_override bypasses the stored rigid table, so the existing physical
  rigid_transform is applied explicitly before interpolation/decoding.
- Coordinate-only hashed spatial blocks define adaptation versus held-out folds;
  one-voxel evaluation guard, fixed subject-ID seeds and saved observation indices.
  Only observation coordinates and SDF values enter adaptation. Fixed steps and
  priors, no held-out early stopping. GT adaptation is labeled test-time fitting,
  not zero-shot. N does not affect selection order, initialization or RNG.
- `mra_sdf_only/individual_metrics.py`: held-out voxel Dice, precision/recall,
  field MAE/clipped MAE/reference-band MAE, boundary-voxel-center ASSD/HD95/surface
  Dice in physical mm; null/status for empty/undefined metrics. Metric clipping
  radius fixed at 5 mm independently of checkpoint training truncation.
- Whole-FOV component counts (6/26), sizes and geometry scores are separately
  labeled descriptive, because they include adaptation regions; topology is not
  evaluated on masks with artificial observation holes. Surface extraction also
  precedes evaluation-support restriction. Mean/sample SD and defined counts are
  saved separately for protocol and metric scope in CSV/JSON.
- `configs/evaluation/sdf_only.yaml`, concise README, detailed PROTOCOL.md;
  `slurm/evaluate_sdf_only.sh` and explicit N=3/N=5 commands. Prepare a 5-subject
  manifest after disjoint data arrive, then use it for both prefixes/future models.
- Strict CPU determinism is default. GPU script uses explicit seeded mode:
  CUDA grid_sample backward is not bitwise deterministic. Strict CUDA fitting
  raises on nondeterministic operations. GPU performance is not validated.

Validation:

- Six synthetic CPU tests passed, including complete N=3/N=5 execution of both
  protocols, exact first-three prediction/metric equality, frozen-state and
  checkpoint preservation, external latent/rigid updates, MoE execution, overlap
  and empty-split rejection, coordinate/rigid mapping, disjoint observations,
  empty-mask and physical-distance metrics, mean/sample SD, partial-FOV NaN
  handling and native-grid export round trips. Shell syntax check passed.
- Saved evidence: `analysis/individual_sdf_baseline/validation.json` and
  `synthetic_validation.txt`. Actual checkpoint SHA256 still matches both
  preflights: `6f7c52af548780a9f5e83abc9be35a17fb62e44e1800be22ef8d3ecf02b16dde`.
- Original training, model, data loader, inference, loss and evaluator code was
  not changed. No population retraining, checkpoint modification or Slurm
  submission occurred. No real held-out baseline scores are available yet.

## Split clarification after user correction (2026-10-30)

User explicitly identified `/leonardo/home/userexternal/xzhang01/data/v1/subject_ids_label_t1_fast.yaml`
as the authoritative split. Re-read that exact file and directly reloaded the
SDF-only checkpoint: train=497, val=55, test=[]; train/val are disjoint.
All 497 train and all 55 val IDs nevertheless occur in checkpoint training
metadata (example IXI403-Guys-0965). Saved data.subject_ids is null. The training
loader did not consume this nested split YAML and therefore used all TSV rows.
This is a split-enforcement error in the experiment setup, not a failure of the
user's train/validation separation. Do not imply that no separated cohort was
defined: it was defined but was included in this checkpoint's training.
Correct future population fitting must use the declared train IDs only. The
existing checkpoint cannot be retroactively made held-out for the 55 val IDs.
No retraining or training-code/config changes were performed in this clarification.

## Split correction implemented (2026-10-30; supersedes previous selection policy)

User explicitly requested fresh SDF-only population training on the existing
497 train subjects and individual evaluation on the existing 55 validation
subjects in **YAML order** (supersedes earlier TSV-order/test-list selection).
The original split file has not been edited.

Changes:

- `mra_sdf_only/splits.py`: shared nested-YAML reader, ordered selection,
  duplicate/overlap/count checks, and strict checkpoint training-ID/order check.
- Population loader requires explicit `data.split: train` and subject YAML;
  null/missing split never falls back to all metadata rows. Requires split union
  to cover the metadata, rejects missing IDs and silently filtered ages, checks
  declared counts, and excludes validation IDs even for explicit dataset frames.
  Small subsets are allowed only for explicit max_steps smoke checks.
- `configs/ablations/02_sdf_only.yaml`: authoritative split path and node, exact
  expected counts 497/55/0, run name `sdf_only_split_corrected`, `resume: null`,
  `require_fresh_start: true`. Architecture, all losses (MRA weight remains 0),
  sampling, optimizer, seed, schedule, atlas settings and stopping epoch are
  identical to original saved SDF-only config. Default generic config.yaml also
  has explicit split handling; use the 02 config for the SDF-only baseline.
- Trainer rejects resumes immediately while fresh-start guard is active. Any
  explicitly enabled later continuation must still match corrected train IDs
  and order, metadata and immutable config. Old 552-subject checkpoints are
  rejected before model construction/state loading.
- Individual evaluation now defaults to `.subject_ids.val`, checks complete
  checkpoint training membership against declared train IDs, and selects the
  first N validation IDs in YAML order. Its default checkpoint is unset: a
  new corrected checkpoint must be supplied explicitly. Inference/adaptation
  and geometry metrics are unchanged.
- `configs/evaluation/validation_cohort.json` freezes the real first five IDs,
  all 55 validation IDs, training IDs, source hashes, FOV and evaluation settings.
  Both N=3/N=5 selections are saved under `analysis/split_correction/`; these
  are selection-only manifests, not performance results.
- First three: IXI403-Guys-0965, IXI037-Guys-0704, IXI050-Guys-0711.
  Fourth/fifth: IXI426-IOP-1011, IXI118-Guys-0764.
- Added `slurm/train_sdf_only_split.sh` using original four-GPU/24-hour resources.
  README documents fresh training, then supplying its new checkpoint to
  `slurm/evaluate_sdf_only.sh` for N=3/N=5 with the fixed validation manifest.

Validation:

- 22 tests passed across test_splits, test_individual and test_pipeline. Tests
  load the actual YAML and metadata: exactly 497 population subjects in train
  YAML order, exactly 55 validation IDs, zero overlap, fixed validation prefixes,
  contaminated-checkpoint and explicit validation-frame rejection. Also test
  null splits, overlaps, duplicates, wrong counts, missing IDs, resume rejection
  before torch.load, and equality of all original training hyperparameters.
- Existing synthetic individual N=3/N=5 pipeline checks pass with YAML ordering.
- Shell syntax checks passed. Original split SHA256 is
  bc84b1dd6d608ad417816458e54f6aa600a1b801d44f976265a801fd6e8b7a9c.
- Original SDF-only checkpoint SHA256 remains
  6f7c52af548780a9f5e83abc9be35a17fb62e44e1800be22ef8d3ecf02b16dde.
- Evidence: `analysis/split_correction/validation.json` and `tests.txt`. No
  original checkpoint/output was changed, and no jobs were submitted. No new
  corrected checkpoint or real validation metrics exist until retraining runs.

Launch from repository root:

```bash
sbatch slurm/train_sdf_only_split.sh
# After the NEW corrected checkpoint exists:
sbatch slurm/evaluate_sdf_only.sh 3 /absolute/path/new_validation3 --checkpoint /absolute/path/NEW_checkpoint.pt --cohort-manifest configs/evaluation/validation_cohort.json
sbatch slurm/evaluate_sdf_only.sh 5 /absolute/path/new_validation5 --checkpoint /absolute/path/NEW_checkpoint.pt --cohort-manifest configs/evaluation/validation_cohort.json
```
