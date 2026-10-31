# Conversation context and restart point

Last updated: 2026-10-30 (Phase 0a/0b).

## Slurm resource sizing

      Before every sbatch, check sinfo/squeue and size CPUs, memory and partition to what is
      free now so the job starts immediately. Prefer fewer CPUs over waiting. Keep the GPU
      count identical across runs that are compared.

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

## 2026-10-30: migration to EURECOM helix cluster

- Repository moved to `/users/zhangx/VATL_SDF`; data to
  `/medias/medical/data/Atlas_training/v1`. Live configs, Slurm scripts, tests
  and docs were updated; `analysis/` and earlier sections here keep the
  original Leonardo paths as historical records.
- Measured: the new `ixi_metadata_label_t1_fast_sdf.tsv` is byte-identical to
  the original after restoring the old path prefix (SHA256 of the substituted
  file = frozen 71d6ab5b…). The new file's SHA256
  b1f896eeccf1f9889da7b63ddd7bb6fd68f0712e4191cc7d878570f2e90134a5 replaced it
  in `configs/evaluation/validation_cohort.json`. Split YAML SHA256 is
  unchanged (bc84b1dd…), 497/55/0.
- `output_root` is now `/medias/medical/data/Atlas_training/Cinema_output/mra_sdf_only`
  (no personal scratch exists). Training hyperparameters unchanged; still 4 GPUs
  (now H200 instead of Leonardo A100), so DDP world size matches.
- Slurm: `-A` removed (default `caravel`), `-p h200`, `--gres=gpu:h200:N`,
  `--mem=256G` added to training (DefMemPerCPU is 4 GB), `module` calls removed.
- The original-config equality test now skips unless the Leonardo
  `config_final.yaml` is copied to the path named in `tests/test_splits.py`;
  `output_root` dropped from that comparison as machine-specific.
- Validation on login node: test_splits + test_pipeline 16 OK (1 skipped),
  test_individual 6 OK; real-data audit of 2 subjects loaded; `sbatch
  --test-only` accepted training and evaluation scripts. No jobs submitted.

## Refactor programme (started 2026-10-30): SDF-only, skeleton supervision, probabilistic atlas

User task (2026-10-30): Phase 0 diagnose data/results; Phase 1 pure-SDF refactor
(remove MRA loading/head/loss/exports, tag `pre-sdf-only-refactor` first);
Phase 2 centerline radius/hinge supervision; Phase 3 population variability
maps; Phase 4 label-free probabilistic graph design doc only. Stop and report
after each phase; wait for confirmation. Pilot Slurm budget without asking:
<=1 node, <=4 GPUs, <=4 h/job, <=2 jobs at once, <=32 GPU-h/phase; outputs only
under `<output_root>/claude_pilots/<date>_<name>/`. Full runs need approval.

### Phase 0 (0a, 0b done; 0c waiting for checkpoint)

User answers: will copy the Leonardo 552-subject 02_sdf_only checkpoint
(`checkpoint_step_00262200.pt`, SHA256 6f7c52af…) to helix for fitting
diagnostics (training subjects only; contaminated for validation). Graphs were
built from the same fast label-1 binary mask resampled to 0.8 mm isotropic.

Graph format (verified): `v1/graph/<id>-Graph.pkl` = dict {graph: networkx.Graph,
node_radius_map, voxel_sizes, input_file}. Node attrs `pos` (world mm, same
RAS world frame as the SDF/MRA affine; no flip) and `radius` (mm); edge attr
`weight` = Euclidean edge length; a few nodes carry `contraction`. Index TSV:
`v1/ixi_metadata_label_t1_fast_graph.tsv`; all 552 SDF subjects have a graph.

Measured over all 552 subjects (job 46213; `claude_pilots/20261030_graph_inspect/`,
script `analysis/phase0_graph/inspect_graphs.py`):

- Nodes median 3,777 (1,422–6,216); median edge length 0.93 mm, max <=2 mm.
- Skeleton graphs are themselves fragmented: components median 51 (5–95%:
  26–122), largest component median 68% of nodes. Independent cycles median 35
  (possible false loops where vessels touch; not yet inspected).
- Radius is a distance transform on the 0.8 mm grid (values 0.8·sqrt(k)):
  83.6% of nodes sit at the floor 0.8 mm (one voxel); 0% below 0.5 mm; max
  1.8–3.7 mm. The floor is a quantization limit, not a measured radius.
- Coordinates: 99.6%+ of nodes inside the SDF FOV; median 90.8% inside the
  label-1 mask; median reference SDF at nodes −0.63 mm; ≤0.4% of nodes >1 mm
  outside the surface; corr(−SDF, r) median 0.68. Coordinate frames agree.
- SDF(c) vs −r(c) (5 subjects, 20,871 nodes): for r >= 0.9 mm, SDF ≈ −r with
  median offset +0.10 to +0.23 mm and 100% of nodes inside. For floor nodes
  (r = 0.8) median SDF −0.44 mm, 21% are outside the reference mask
  (median SDF+r = +0.36). Implication for Phase 2: do not regress floor radii;
  even a hinge on floor nodes contradicts the reference SDF for ~1/5 of them.
- Mask voxels farther than (r+1 mm) from all nodes: median 4.5%, 95th pct 20%.
- `sdf < 0` equals the label-1 mask exactly for every subject.

Infrastructure fix: `slurm/environment.sh` aborted every `set -e` batch job with
exit 127 (failing command inside ~/.bashrc1; job 46211 failed in 0 s). It now
relaxes -e/-u while sourcing and restores the caller's options.

Prepared, not yet run: `analysis/phase0_graph/diagnose_checkpoint.py` and
`slurm/diagnose_checkpoint.sh` (1×L40S, dev QoS, 2 h): six training subjects
(3 per sex at age quantiles 0.1/0.5/0.9), native grid and 2× supersampled grid
vs reference (Dice, volume, 6/26 components, largest fraction), grid-free
centreline recall by radius bin, mean-latent atlas at 1 mm and 0.5 mm.

Cluster notes: QoS `dev` = 2 h, 8 GPUs/128 CPU per user; h200 queue estimated
multi-day waits on 2026-10-30, l40s started within minutes.

### Phase 0a: graph–mask correspondence (done 2026-10-30; job 46230, CPU)

User's consolidated plan (2026-10-30) supersedes earlier prompts: inputs are SDF +
graph only (no MRA anywhere, Phase 1), binary masks without anatomical labels,
graphs are genuinely imperfect (fragments, extra loops) and must not be "fixed";
connectivity is always judged relative to the subject's reference. Phases 0–5,
stop and report after each. Reference loader: `reference/graph_store.py` (user
copy of `VATL_MM/data_loading/graph_store.py`; committed read-only as
`reference/graph_store_reference.py` with a provenance header).

Script `analysis/phase0_graph/correspondence.py` (modes headers / conventions /
stats / qc), launcher `slurm/graph_correspondence.sh`, convention module
`mra_sdf_only/graph_coords.py`, tests `tests/test_graph_coords.py` (8 pass, CPU).
Outputs: `claude_pilots/20261030_graph_correspondence/`.

Verified facts:
- Pickle: dict {graph, node_radius_map, voxel_sizes (0.7999918 iso), input_file
  (`/home/cazzolla/.../IXI_graphs/_bin_masks/<id>-Mask.nii.gz`, not on helix)};
  g.graph = {}; node attrs pos, radius (+ contraction on ~1–10 merged nodes);
  edge attr weight = length. No affine stored. Generator script NOT found (searched
  /users/zhangx, data root, VATL_MM/VATL_SDF history); its exact regrid is unknown.
- Headers (552): mask, SDF, MRA share shape and sform exactly (sform code 2;
  qform code 1 for mask/SDF, 0 for MRA; qform = sform within 1.5e-5). Grids are
  anisotropic 0.9375×0.9375×0.8 mm (some 1.0547×1.0547×0.8), oblique ~7°.
- The "mask" file is MULTI-LABEL (0–10; labels 3/4 have ~0.5–0.8 M voxels, i.e.
  tissue classes). Vessel = label 1; `SDF < 0 == label 1` exactly in all 552.
  Only label 1 is used anywhere. (User says masks are binary — to confirm.)
- Convention: the reference convention is correct: pos = world mm, voxel =
  inv(sform of segmentation) @ pos, integer index = voxel centre, nearest voxel =
  round. 5 subjects, nearest-voxel inside fraction (all / floor): current
  0.887/0.860; qform identical; every ±0.5-voxel shift worse (best 0.895/0.872 for
  −0.5 j only); flips, permutations, index or index×spacing interpretations
  0–5%. Per-subject optimal global translations (grid search ±1 mm) reach
  0.95–0.96/0.93–0.96 but are inconsistent across subjects ([−.3,−.3,0],
  [0,−.3,.1], [−.2,.1,0], [0,−.4,0], [−.1,−.1,0] mm) and overfit by search:
  NOT adopted. Inside floor nodes have mean offset ≈0 to the nearest mask voxel
  (std 0.28 voxel = uniform), so there is no global shift.
- Why thin nodes miss: nodes are sub-voxel points on the graph's 0.8 mm grid;
  misses are in-plane (i, j, where 0.9375 ≠ 0.8), ~10% in k (0.8 on both). Distance
  to the voxelized mask (union of voxel boxes): floor-outside median 0.08 mm, p95
  0.32 mm (552 subjects); 99.8–99.95% of nodes within 0.40 mm = half a graph voxel;
  max 0.57 mm; ≤0.3% of nodes >0.5 mm, none >1 mm. Re-gridding hypotheses
  (same-rotation 0.8 mm origin/corner aligned, order 0/1; nibabel
  resample_to_output order 0/1/3) do not contain the nodes or reproduce their
  lattice. Hypothesis (unverified without the generator): extraction on a 0.8 mm
  in-plane regrid whose phase differs from the native grid.
- Earlier "21% of floor nodes outside" stands roughly (552 subjects: floor
  inside median 0.892, 5–95% 0.79–0.96), but it is a ≤0.4 mm sub-voxel
  discretization effect, not misalignment.
- Radius: mm, scale_to_mm = 1. radius/0.8 = sqrt(k) exactly (EDT on the 0.8 mm
  grid); matches the native mask's interior EDT in mm (ratio 0.92–1.07). The 0.8
  floor = one graph voxel, the minimum EDT any foreground voxel can have; a
  one-voxel-thick vessel's true radius is ≈0.4–0.47 mm, so floor radii
  overestimate thin vessels ~2× (reference SDF + r at floor nodes median +0.28 mm).
- Inside fraction by radius (552, medians): floor 0.892, 0.9–1.2 mm 0.995,
  1.2–1.6 mm 1.000, ≥1.6 mm 1.000. SDF + r median: 0.13 / 0.17 / 0.17 mm.

Implications for Phase 2 (proposal, needs user decision): "mask-supported" with
the strict nearest-voxel rule drops ~11% of floor nodes that are within 0.4 mm of
the mask; a tolerance of 0.4 mm (MaskDistance.distance <= 0.4) keeps 99.8%.

### Phase 0b: graph imperfection vs mask (done; job 46230; descriptive only)

`graph_mask_stats.tsv` (per subject) and `loops.tsv` (per loop). Medians (5–95%):
- Mask 26-components 93.5 (15–397); largest 0.84 of voxels (0.66–0.97); most
  are tiny: 1 voxel 25.5, 2–9 voxels 34, 10–99 voxels 25, ≥100 voxels 5.
- Graph components 51 (26–122); largest 0.68 of nodes (0.45–0.88), 0.69 of length.
  No single-node pieces; 2–4 nodes 1; 5–19 nodes 35; ≥20 nodes 15; median piece
  9.8 mm.
- Every graph component lies within exactly one mask component (all 552; nodes
  snapped ≤1 mm, none needed >1 mm). Zero edges join two mask components: the
  graph never bridges separate mask pieces.
- Graph fragments inside connected mask: 17 graph pieces in the largest mask
  component (8–44); mask components with ≥2 graph pieces 5. Mask components
  without any node 60 (5–307) holding 1.1% of mask voxels (0.1–10%).
- Edges with an endpoint outside the mask (nearest voxel) 597 per subject — Phase
  2 edge filtering on strict inside would drop many thin-vessel edges.
- Loops (approximate minimum cycle basis; found = cycle rank in 550/552):
  35 per subject; short (<5 mm) 3, long 31; loop length median 20.6 mm.
  Short loops: max radius 0.8 mm, median 50% of nodes inside the mask (likely
  touching/one-voxel artefacts, hypothesis). Long loops: median max radius
  1.13 mm, 95% nodes inside. Every loop lies within one mask component.
  5 mm threshold is arbitrary; per-loop lengths are saved.

### User decisions after Phase 0a/0b report (2026-10-30)

1. Binary vessel mask = (label == 1). Never use the other labels anywhere. All
   current code defaults to 1; Phase 1 must replace the `--vessel-label` options
   (preprocess_sdf.py, qc_sdf.py, evaluate.py) with `graph_coords.VESSEL_LABEL`.
2. Mask support for graph terms: 0.4 mm tolerance (half a graph voxel) by default;
   tolerated nodes outside the mask are snapped to their nearest mask voxel centre
   for hinge targets, so a hinge never asks for SDF < 0 where the reference is
   outside. Edge-path hinge: sample along edges between snapped nodes and keep
   edges whose samples are all within 0.4 mm of the mask. Strict nearest-voxel
   rule remains a config option. Implemented and tested in
   `mra_sdf_only/graph_coords.py` (`mask_support`, `supported_edges`,
   `edge_samples`; tests in `tests/test_graph_coords.py`).
3. User will copy the SDF-only checkpoint to helix and give the path (0c).
4. Reference loader committed as `reference/graph_store_reference.py`.

Phase 2 change (supersedes the plan's centreline-radius default): graph radii are
quantized on the 0.8 mm grid, the floor overstates one-voxel vessels ~2×, and
larger radii are biased by ~0.1–0.2 mm vs the reference SDF. The SDF is the ground
truth, so the DEFAULT centreline term is centreline-weighted sampling: extra SDF
samples at (mask-supported, snapped) graph nodes with the reference SDF as target.
Graph-radius regression (r >= 0.9 mm only, never floor nodes) is an optional,
separate ablation.

Reference-relative connectivity baseline: the reference MASK (26-connectivity),
median 93.5 components, largest component 0.84 of voxels. The graph is more
fragmented than the mask (median 51 pieces, largest 0.68; 17 graph pieces inside
the largest mask component). Do not bridge graph pieces now. Possible later step:
bridging that only adds connections inside a single mask component (never between
mask components).

### Phase 0c: checkpoint diagnostics — pending (checkpoint not yet on helix)

### 2026-10-30 (late): user authorization and output location

User: there is no checkpoint for this session; put all results under
`/medias/medical/caravel/zhangx/training_outputs/sdf_only` and execute the
experiments without asking for approval. Consequently: Phase 0c uses a NEW
SDF-only baseline trained here (497/55 split); phases proceed without waiting
for confirmation; reports are recorded here. Layout: `runs/` (full runs),
`pilots/` (screening), `logs/`, `analysis/`.

### Phase 1: pure SDF + graph pipeline (done; commit 264a170, tag pre-sdf-only-refactor)

- Tag `pre-sdf-only-refactor` = last code with MRA loader/head/loss (package
  `mra_sdf_only`); export old MRA-head checkpoints from a worktree of that tag.
- Package renamed `mra_sdf_only` -> `vatl_sdf`. Dataset (`SDFGraphDataset`)
  reads subject_id, SDF, Graph, scan_age, sex only (TSV MRA/T1/Seg columns are
  ignored); crop and normalization from the SDF affine; graph nodes normalized
  by the same `graph_coords.normalize_world`. Vessel mask = SDF < 0 (= label 1).
- Model `SDFModel`: single SDF head, output (N, 1). Loss: sdf_weight*L1(norm.
  SDF) + moe_weight*balance (+ tf_weight*transform). MRA metrics, weights and
  gradient diagnostics removed. Exports VesselSDF + VesselMask only.
- `evaluate.py`: SDF MAE, truncation-band MAE, zero-level Dice, sign agreement,
  components, and `reference_relative_connectivity` (26-conn counts, ratio,
  excess, >=10-voxel counts, largest-fraction difference vs reference mask).
- Config `configs/sdf/00_sdf_baseline.yaml` (= `config.yaml`): legacy
  02_sdf_only except no MRA and sdf_weight 1.0 (test checks this). Legacy
  configs/launchers in `configs/ablations/legacy_mra/`, `slurm/legacy_mra/`.
- `--vessel-label` options removed (graph_coords.VESSEL_LABEL).
- Tests (CPU): 46 pass (pipeline incl. exact resume and graph–SDF coordinate
  consistency; graph terms; population maps; splits on the real 497/55 YAML).
- GPU smoke job 46330 (3 L40S, DDP, 8 subjects, full-size model): 200 steps,
  exact resume to 300, atlas + native subject export + evaluation, DDP table
  difference 0.0, export mask mismatch 0. Throughput ~3 optimizer steps/s per
  rank (25k SDF points/rank/step) -> full 200-epoch 497-subject run ~23 h on
  4 L40S, ~31 h on 3. l40s nodes have 32 CPUs: request 16 CPUs per job.
- GPU availability: h200 congested; no l40s node had 4 free GPUs (helix02: 3).
  Full runs therefore use 3 GPUs (world size 3 = 15 subjects per distributed
  batch, config otherwise unchanged); all compared full runs must match.

### Phase 2: graph supervision (code done; commit e06b276; pilots running)

Design (all terms use the Phase 0a support rule; each enabled term draws its
own points_fraction (0.15) x SDF points EXTRA per subject, so 04 = union):
- `centerline_sdf` (DEFAULT centreline term per user): points uniform along
  edges with both endpoints supported (original positions), target =
  trilinear reference SDF. Never contradicts the reference.
- `centerline_hinge`: supported nodes, snapped; relu(pred - ceiling).
- `edge_hinge`: points along mask-supported edges between snapped nodes.
  Cannot connect separate graph pieces (by construction).
- Hinge ceiling = reference SDF where it is > 0 (tolerated points <= 0.4 mm
  out), else -epsilon (0): a hinge never asks SDF < 0 where reference > 0.
- `radius` (optional 05): |pred + r| at supported nodes with r >= 0.9 mm.
- `eikonal` (optional 06): (|grad SDF|_mm - 1)^2, gradient converted with the
  anisotropic bbox half-lengths and truncation; DDP not tested.
- MoE balancing on SDF points only (balance_count).
Configs `configs/sdf/01_centerline_sdf` … `06_eikonal` (one change each).
Pilot protocol (screening evidence only): single L40S, first 100 training
subjects (YAML order), seed 42, 40,000 steps (40 epochs), then
`analysis/phase0_graph/diagnose_checkpoint.py` on 6 fixed training subjects
(3 per sex, age quantiles), native grid + 1 mm atlas.

### Claude-submitted jobs

| Job | Purpose | Overrides | Output | GPU-h | Outcome |
| --- | --- | --- | --- | ---: | --- |
| 46211 | graph inspection (CPU) | -p l40s | claude_pilots/20261030_graph_inspect | 0 | FAILED 0 s: environment.sh exit 127 |
| 46212 | environment test (CPU) | -p l40s, dev | same dir, envtest_46212.log | 0 | COMPLETED, diagnosed 46211 |
| 46213 | graph inspection, 552 subjects (CPU) | -p l40s | claude_pilots/20261030_graph_inspect | 0 | COMPLETED 3 min |
| 46230 | Phase 0a/0b correspondence + stats, 552 subjects (CPU) | -p l40s, dev, 16 CPU | claude_pilots/20261030_graph_correspondence | 0 | COMPLETED 4 min 49 s |
| 46325 | Phase 1 smoke (3 L40S) | 32 CPUs requested | — | 0 | CANCELLED: not schedulable (node has 16 free CPUs) |
| 46330 | Phase 1 smoke, DDP 3 L40S, 8 subj, 200+100 steps | 16 CPUs | sdf_only/pilots/20261030_phase1_smoke_ddp | 0.13 | COMPLETED 2.5 min, SMOKE_OK |
| 46332 | FULL 00_sdf_baseline, 497 subj, 200 epochs (part 1) | 3 L40S, 24 h | sdf_only/runs/00_sdf_baseline_full | running | |
| 46333 | continuation of 46332 (afterany) | | | 0 | CANCELLED: would import live code; replaced by 46342 |
| 46335–46339 | pilots 01–05 (first submission) | | | 0 | CANCELLED before start: live code tree; replaced by 46343–46347 |
| 46334 | Phase 2 pilot 00_sdf_baseline (100 subj, 40k steps) + diagnostics | 1 L40S | sdf_only/pilots/20261030_p2_00_sdf_baseline | 1.80 | COMPLETED 1 h 48 min, PILOT_OK |
| 46342 | continuation of 46332 (afterany; stop guard exits if done) | 3 L40S, frozen e06b276 | sdf_only/runs/00_sdf_baseline_full | pending | |
| 46343 | Phase 2 pilot 01_centerline_sdf | 1 L40S, frozen e06b276 | sdf_only/pilots/20261030_p2_01_centerline_sdf | 2.03 | COMPLETED, PILOT_OK; degenerate field (see screening) |
| 46344–46347 | Phase 2 pilots 02–05 | 1 L40S each, frozen e06b276; 03–05 nice 100 | sdf_only/pilots/20261030_p2_<config> | running/pending | |
| 46468 | Phase 2 pilot 01b_centerline_sdf_pointweight | 1 L40S, frozen 631f3f0 | sdf_only/pilots/20261031_p2_01b_centerline_sdf_pointweight | pending | |
| 46389 | post-baseline: 0c (x1/x2, atlas 1/0.5 mm), population maps 1 mm, centreline density | 1 L40S, 12 h, afterok:46342, frozen 41199e4 | sdf_only/analysis/20261031_00_baseline_post | pending | |
| 46354 | Phase 5 rendering, 20 subjects (CPU) | moved to h200 partition | sdf_only/analysis/20261030_phase5_render | 0 | COMPLETED |

Frozen-code practice: queued jobs set `VATL_CODE_DIR=/users/zhangx/VATL_SDF_frozen/<commit>`
(a `git worktree` at that commit) so live edits never reach pending jobs.

### Phase 2 pilot screening (screening evidence: 100 subjects, 40k steps, 1 seed)

`python -m analysis.phase2.summarize_pilots` (medians over 6 training subjects,
native grid; reference-relative connectivity). Rows added as pilots finish.

| Pilot | Dice | band MAE mm | vol ratio | comp ratio >=10 | largest diff | inside floor / 0.9-1.2 / 1.2-1.6 / >=1.6 | atlas 1 mm comps (largest) | SDF L1 last 2k |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 00 baseline | 0.184 | 1.73 | 1.39 | 5.40 | -0.33 | 0.08 / 0.24 / 0.47 / 0.75 (ref floor 0.95) | 1658 (0.55) | 0.139 |
| 01 centerline_sdf (w 1.0) | 0.019 | 3.43 | 96.1 | 0.37 | +0.09 | 0.99 / 1.00 / 1.00 / 1.00 | 941 (1.00), 1.67 M voxels | 0.251 |

Observation (fact): at pilot scale individual fits are poor (Dice 0.18, 5.4x
the reference's >=10-voxel components), so pilot comparisons measure early
training behaviour, not converged fits. Whether this persists at full scale is
answered by 0c on the full baseline.

Pilot 01 (job 46343, 2.0 h): DEGENERATE. Facts: the predicted SDF is negative almost
everywhere (volume ratio 96, atlas 1.67 M voxels; "connectivity" and centreline
inside fractions are therefore meaningless). The SDF L1 rises from step ~3 onward
(0.418 -> 0.455 by step 5, while 00 falls to 0.35), and centerline_sdf_raw plateaus
at ~0.066 normalized, the magnitude of the targets. Code checks (coordinates,
normalization, target lookup, loss, split of the concatenated forward, MoE
balance_count) found no bug. Hypothesis: term weighting. Each graph term is its own
mean with weight 1.0, so a centreline point weighs ~6.7x an SDF point (1/0.15),
and all centreline targets are <= ~0; a global negative shift is the cheapest
reduction early in training. Test: 01b_centerline_sdf_pointweight (weight 0.15
= per-point weight of an SDF sample, i.e. literally "extra SDF samples"; job
46468). Pilots 02-05 also use weight 1.0 and the hinges also push only downward;
03-05 were deprioritized (nice 100) so 01b runs first; 02 was already running.

Full baseline (46332) trajectory (fact): SDF L1 (1k-step means) 0.172 at step 0,
0.118 at 10k, a plateau ~0.111-0.114 from 15k to 87k (epoch ~52) with a slight upward
drift. Pilot 00 shows the same rise after ~10k (0.123 -> 0.137). Not yet explained;
learning rate / MoE routing are candidate explanations (hypotheses).
