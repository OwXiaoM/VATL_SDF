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

### Phase 0c: checkpoint diagnostics — done on full 00_sdf_baseline (2026-10-31; see end of file)

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
| 46332 | FULL 00_sdf_baseline, 497 subj, 200 epochs (part 1) | 3 L40S, 24 h | sdf_only/runs/00_sdf_baseline_full | 44.5 | COMPLETED 14 h 51 min (diverged late, see divergence) |
| 46333 | continuation of 46332 (afterany) | | | 0 | CANCELLED: would import live code; replaced by 46342 |
| 46335–46339 | pilots 01–05 (first submission) | | | 0 | CANCELLED before start: live code tree; replaced by 46343–46347 |
| 46334 | Phase 2 pilot 00_sdf_baseline (100 subj, 40k steps) + diagnostics | 1 L40S | sdf_only/pilots/20261030_p2_00_sdf_baseline | 1.80 | COMPLETED 1 h 48 min, PILOT_OK |
| 46342 | continuation of 46332 (afterany; stop guard exits if done) | 3 L40S, frozen e06b276 | sdf_only/runs/00_sdf_baseline_full | 0.02 | COMPLETED 23 s (stop guard, no updates) |
| 46343 | Phase 2 pilot 01_centerline_sdf | 1 L40S, frozen e06b276 | sdf_only/pilots/20261030_p2_01_centerline_sdf | 2.03 | COMPLETED, PILOT_OK; degenerate field (see screening) |
| 46344 | Phase 2 pilot 02_centerline_hinge | 1 L40S, frozen e06b276 | sdf_only/pilots/20261030_p2_02_centerline_hinge | 2.03 | COMPLETED; degenerate field |
| 46345 | Phase 2 pilot 03_edge_hinge | 1 L40S, frozen e06b276 | sdf_only/pilots/20261030_p2_03_edge_hinge | 2.04 | COMPLETED; degenerate field |
| 46346 | Phase 2 pilot 04_all_graph_terms | 1 L40S | — | 0.02 | CANCELLED by Claude after 1.5 min: union of weight-1.0 terms, uninformative after 01/02 |
| 46347 | Phase 2 pilot 05_radius_regression | | | 0 | CANCELLED by Claude before start: weight-1.0 negative-only targets, replaced by 05b |
| 46530 | Phase 2 pilot 02b_centerline_hinge_pointweight | 1 L40S, frozen 970b2b4 | sdf_only/pilots/20261031_p2_02b_centerline_hinge_pointweight | 2.03 | COMPLETED |
| 46531 | Phase 2 pilot 03b_edge_hinge_pointweight | 1 L40S, frozen 970b2b4 | sdf_only/pilots/20261031_p2_03b_edge_hinge_pointweight | 2.03 | COMPLETED |
| 46532 | Phase 2 pilot 04b_all_graph_terms_pointweight | 1 L40S, frozen 970b2b4 | sdf_only/pilots/20261031_p2_04b_all_graph_terms_pointweight | 2.51 | COMPLETED; partial inflation |
| 46533 | Phase 2 pilot 05b_radius_regression_pointweight | 1 L40S, frozen 970b2b4 | sdf_only/pilots/20261031_p2_05b_radius_regression_pointweight | 2.04 | COMPLETED |
| 46468 | Phase 2 pilot 01b_centerline_sdf_pointweight | 1 L40S, frozen 631f3f0 | sdf_only/pilots/20261031_p2_01b_centerline_sdf_pointweight | 2.04 | COMPLETED, PILOT_OK |
| 46632–46634 | Phase 4 pilots 07a/07b/07c | 1 L40S each, frozen e7fe991 | sdf_only/pilots/20261031_p4_<config> | 2.06 each | COMPLETED |
| 46635 | Phase 4 pilot 07d junction | 1 L40S, frozen e7fe991 | sdf_only/pilots/20261031_p4_07d_field_junction | running | |
| 46700 | FULL 07a_field_centerline_distance (afterany:46342) | 3 L40S, 24 h, frozen e7fe991 | sdf_only/runs/07a_field_centerline_distance_full | pending | |
| 46701 | continuation of 46700 | 3 L40S, frozen e7fe991 | same | pending | |
| 46636 | FULL 01b_centerline_sdf_pointweight, 497 subj, 200 epochs (part 1; afterany:46342) | 3 L40S, 24 h, frozen e7fe991 | sdf_only/runs/01b_centerline_sdf_pointweight_full | pending | |
| 46637 | continuation of 46636 (afterany) | 3 L40S, frozen e7fe991 | same | pending | |
| 46799 | Alignment Step 0 (CPU) | dev, 16 CPU, frozen ce77ff2 | sdf_only/analysis/20261031_alignment_step0 | 0 | COMPLETED 1.5 min |
| 46800 | Divergence inspection of 00 baseline | 1 L40S | sdf_only/analysis/20261031_divergence_00_baseline | 0.01 | COMPLETED |
| 46802–46806 | Alignment pilots C0, A1–A4 | 1 L40S each, frozen b6ac683 | sdf_only/pilots/20261031_al_<config> | ~10 | COMPLETED (2 h 00–2 h 05 each) |
| 46807 | Recurrence of baseline epoch-20 learned transforms (CPU) | dev, 8 CPU | sdf_only/analysis/20261031_alignment_baseline_ep20_{first100,all} | 0 | COMPLETED |
| 46808/46811 | CPU smoke of A4 config (+centroid init) on 2 real subjects, 8 steps | dev, 8 CPU | sdf_only/pilots/20261031_al_smoke_cpu{,2} | 0 | 46808 refused (max_subjects needs max_steps); 46811 SMOKE_OK |
| 46389 | post-baseline on final checkpoint | | | 0 | CANCELLED by Claude before start: final checkpoint degraded (see divergence); replaced by 46764–46766 |
| 46636/46637, 46700/46701 | full 01b / 07a runs, 200 epochs | | | 0 | CANCELLED before start; replaced with stop_after_epoch=59 (46767–46770) |
| 46764 | 0c + population maps + centreline density on baseline step 33200 (epoch 20) | 1 L40S, frozen ed1cc9d | sdf_only/analysis/20261031_00_baseline_step33200 | 0.51 | COMPLETED (0c 703 s, maps 877 s, density 252 s) |
| 46765 | 0c on baseline step 99600 (epoch 60) | 1 L40S | sdf_only/analysis/20261031_00_baseline_step99600 | 0.21 | COMPLETED |
| 46766 | 0c on baseline final checkpoint (afterok:46342) | 1 L40S | sdf_only/analysis/20261031_00_baseline_final | 0.21 | COMPLETED; collapsed field |
| 46767/46768 | FULL 01b, stop_after_epoch=59 (+continuation) | 3 L40S, frozen e7fe991 | sdf_only/runs/01b_centerline_sdf_pointweight_full | 16.2 | COMPLETED 5 h 23 (checkpoints to step 99600 = epoch 60); continuation: stop guard, no updates |
| 46769/46770 | FULL 07a, stop_after_epoch=59 (+continuation) | 3 L40S, frozen e7fe991 | sdf_only/runs/07a_field_centerline_distance_full | 15.3 | COMPLETED 5 h 07 (to step 99600); continuation: stop guard, no updates |
| 46829–46832 | 0c on full 01b / 07a, epochs 20 and 60 (steps 33200/99600) | 1 L40S each, frozen b6ac683 | sdf_only/analysis/0c_<01b,07a>_full_epoch<20,60> | 0.8 | COMPLETED (12 min each) |
| 46841 | FULL A4 (13_a4_full): 497 subj, 36 epochs = 59,760 steps, run horizon, alignment 10 epochs | 3 L40S, frozen 0ac21bc | sdf_only/runs/13_a4_full | 9.5 | COMPLETED 3 h 11 (21:46 -> 00:57); horizon 36 epochs |
| 46847 | gen_align: test-time ALIGNMENT-only fit (10 epochs, horizon 36) on step 16,600; 5 test + 6 train_refit | 1 L40S, frozen a8b603c | sdf_only/analysis/20261031_a4full_gen_align | 0.65 | COMPLETED 39 min (during training) |
| 46848–46851 | Probabilistic atlas shards (4 x 124 subjects, 1 mm) on step 16,600 | 1 L40S each, frozen a8b603c | sdf_only/analysis/20261031_a4full_patlas_shards_align | 1.4 | COMPLETED 20–22 min each |
| 46852 | Probabilistic atlas merge + held-out evaluation, step 16,600 (CPU) | 16 CPU, frozen a8b603c | sdf_only/analysis/20261031_a4full_patlas_align | 0 | COMPLETED 1 min |
| 46858 | gen_final: test-time fit 10 + 26 epochs (training budget) on step 59,760 | 1 L40S, frozen a8b603c | sdf_only/analysis/20261031_a4full_gen_final | 0.93 | COMPLETED 56 min |
| 46859–46862 | Probabilistic atlas shards, step 59,760 | 1 L40S each, frozen a8b603c | sdf_only/analysis/20261031_a4full_patlas_shards_final | 1.3 | COMPLETED 20–21 min each |
| 46863 | Probabilistic atlas merge + held-out evaluation, step 59,760 (CPU) | 16 CPU, frozen a8b603c | sdf_only/analysis/20261031_a4full_patlas_final | 0 | COMPLETED 1 min |
| 46864 | gen_long: test-time fit 10 + 98 epochs (3x budget, horizon 108), test only, step 59,760 | 1 L40S, frozen a8b603c | sdf_only/analysis/20261031_a4full_gen_final_long | 0.74 | COMPLETED 44 min |
| 46865 | inspect_run on the full A4 run (reference steps 20k–35k) | 1 L40S, frozen a8b603c | sdf_only/analysis/20261031_a4full_inspect_run | 0.01 | COMPLETED 32 s |
| 46885 | Scale-gap probe: per-subject SDF L1 with latent/condition swaps, all checkpoints of pilot A4 and full A4, 100 shared subjects | 1 L40S, frozen fd206c0 | sdf_only/analysis/20261101_scale_gap_probe | 0.07 | COMPLETED 4 min |
| 46886 | inspect_run on the Step 2 A4 pilot | 1 L40S, frozen fd206c0 | sdf_only/analysis/20261101_scale_gap_inspect_pilot | 0.01 | COMPLETED |
| 46887 | Test-time fitting with the PILOT A4 network (5 test + its 6 diagnostic subjects, 10 + 30 epochs) | 1 L40S, frozen fd206c0 | sdf_only/analysis/20261101_scale_gap_gen_pilot | 0.93 | COMPLETED 56 min |
| 46889 | Vessel-graph atlas step 1 on full A4 final (CPU) | 16 CPU, frozen 08c27c1 | sdf_only/analysis/20261101_a4full_graph_atlas | 0 | COMPLETED 2.5 min |
| 46890/46891 | sg_r20: 14_a4_r20 (reference, 497 subj, 20 epochs) + post (0c, alignment, generalization, inspect) | 3 L40S + 1, frozen 73d5426 | sdf_only/runs/sg_r20, analysis/20261101_sg_r20 | | running |
| 46892/46893 | sg_latent6: 15_a4_r20_latent6 + post | 3 L40S + 1, frozen 73d5426 | sdf_only/runs/sg_latent6, analysis/20261101_sg_latent6 | | running |
| 46894/46895 | sg_width1536: 16_a4_r20_width1536 + post | 3 L40S + 1, frozen 73d5426 | sdf_only/runs/sg_width1536, analysis/20261101_sg_width1536 | | pending |
| 46896/46897 | sg_r40: 17_a4_r40 (40 epochs) + post | 3 L40S + 1, frozen 73d5426 | sdf_only/runs/sg_r40, analysis/20261101_sg_r40 | | pending |
| 46898/46899 | sg_latentlr: 18_a4_r20_latentlr + post | 3 L40S + 1, frozen 73d5426 | sdf_only/runs/sg_latentlr, analysis/20261101_sg_latentlr | | pending |
| 46900/46901 | sg_s100: 19_a4_subset, 100 subj (incl. the 6), 20k steps, 1 GPU + post (generalization, inspect) | 1 L40S + 1, frozen 73d5426 | sdf_only/pilots/20261101_sg_s100 | | running |
| 46902/46903 | sg_s100nocond: 20_a4_subset_nocond, 100 subj, condition zeroed + post | 1 L40S + 1, frozen 73d5426 | sdf_only/pilots/20261101_sg_s100nocond | | pending |
| 46904/46905 | sg_s250: 19_a4_subset, 250 subj, 50k steps + post | 1 L40S + 1, frozen 73d5426 | sdf_only/pilots/20261101_sg_s250 | | pending |
| 46906/46907 | sg_s100g3: 19_a4_subset, 100 subj on 3 GPUs (15 per step), 6800 steps + post | 3 L40S + 1, frozen 73d5426 | sdf_only/pilots/20261101_sg_s100g3 | | pending |
| 46908 | Connectivity + radius atlas on full A4 final (CPU; moved to QOS dev, 2 h) | 16 CPU, frozen 9d8fd5f | sdf_only/analysis/20261101_a4full_attribute_atlas | 0 | COMPLETED 1.3 min |
| 46909 | Same + node/edge atlas (node position, covariance) (CPU, QOS dev) | 16 CPU, frozen 1af803e | sdf_only/analysis/20261101_a4full_attribute_atlas_v2 | 0 | COMPLETED 1.3 min |
| 46843 | Smoke: generalization + probabilistic atlas on the Step 2 A4 pilot checkpoint (tiny budgets, 4 mm grid) | 1 L40S, frozen a8b603c | sdf_only/analysis/20261031_eval_smoke_a4pilot | 0.06 | FAILED 3 min: CUDA OOM, caused by the smoke setting --steps-per-epoch 5 (50k points per subject per step; real setting 50 -> 5k) |
| 46844 | Same smoke, --steps-per-epoch 50 | 1 L40S, frozen a8b603c | sdf_only/analysis/20261031_eval_smoke_a4pilot_2 | 0.09 | COMPLETED 5.5 min, SMOKE_OK (shard: 100 subjects at 4 mm in 17 s) |
| (auto) | Watcher (login-node poller, scratchpad autosubmit.sh, frozen a8b603c): at step 16,600 -> gen_align + 4 shards + merge; at step 59,760 -> gen_final + 4 shards + merge, gen_long, inspect_run (afterany 46841) | | sdf_only/analysis/20261031_a4full_* | | done: submitted 22:35 and 00:57 (first launch failed: scratchpad not executable; rerun with bash) |
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
| 02 centerline_hinge (w 1.0) | 0.025 | 3.03 | 58.6 | 1.36 | +0.06 | 0.74 / 0.89 / 0.97 / 0.95 | 4662 (0.99), 1.36 M voxels | 0.262 |
| 03 edge_hinge (w 1.0) | 0.023 | 3.08 | 77.9 | 1.23 | +0.04 | 0.78 / 0.92 / 0.94 / 0.98 | 2545 (1.00), 1.43 M voxels | 0.250 |
| 01b centerline_sdf (w 0.15) | 0.369 | 1.38 | 0.95 | 5.78 | -0.38 | 0.14 / 0.45 / 0.85 / 1.00 | 847 (0.15), 14.9 k voxels | 0.086 |
| 02b centerline_hinge (w 0.15) | 0.338 | 1.43 | 0.87 | 3.47 | -0.50 | 0.10 / 0.37 / 0.74 / 1.00 | 882 (0.46), 16.0 k voxels | 0.094 |
| 03b edge_hinge (w 0.15) | 0.177 | 1.63 | 1.69 | 9.66 | -0.58 | 0.09 / 0.21 / 0.42 / 0.85 | 1724 (0.61), 29.4 k voxels | 0.111 |
| 04b all three (w 0.15 each) | 0.083 | 2.04 | 6.85 | 6.81 | +0.01 | 0.30 / 0.47 / 0.54 / 0.60 | 1175 (0.98), 168 k voxels | 0.130 |
| 05b radius (w 0.15) | 0.207 | 1.69 | 1.95 | 7.96 | -0.24 | 0.12 / 0.35 / 0.66 / 0.87 | 1660 (0.74), 26.6 k voxels | 0.129 |

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
Pilot 02 (job 46344, 2.0 h): DEGENERATE in the same way (volume ratio 59), consistent
with the weighting hypothesis. 03 and 04 started anyway once GPUs freed; 04 (union of
weight-1.0 terms) was cancelled after 1.5 min as uninformative; 03 left to finish.
Interim (01b at 21k of 40k steps, screening): SDF L1 500-step means 0.192 / 0.134 /
0.123 / 0.110 / 0.102 / 0.097 at 0 / 2k / 5k / 10k / 15k / 20k steps vs 00: 0.185 /
0.136 / 0.130 / 0.123 / 0.120 / 0.124 and 01: 0.241 / 0.179 / 0.156 / 0.158 / 0.166 /
0.177. 03 (weight 1.0) rises after 10k like 01. Weight hypothesis supported (not
proven: one seed, 100 subjects). The Phase 2 screening set is therefore the
point-weight variants 01b-05b (configs 02b-05b added; 05 cancelled before start).
Note: 01b's SDF L1 falls below 00's, so the centreline points may also help the SDF
fit itself; whether this reflects more points per step (1.15x compute) is not
separated (hypothesis).

Final 01b vs 00 (screening, 6 training subjects, medians; table above; TSV
sdf_only/pilots/p2_summary.tsv). Facts: Dice 0.18 -> 0.37, band MAE 1.73 -> 1.38 mm,
volume ratio 1.39 -> 0.95, SDF L1 0.139 -> 0.086, centreline inside fraction up in
every radius bin (floor 0.08 -> 0.14, >= 1.6 mm 0.75 -> 1.00); all-size component
ratio 24.7 -> 15.4, but >= 10-voxel component ratio unchanged (5.4 -> 5.8) and
largest-fraction difference unchanged (-0.33 -> -0.38). 1 mm atlas: fewer components
(1658 -> 847) but the largest is smaller (0.55 -> 0.15) with fewer voxels.
Interpretation (hypothesis): the term improves the SDF fit and thin-vessel recall but
does not by itself fix fragmentation at pilot scale; individual fits remain far
from the reference (Dice 0.37). 03 (weight 1.0) collapsed like 01/02.
02b (job 46530): Dice 0.34, fewest >= 10-voxel pieces so far (ratio 3.47 vs 5.40 for
00), but the largest piece is relatively smaller (difference -0.50 vs -0.33) and the
volume is lower (0.87); centreline inside fractions between 00 and 01b. Mixed:
fewer mid-size fragments without a larger main component (fact); no ranking yet.
03b (job 46531): no improvement over 00 on individual fits (Dice 0.18, centreline
inside fractions at baseline level), more >= 10-voxel pieces (ratio 9.66) and a
relatively smaller largest piece (-0.58); volume up (1.69); SDF L1 lower (0.111).
The 1 mm atlas is slightly larger and more connected (largest 0.61 vs 0.55), but
atlas metrics at this scale are one decoding (fact). Edge hinge: not promising at
pilot scale.
04b (job 46532, 2.5 h): PARTIAL INFLATION: volume ratio 6.85, Dice 0.08, atlas 168 k
voxels; its higher floor inside fraction (0.30) and largest-fraction difference
(+0.01) follow from the inflated volume, not better fits (interpretation). The three
terms together add 0.45 x SDF points whose targets are all <= ~0 (hinges push down
only), i.e. the 01-03 mechanism at smaller scale (hypothesis). The union is not a
candidate as configured.
05b (job 46533): small gains in Dice (0.21) and centreline recall, volume up (1.95),
more >= 10-voxel pieces (7.96), largest-fraction difference closest to 0 among
non-inflated pilots (-0.24).

#### Phase 2 screening conclusion (2026-10-31; screening evidence only)

Facts (100 subjects, 40k steps, seed 42, 6 training subjects):
1. Weight 1.0 per graph term (~6.7x per point) collapses the field to mostly
   negative for all three terms tested (01, 02, 03). Per-point weight equal to an
   SDF sample (0.15) does not collapse for any single term.
2. 01b (centreline SDF, the user's default term) gives the best individual fits:
   Dice 0.18 -> 0.37, band MAE 1.73 -> 1.38 mm, volume 0.95, SDF L1 0.139 -> 0.086,
   centreline inside fraction up in every radius bin.
3. 02b (node hinge) is second on fits (Dice 0.34) and the only variant with fewer
   >= 10-voxel pieces than 00 (3.47 vs 5.40).
4. No non-inflated variant makes the largest component relatively larger than 00's
   except 05b (-0.24 vs -0.33), which fits worse. Fragmentation is not solved by any
   single term at pilot scale.
5. 03b (edge hinge) gives no gain; 04b (union) partly inflates.
Selection: 01b is the Phase 2 candidate (default term, best fit, no collapse).
Caveats: one seed; 6 subjects; 01b adds 15% points per step (compute not matched);
pilot fits are far from converged (00 Dice 0.18 vs full-run status TBD by 0c).
Next: full 01b run (3 L40S, same schedule as 00) for confirmation (jobs 46636/46637);
Phase 4 field heads screened on top of 01b, one head per pilot (07a-07d, jobs
46632-46635; control = 01b pilot). Field heads are supervised at the SDF samples
and are separate heads, so they cannot shift the SDF directly (by construction).

#### Phase 4 field-head screening (2026-10-31; screening evidence; control = 01b pilot)

`python -m analysis.phase2.summarize_pilots --tag _p4_` (TSV sdf_only/pilots/p4_summary.tsv).

| Pilot | Dice | band MAE mm | vol ratio | comp ratio >=10 (all) | largest diff | inside floor / 0.9-1.2 / 1.2-1.6 / >=1.6 | atlas 1 mm comps (largest), voxels | SDF L1 last 2k |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: |
| 01b control | 0.369 | 1.38 | 0.95 | 5.78 (15.4) | -0.38 | 0.14 / 0.45 / 0.85 / 1.00 | 847 (0.15), 14.9 k | 0.086 |
| 07a centerline_distance | 0.793 | 0.64 | 0.95 | 2.68 (2.0) | -0.23 | 0.77 / 0.98 / 1.00 / 1.00 | 95 (0.23), 3.7 k | 0.036 |
| 07b radius | 0.219 | 1.61 | 1.64 | 8.78 (26.0) | -0.59 | 0.09 / 0.28 / 0.60 / 0.95 | 1585 (0.35), 21.8 k | 0.110 |
| 07c tangent | 0.530 | 1.15 | 0.83 | 3.43 (6.9) | -0.37 | 0.26 / 0.74 / 0.99 / 1.00 | 392 (0.23), 8.9 k | 0.070 |

Facts: 07a improves every individual-fit metric by a large margin over 01b (Dice
0.37 -> 0.79, band MAE 1.38 -> 0.64 mm, >= 10-voxel component ratio 5.8 -> 2.7,
all-size ratio 15.4 -> 2.0, floor-radius centreline inside 0.14 -> 0.77, training
SDF L1 0.086 -> 0.036); 07c improves moderately; 07b is worse than 01b. Checks: no
code difference under vatl_sdf/ between the 01b and 07 frozen trees; the diagnostics
evaluate the SDF head (forward without return_fields). Optimizer is Adam
(loss-scale invariant), so a larger effective loss weight alone is an unlikely
explanation (hypothesis). Proposed mechanism (hypothesis): the distance target is
0 exactly on the centreline and smooth around it, giving the shared backbone a
location signal for one-voxel vessels whose SDF is only barely negative. The 1 mm
mean-latent atlas becomes much smaller (3.7 k voxels): better individual fits did
not give a fuller atlas (a Case 2 pattern at pilot scale; fact at this scale only).
Full 07a run queued (one factor vs full 01b).
07d junction (job 46635): worse than 01b (Dice 0.14, >= 10-voxel ratio 11.5, volume
2.25, atlas 37.9 k voxels). Ranking of heads at pilot scale: distance >> tangent >
control > radius, junction.

#### Full baseline divergence (fact; 2026-10-31)

00_sdf_baseline_full SDF L1 (2k-step means): 0.160 (step 0), 0.112 (20k, epoch 12),
0.111 (30k), then slowly rising: 0.115 (100k, epoch 60), 0.128 (160k), 0.134 (170k);
a MoE balance-loss spike 0.036 -> 0.14 -> 0.18 at steps 180k-200k (epochs 108-120)
with SDF L1 jumping to 0.22, then 0.265 from epoch ~150 to the end (epoch 191 at
12:48). The rigid-transform penalty grows monotonically (0.0003 -> 0.0047). The
final checkpoint is therefore a degraded model. Pilot 00 shows the same slow rise
after 10k steps. Learning rate stays near 1e-4 because the cosine horizon is 1000
epochs while training stops at 199 (fact from config). Causes (hypotheses, not
tested): too-high constant LR for the late phase, MoE routing collapse, drifting
latents/transforms. Consequences:
- 0c is run on epoch 20 (step 33200, near the loss minimum), epoch 60 (99600) and
  the final checkpoint; population maps and centreline density on epoch 20.
- Full 01b and 07a runs are stopped at epoch 59 (all compared at epochs 20/60).
- Stabilizing training (e.g. scheduler horizon = run length, or lower LR) is a
  candidate one-factor ablation for the user to decide; not run.

Full baseline (46332) trajectory (fact): SDF L1 (1k-step means) 0.172 at step 0,
0.118 at 10k, a plateau ~0.111-0.114 from 15k to 87k (epoch ~52) with a slight upward
drift. Pilot 00 shows the same rise after ~10k (0.123 -> 0.137). Not yet explained;
learning rate / MoE routing are candidate explanations (hypotheses).

### Phase 0c on the full 00_sdf_baseline (jobs 46764/46765; 2026-10-31)

`analysis/phase0_graph/diagnose_checkpoint.py --factors 1 2 --atlas-spacings 1 0.5` on
6 training subjects (3 per sex, age quantiles of the 497) at epoch 20 (step 33200)
and epoch 60 (step 99600); final checkpoint pending (job 46766). Outputs:
sdf_only/analysis/20261031_00_baseline_step{33200,99600}/0c.

| Median of 6 | epoch 20 | epoch 60 |
| --- | ---: | ---: |
| Dice native / 2x supersampled | 0.26 / 0.27 | 0.34 / 0.34 |
| band MAE mm (x1) | 1.63 | 1.61 |
| volume ratio (x1) | 0.87 | 1.44 |
| >= 10-voxel component ratio (x1) | 5.5 | 7.1 |
| largest-fraction difference (x1) | -0.61 | -0.32 |
| centreline inside floor / 0.9-1.2 / 1.2-1.6 / >= 1.6 | 0.05 / 0.16 / 0.46 / 0.90 | 0.13 / 0.49 / 0.81 / 0.98 |
| atlas 1 mm: voxels, 26-comps, largest, >= 10 vox | 20.2 k, 707, 0.66, 99 | 23.1 k, 689, 0.64, 104 |
| atlas 0.5 mm: voxels, 26-comps, largest, >= 10 vox | 162 k, 1034, 0.65, 498 | 185 k, 1007, 0.63, 486 |

Conclusion: CASE 1 (individual fits broken). Evidence: on TRAINING subjects the
zero level reaches Dice <= 0.34 with band MAE ~1.6 mm (about two in-plane voxels),
5-7x the reference's >= 10-voxel components, and one-voxel (floor) vessels are
almost entirely missed (5-13% of supported floor nodes inside vs 95% for the
reference). 2x supersampling changes nothing (not a sampling artefact). The atlas
is not more fragmented than the individual decodings (largest 0.63-0.66 of its
voxels; 0.5 mm grid gives the same connectivity as 1 mm), so no additional Case 2
failure is visible at this stage; Case 2 cannot be judged properly while
individual fits are this poor. Epoch 60 fits better than epoch 20 although the
training SDF L1 is slightly higher (0.115 vs 0.111; fact; reason not established).
Relation to screening: the pilot 07a centreline-distance head reached Dice 0.79
on its (different) 6 pilot subjects, so the Phase 4 head directly targets the
Case 1 failure; the full 07a run (stop epoch 59) tests this at scale.

Final checkpoint (epoch 200, step 332000; job 46766): collapsed field, median Dice
0.02, volume ratio 36.8 (like the weight-1.0 pilots) -> confirms the divergence.

### Phase 3 / Phase 5 on baseline epoch 20 (job 46764; 2026-10-31)

Population maps (vatl_sdf.population_maps, 1 mm, age 20, each subject decoded with
its own age/sex; runtime 850 s for 72 subjects with weight >= 1e-4): mean-latent
atlas 20.2 k voxels / 707 comps (largest 0.66); frequency >= 0.1: 26.4 k voxels,
754 comps, largest 0.93; >= 0.25: 3.1 k voxels; >= 0.5: 18 voxels. Individual
decodings barely overlap above 50%. Outputs:
sdf_only/analysis/20261031_00_baseline_step33200/population_maps_1mm.
Centreline density / recurrence (497 subjects, 300 nodes each, rigid transforms
from the checkpoint; 252 s): median fraction of other subjects with a centreline
within 1 / 2 / 3 mm: floor 0.02 / 0.08 / 0.17; 0.9-1.6 mm 0.03 / 0.12 / 0.24;
>= 1.6 mm 0.03 / 0.12 / 0.25. Recurrence barely depends on calibre.
Learned rigid transforms (fact): translation median 0.09 mm (95%: 0.17 mm) at epoch
20 and 0.21 mm (0.42) at epoch 60; rotation median 2.6 deg (4.6) and 4.2 deg (7.8).
Translations are parameters in mm with lr 1e-4, so they effectively stay at 0:
template space = each image's bbox centre, i.e. no real registration.
Interpretation (hypothesis, strongly suggested): low recurrence, even for large
vessels that every subject has, and the near-empty frequency >= 0.5 map are
dominated by misalignment, not anatomical variability. Phase 3/5 statistics are
therefore not yet interpretable as population variability; they need a
correspondence step (design doc section 6) or a working transform (candidate
one-factor ablation: translation in normalized units or a higher lr_transform;
not run, user decision).

### User decisions 2026-10-31 (afternoon): alignment and divergence

1. Alignment must be learned by the model (no external pre-registration). Subjects share
   orientation; residual offsets of a few mm and head-size differences must be learned.
   Programme: Step 0 data analysis -> Step 1 switchable transform/staging code -> Step 2 pilots
   A1-A4 on top of 07a (100 subjects, 40k steps, seed 42). Success = large-vessel recurrence
   at 3 mm far above the current value without worse individual fits. Report after Step 2.
2. Scheduler horizon = run length in all NEW pilots and runs (`optimizer.scheduler_epochs: run`).
   The full 01b/07a runs (46767/46769, started before the decision) keep the 1000-epoch horizon.

### Alignment Step 0 (job 46799, CPU, 497 training subjects; data only)

`analysis/alignment/step0_mask_geometry.py`; outputs sdf_only/analysis/20261031_alignment_step0.
Vessel = SDF < 0 (identical to label 1). Facts:
- Vessel centroid offset from the image bbox centre (the model's frame), std x / y / z =
  1.8 / 5.6 / 2.8 mm (mean -0.3 / +1.6 / -4.1); distance from the mean offset: median 5.0 mm,
  p95 12.2 mm, max 17.4 mm.
- Principal extents (sqrt eigenvalues) 32.4 / 27.9 / 19.9 mm, CV 8-11% (head/vessel-tree size
  variation of ~10%). Vessel volume 12.2 +- 2.0 ml.
- Principal axes: the smallest-extent axis is within median ~8 deg (mean 9.8, p95 22.6) of the
  population axis; axes 1 and 2 have similar extents (32 vs 28 mm), so their angles (mean 34-36
  deg) mostly reflect axis swapping, not rotation (interpretation).
- Large-vessel recurrence (NEW definition, vatl_sdf/alignment_metrics.py: centreline along
  edges whose endpoints both have graph radius >= 1.2 mm, densified at 0.4 mm; fraction of other
  subjects with a LARGE-vessel centreline within d), median over queries:
  | frame | 1 mm | 2 mm | 3 mm |
  | --- | ---: | ---: | ---: |
  | bbox-centred (identity), all 497 | 0.010 | 0.046 | 0.107 |
  | centroid-translated, all 497 | 0.014 | 0.062 | 0.143 |
  | bbox-centred, first 100 | 0.010 | 0.040 | 0.101 |
  | centroid-translated, first 100 | 0.010 | 0.061 | 0.141 |
  The earlier "~0.25 at 3 mm" (Phase 5 centerline_density) counted other subjects' centreline
  of ANY calibre, so it is not comparable. Re-scored with the new definition (job 46807), the
  full baseline's epoch-20 learned rigid transforms give 0.010 / 0.051 / 0.121 (first 100) and
  0.012 / 0.052 / 0.121 (all 497): between identity and centroid translation. The bbox and
  centroid rows of job 46807 reproduce Step 0 exactly (consistency check).
  Reference values for Step 2 (first 100, 3 mm): identity 0.101, baseline learned 0.121,
  centroid 0.141.
Interpretation (hypothesis): centroid translation alone gains little (0.107 -> 0.143), so
rotation/scale and genuine anatomical variability also matter.

### Divergence of the full baseline: what blows up first (job 46800)

`analysis/divergence/inspect_run.py` on all 20 checkpoints; outputs
sdf_only/analysis/20261031_divergence_00_baseline. Facts (in time order):
1. From epoch 10 on, with no plateau (LR ~ constant 1e-4, no weight decay): expert weight norms
   grow monotonically (MoE layer 2 expert 0: 15.4 -> 42.4 at epoch 100 -> 60.7 at 200); latent
   abs max 0.15 -> 0.36 (epoch 60) -> 0.64 (epoch 100); modulator weight norm 5.9 (epoch 20) ->
   9.8 (epoch 100); rotations creep (median 1.8 -> 5.2 deg).
2. SDF L1 leaves its reference band (mean +- 4 std of steps 20k-60k) at step 102k (epoch 61).
3. Epochs 100-120: latent norm median 3.7 -> 6.7 (max 4.3 -> 11.1), modulator 9.8 -> 23.7;
   MoE layer 1 routing becomes imbalanced (max expert importance 0.30 -> 0.41 at epoch 110);
   MoE balance loss leaves its band at step 173k (epoch 104).
4. Epochs 110 -> 120: gradient explosion: Adam second moments jump by 1e5 (backbone) to 1e7
   (transforms) and keep growing to 1e16-1e19 by epoch 200; latent norms reach 11.6 (max 33).
Conclusion: unregularized parameter growth (latents, modulator, experts) under a constant LR
comes first; loss degradation follows; MoE routing imbalance and the gradient explosion come
last. Mechanism (hypothesis): growing SIREN weights/latents raise effective frequencies until
optimization becomes unstable. Mitigations now in new runs: decaying LR (horizon = run length);
A3/A4 add an L2 latent prior. Weight decay is a further candidate (not run).

### Alignment Step 1 (code; commit ea97105; tests/test_alignment.py, 70 tests pass)

- `model.transform`: mode legacy (default, unchanged parameter table and optimizer groups) |
  normalized (rotation rad, translation in bbox half-lengths, separate Adam groups lr_rotation /
  lr_translation / lr_scale); type rigid | affine (log-scale + shear, `affine_reg_weight` on
  their mean square); `mean_identity` (effective params minus their population mean: fixes the
  global gauge); `centroid_init` (translation_i = mean(c) - c_i from vessel centroids).
- `alignment` section: stage 1 for `epochs` epochs with latents gated to zero (all subjects decode
  one template) and loss = SDF L1 at a wider truncation (`sdf_truncation_mm`, head units
  unchanged) + L1 of a dedicated head predicting distance to the large-vessel centreline (local
  radius = -reference SDF >= 1.2 mm); stage 2 = normal losses + `latent_prior_weight` x mean
  squared latent norm. Latents start at zero.
- `optimizer.scheduler_epochs: run`; `transforms.jsonl` per epoch (rotation, translation,
  scale, anisotropy, latent norms); pilot launcher adds `analysis/alignment/evaluate_alignment.py`
  (transform stats + recurrence in bbox / centroid / learned frames).
- SDFModel.maps_mm gives template_mm = A_i (world - image centre_i) + t_i for all modes;
  centerline_density uses it.

### Alignment Step 2 pilots (submitted; frozen b6ac683; 100 subj, 40k steps, seed 42)

| Pilot | Config | One change vs |
| --- | --- | --- |
| C0 | 08_c0_runhorizon | 07a: scheduler horizon = run length (control) |
| A1 | 09_a1_normalized_transform | C0: normalized-unit transforms (+ mean identity; same numeric lr 1e-4) |
| A2 | 10_a2_centroid_init | A1: centroid initialization |
| A3 | 11_a3_staged | A1: staged training (10 of 40 epochs alignment stage) |
| A4 | 12_a4_staged_affine | A3: regularized affine |

Results (screening evidence; fit metrics = medians over the 6 diagnostic subjects; recurrence = first 100
subjects, large vessels r >= 1.2 mm, fraction of other subjects within d; identity/centroid refs 0.101/0.141 at 3 mm):

| Pilot | Dice | Band MAE mm | Comp ratio >=10 | SDF L1 | Atlas vox / largest | Rot deg med | Transl mm med | Learned rec 1/2/3 mm |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 07a (old 1000-ep horizon) | 0.79 | 0.64 | 2.68 | 0.036 | 3.7k / – | – | 0.1 | – |
| C0 | 0.929 | 0.52 | 1.41 | 0.022 | 3141 / 0.28 | 1.74 | 0.07 | 0.010/0.051/0.111 |
| A1 | 0.930 | 0.53 | 1.34 | 0.023 | 3679 / 0.30 | 1.65 | 1.95 | 0.010/0.071/0.162 |
| A2 | 0.934 | 0.48 | 1.22 | 0.021 | 4558 / 0.31 | 1.65 | 5.37 | 0.020/0.091/0.202 |
| A3 | 0.911 | 0.59 | 1.52 | 0.026 | 3982 / 0.24 | 4.15 | 5.30 | 0.030/0.121/0.263 |
| A4 | 0.921 | 0.54 | 1.39 | 0.024 | 4235 / 0.29 | 4.48 | 5.29 | 0.030/0.141/0.313 |

- Fact: run-length horizon alone (C0 vs 07a) raises Dice 0.79 -> 0.93 and lowers the component ratio
  2.7 -> 1.4. Transforms stay near identity (translation 0.07 mm); learned recurrence 0.111 at 3 mm is
  below the centroid-only reference (0.141). Atlas: 95 components, largest 0.28.
- Fact (A1 vs C0): normalized-unit translations move (median 1.95 mm vs 0.07 mm) with unchanged fits;
  learned recurrence at 3 mm 0.162 (mean 0.176) vs C0 0.111 and centroid-only 0.141. Gain is modest, not
  "far above" the references.
- Fact (A2 vs A1): centroid init gives learned 3 mm recurrence 0.202 (mean 0.218) and the best fits
  (Dice 0.934, component ratio 1.22, atlas 4558 voxels).
- Fact (A3 vs A1): staged training gives learned 3 mm recurrence 0.263 (mean 0.283; 1/2 mm 0.030/0.121,
  3x/2x the identity frame) with rotations 4.2 deg and translations 5.3 mm; fits slightly worse
  (Dice 0.911 vs 0.930, component ratio 1.52 vs 1.34, SDF L1 0.026 vs 0.023). Transforms move almost only in
  the alignment stage (translation median 1.3/2.5/3.8/5.1 mm at epochs 0/2/5/9; 5.2 -> 5.3 mm afterwards), and
  were still moving when it ended at epoch 10. Hypothesis: a longer alignment stage would go further.

### Phase 0c on the full 01b and 07a runs (jobs 46829–46832; old 1000-epoch horizon, stopped at epoch 59)

Medians of the same 6 training subjects as the baseline 0c (x1 grid unless stated):

| | baseline e20 | baseline e60 | 01b e20 | 01b e60 | 07a e20 | 07a e60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dice (x2) | 0.26 (0.27) | 0.34 (0.34) | 0.24 (0.24) | 0.25 (0.25) | 0.24 (0.25) | 0.29 (0.29) |
| band MAE mm | 1.63 | 1.61 | 1.67 | 1.71 | 1.62 | 1.47 |
| volume ratio | 0.87 | 1.44 | 1.21 | 2.27 | 1.34 | 1.80 |
| >= 10-voxel component ratio | 5.5 | 7.1 | 10.3 | 9.5 | 10.1 | 8.4 |
| largest-fraction difference | -0.61 | -0.32 | -0.59 | -0.17 | -0.51 | -0.38 |
| centreline inside given ref inside | 0.11 | – | 0.13 | 0.26 | 0.16 | 0.30 |
| atlas 1 mm: voxels, 26-comps, largest, >= 10 vox | 20.2 k, 707, 0.66, 99 | 23.1 k, 689, 0.64, 104 | 31.8 k, 1461, 0.43, 197 | 42.4 k, 1992, 0.76, 205 | 30.6 k, 940, 0.61, 229 | 38.0 k, 833, 0.66, 191 |

- Fact: at full scale 01b is still Case 1 (Dice <= 0.25, MAE ~1.7 mm), unlike its pilot (Dice 0.37) and far
  from the run-length-horizon pilots (Dice ~0.93). The graph term does not fix individual fits at full
  scale under the old horizon; its volume grows (2.3x at epoch 60) and fragmentation is about 1.5x the baseline's.
- Hypothesis (untested): the gap between pilots and full runs is dominated by the learning-rate schedule
  (near-constant lr under the 1000-epoch horizon) and by per-subject optimisation budget, not by the graph term.
- Fact: full 07a is also Case 1 (Dice 0.24/0.29, MAE 1.6/1.5 mm), slightly better than full 01b at epoch 60
  and with a less fragmented atlas (833 vs 1992 components), but far from its pilot (Dice 0.79) and the
  run-length-horizon pilots (~0.93).

### Step 2 conclusion (screening evidence, 100 subjects, 6 diagnostic subjects for fits)

- A4 (A1 + staged + regularized affine) gives the highest large-vessel recurrence: 0.030/0.141/0.313 at
  1/2/3 mm (3 mm mean 0.331), vs 0.101 identity, 0.141 centroid-only, 0.111 C0 (2.2x centroid-only, 2.8x C0).
  Fits are close to C0 (Dice 0.921 vs 0.929, MAE 0.54 vs 0.52 mm, component ratio 1.39 vs 1.41, SDF L1 0.024
  vs 0.022). Learned scale p5-p95 0.955-1.050, anisotropy median 1.06, rotation 4.5 deg, translation 5.3 mm.
- Ranking of single changes on recurrence (3 mm): staging (A3 vs A1: +0.10) > centroid init (A2 vs A1: +0.04)
  ~ normalized units (A1 vs C0: +0.05) ~ affine (A4 vs A3: +0.05; also recovers fits lost by staging).
- The user's "~0.25" used any-calibre matching; on the large-vessel definition the matched references are
  0.101 (identity) and 0.111 (C0). A4's 0.313 is numerically above 0.25 on the stricter definition.
- Not tested: combining centroid init with staging; a longer alignment stage (transforms still moving at
  epoch 10); behaviour at full scale.

### User decisions 2026-10-31 (evening): full-scale A4, generalization test, probabilistic atlas

Base recipe A4 with run-length horizon. Full run on all 497 training subjects, seed 42, ~60k steps, alignment stage
kept at 10 epochs, 3 L40S as the previous full runs; scale-up check (not an ablation). Test-time fitting of the
first 5 validation IDs (YAML order) with a frozen network (alignment stage, then latent + L2 prior); 0c metrics on
test and the 6 training diagnostic subjects; recurrence of test subjects against training subjects. Probabilistic
atlas (occupancy, soft occupancy tau 0.5 mm, large-vessel centreline probability within 1 mm) on the final and
end-of-alignment checkpoints, with connectivity, calibration (centroid-only control) and coverage. Training
objective unchanged. No per-job confirmation; update CONTEXT and commit after each sub-step; report at the end.

### Full A4 run (job 46841; config 13_a4_full; frozen 0ac21bc)

- Length: 36 epochs x 1660 steps (3 GPUs, 497 subjects) = 59,760 steps (240 short of 60k so that the run is a whole
  number of epochs; the scheduler is per epoch). Cosine horizon 'run' = 36 epochs (printed by the job).
  Alignment stage: epochs 0-9 = 16,600 steps (28% of the run; pilot 25%). Checkpoints at steps 16,600 (end of the
  alignment stage), 33,200, 49,800 and 59,760 (final).
- Factors that differ between the compared runs (none is a one-factor comparison):

| | Step 2 A4 pilot | full A4 (this) | full 07a | full baseline 00 |
| --- | --- | --- | --- | --- |
| subjects / GPUs / subjects per step | 100 / 1 / 5 | 497 / 3 / 15 | 497 / 3 / 15 | 497 / 3 / 15 |
| steps (epochs) | 40k (40) | 59,760 (36) | 99,600 (60, stopped) | 332k (200) |
| per-subject updates (50 per epoch) | 2000 | 1800 | 3000 | 10,000 |
| cosine horizon | run (40 ep) | run (36 ep) | 1000 epochs | 1000 epochs |
| graph term / field head | centreline SDF 0.15 / centreline distance | same | same | none / none |
| transforms | normalized affine, mean identity | same | legacy rigid | legacy rigid |
| staged alignment + latent prior | yes (10 ep) | yes (10 ep) | no | no |

### Test-time fitting and evaluation code (commit a8b603c; tests/test_generalization.py; 82 tests pass)

- vatl_sdf/test_time_fit.py: trained network frozen (checked after fitting); fresh per-subject tables (latent 0,
  identity transform); alignment stage (latent gated to zero, alignment loss) then refinement with the training
  loss (SDF + centreline-SDF graph term + centreline-distance head + L2 latent prior + affine regularization).
  Budget mirrors training per subject: 50 steps per epoch on a fresh 250k-point sample, default 10 + 26 epochs
  (1800 updates), training learning rates, per-epoch cosine with the checkpoint's horizon. The mean-identity
  gauge is NOT applied to new subjects (template frame fixed by training).
- analysis/generalization/evaluate_generalization.py: test = first N YAML val IDs (checked against the manifest and
  the training IDs); groups test / train_refit (6 diagnostic training subjects refitted by the same procedure:
  separates generalization from fitting budget) / train (trained latents); 0c metrics at chosen test-time
  epochs; large-vessel recurrence of each group against the 497 training subjects (trained transforms), with a
  centroid-only control; fitted transforms saved (fitted_test.pt).
- vatl_sdf/probabilistic_atlas.py: `shard` (GPU partial sums of 1[SDF<0] and sigmoid(-SDF/tau) over training
  subjects, uniform weights, template grid 1 mm) and `merge` (CPU: maps, centreline probability, connectivity at
  0.1/0.25/0.5, held-out Brier / Brier skill / smoothed log-likelihood / reliability / ECE over the subject FOV
  and over a vessel ROI, coverage of held-out large-vessel centrelines; learned vs centroid-only frame).
- analysis/divergence/inspect_run.py now reads normalized/affine transforms through maps_mm.

### Full A4 results (screening evidence: one seed, 6 training diagnostic subjects, 5 test subjects)

Stability (inspect_run, losses.jsonl): no divergence. SDF L1 (1k-step means) 0.151 (step 17k, start of refinement)
-> 0.106 (35k) -> 0.0925 (59k), monotonic (the baseline's best was 0.111). Latent norm median 1.25 / 1.26 / 1.24 at
epochs 20/30/36 (plateau). Adam second-moment max of the network group falls 7.9 -> 0.11. Transforms: rotation
5.4 -> 5.7 deg, translation 5.9 -> 6.2 mm, scale p95 1.057 -> 1.044 (almost all motion in the alignment stage).
Watch item (fact): MoE layer-1 gate entropy 0.92 -> 0.64 -> 0.58 -> 0.55 and the MoE balance loss 0.006 -> 0.012
(routing concentrating slowly; importance max stays 0.28 of 4 experts).

Individual fits, 6 training diagnostic subjects (0c metrics, medians, x1 grid):

| | Dice | band MAE mm | volume ratio | comp ratio >=10 | centreline inside (given ref) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 2 A4 pilot (100 subj; its own 6 subjects) | 0.921 | 0.54 | 1.10 | 1.39 | 0.96 |
| full A4, trained latents, final | 0.393 | 1.39 | 1.52 | 11.3 | 0.35 |
| full A4, end of alignment (template only, latent 0) | 0.135 | 2.31 | 0.41 | 0.80 | 0.04 |
| full 07a epoch 60 (old horizon) | 0.294 | 1.47 | 1.80 | 8.4 | 0.30 |
| full baseline epoch 60 | 0.34 | 1.61 | 1.44 | 7.1 | – |

- Fact: at full scale the A4 recipe is still Case 1 (Dice 0.39, MAE 1.4 mm, 11x reference components); better Dice
  than full 07a / baseline, more fragmented. The pilot -> full gap (0.92 -> 0.39) is the dominant effect.
- Hypothesis (untested): per-subject capacity/optimisation at 5x the subjects (497 vs 100; 1800 vs 2000 updates per
  subject; 15 vs 5 subjects per step) rather than the recipe.

Generalization (test-time fitting; network frozen; test = first 5 YAML val IDs, none in training):

| test-time epoch (per-subject updates) | test Dice | test MAE | test vol | test comp | train_refit Dice | train_refit MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10, end of alignment (500) | 0.163 | 1.70 | 1.68 | 12.3 | 0.118 | 1.79 |
| 12 (600) | 0.294 | 1.68 | 2.55 | 11.1 | 0.308 | 1.70 |
| 20 (1000) | 0.356 | 1.65 | 2.56 | 10.0 | 0.371 | 1.67 |
| 36 = training budget (1800) | 0.373 | 1.64 | 2.54 | 10.2 | 0.387 | 1.66 |
| 3x budget, 108 (5400) | 0.404 | 1.63 | 2.60 | 9.0 | – | – |

- Fact: train-test gap at the training budget: Dice 0.373 (test) vs 0.393 (trained) / 0.387 (train subjects refitted
  by the same procedure); MAE 1.64 vs 1.39 / 1.66; volume ratio 2.5 vs 1.5 / 2.6. With the procedure matched
  (train_refit) the gap is ~0; the MAE/volume difference to the trained fits comes from test-time fitting (latents
  of a frozen network inflate the vessels), not from the subjects being unseen. Dice rises slowly with budget
  (0.37 -> 0.40 at 3x).
- Alignment generalization, large-vessel recurrence (references: 497 training subjects, trained transforms; train
  queries: first 100 training subjects; medians, 1/2/3 mm):
  final checkpoint: train vs train 0.046/0.190/0.407; test vs train 0.050/0.207/0.437 (per subject 3 mm 0.32–0.55);
  train_refit vs train 0.032/0.145/0.335; centroid-only control: train 0.135, test 0.153 (3 mm).
  end-of-alignment checkpoint (alignment-only test fit): train 0.327, test 0.394, train_refit 0.283 (3 mm).
- Fact: test subjects align to the training cohort as well as training subjects among themselves (0.44 vs 0.41
  at 3 mm, ~3x the centroid control). Refitted training subjects reach 0.335 < their trained 0.41
  (hypothesis: trained transforms co-adapted with the template over the whole run; 5 vs 6 subjects, so the
  test/train_refit difference is within subject-to-subject spread, per-subject range 0.32–0.55).

Probabilistic atlas (1 mm template grid, 497 subjects, uniform weights; tau 0.5 mm; centreline radius 1 mm):

| map (final) | >= 0.1: vox / comps / largest | >= 0.25 | >= 0.5 |
| --- | --- | --- | --- |
| occupancy p | 38.1 k / 108 / 0.98 | 10.4 k / 3 / 0.91 | 1.7 k / 6 / 0.46 |
| soft occupancy | 135.8 k / 13 / 1.00 | 12.4 k / 5 / 0.91 | 1.4 k / 5 / 0.50 |
| centreline (1 mm) | 1.1 k / 4 / 0.46 | 0 | 0 |
| end of alignment, occupancy | 19.2 k / 5 / 0.81 | 8.4 k / 6 / 0.89 | 2.3 k / 6 / 0.47 |

Deterministic zero-level atlases for comparison: hundreds of components (full 07a e60 833, baseline e60 689).

Held-out calibration (5 test subjects pooled; learned = test-time fitted affine; control = centroid translation):

| final checkpoint | Brier skill FOV | Brier skill ROI | loglik/voxel FOV (constant) | ECE ROI |
| --- | ---: | ---: | --- | ---: |
| occupancy, learned | +0.075 | +0.072 | -0.0134 (-0.0197) | 0.0078 |
| occupancy, centroid control | -0.037 | -0.040 | -0.0148 (-0.0194) | 0.0092 |
| soft tau 0.5, learned | -0.056 | -0.020 | -0.0168 (-0.0229) | 0.024 |
| soft tau 0.5, centroid control | -0.162 | -0.122 | -0.0179 (-0.0225) | 0.024 |

Reliability (occupancy, learned, ROI): over-confident at low-mid p (bin 0.1–0.2: mean p 0.137, observed 0.074;
0.2–0.3: 0.245 vs 0.154; 0.3–0.4: 0.344 vs 0.230), calibrated at 0.5–0.8 (0.544/0.494, 0.646/0.636, 0.733/0.698).
Coverage of test large-vessel centreline points (learned / centroid control): occupancy >= 0.1 85% / 53%,
>= 0.25 63% / 29%, >= 0.5 27% / 10%; centreline map >= 0.1 24% / 7%.
- Facts: the thresholded probabilistic atlas is connected at 0.25 (3 components, largest 0.91) where the
  deterministic atlases had hundreds; it predicts held-out vessels better than a constant only with learned
  alignment (Brier skill +0.07 vs -0.04 for centroid-only); the soft version is worse than the hard frequency
  (sigmoid with tau 0.5 mm spreads mass outside vessels); no voxel reaches centreline probability 0.25 at 1 mm.
- Hypotheses: low-mid over-confidence reflects inflated individual decodings (trained volume ratio 1.5); the empty
  centreline map at 1 mm reflects residual misalignment of several mm (1 mm recurrence ~0.05).
- ROI caveat: the ROI (3-voxel dilation of occupancy > 0 or subject vessel) depends on the atlas, so ROI numbers are
  not comparable between the two checkpoints; FOV numbers are.
- End-of-alignment checkpoint (latents 0, decodings differ only by age/sex): occupancy Brier skill FOV +0.060
  (control -0.018), coverage >= 0.25 51% (control 30%).

Wall-clock: training 3 h 11 (submitted 21:46, ended 00:57). End-of-alignment evaluations ran during training
(22:35 -> 23:18). Final evaluations 00:57 -> 01:55 (gen_final 56 min was the critical path; shards 20 min in
parallel; merge 1 min); gen_long (extra) ended 02:02. Total 4 h 16 from submission to last result (training 75%).

Probabilistic training objective: not proposed at this stage. The failure that dominates is individual fit quality
at full scale (Dice 0.39 vs 0.92 in the pilot), which a variance head or variational latents would not address; the
inference-time atlas already gives connected, positively skilled maps with learned alignment.

### User decisions 2026-11-01: Priority 1 scale gap, Priority 2 probabilistic atlas, graph atlases

Priority 1: explain and fix the pilot (Dice 0.92) -> full-scale (0.39) fit gap; diagnose from existing outputs first,
then one-factor screening ablations measured by 0c on the same 6 training subjects and test-time fitting on the 5
test subjects; recommend and (if promising) run a full-scale recipe with recurrence >= ~0.41–0.44. Priority 2:
probabilistic atlas as the main deliverable (calibration, coverage, centreline radius 2/3 mm, mid-probability
over-confidence); no training-objective change (proposal first if needed). Additional (CPU, inference only):
probabilistic vessel-graph atlas (centreline maps 2/3 mm, population graph with edge occurrence; node/edge atlas if
step 1 works); connectivity atlas (graph vs mask criteria, no repair) and radius atlas (-SDF primary). Monitor MoE
layer-1 gate entropy in all runs.

### Scale-gap diagnosis (inference only on existing outputs; screening evidence)

Factors that differ between the Step 2 A4 pilot and the full A4 run, and what the existing evidence says:

| factor | pilot | full | evidence |
| --- | --- | --- | --- |
| subjects | 100 | 497 | favoured, through memorization (below) |
| subjects per optimizer step / GPUs (DDP) | 5 / 1 | 15 / 3 | not ruled out by existing data; ablation sg_s100g3 |
| per-subject updates | 2000 | 1800 | ruled out as main cause: at matched updates (epoch 20 = 1000) pilot 0.060 vs full 0.107 |
| total optimizer steps | 40k | 59,760 | ruled out (full has more network steps) |
| LR schedule | cosine over 40 epochs | over 36 epochs | ruled out (same base / eta_min, same fraction of run at matched epochs) |
| alignment stage | 10 epochs = 10k steps | 10 epochs = 16.6k steps | ruled out: alignment losses equal at matched epochs (align_sdf 0.207 vs 0.210 at epoch 9), refinement starts at the same SDF L1 (0.159 vs 0.155) and stays equal to epoch 12 (0.123 vs 0.121) |
| MoE routing | layer-1 entropy 1.06 -> 0.68 | 0.92 -> 0.55 | same trend in both; not distinguishing |

Facts:
- Training SDF L1 per epoch (pilot / full): epoch 12 0.123 / 0.121, 15 0.098 / 0.113, 20 0.060 / 0.107, 30 0.032 / 0.099,
  final 0.024 / 0.092. Latent norm median: pilot 0.85 -> 0.77 (shrinks while the loss falls), full 0.93 -> 1.24.
- Per-subject probe (100 shared subjects, median normalized-SDF L1 at epoch 20 / 30 / final):
  pilot own 0.103 / 0.044 / 0.029; latent zeroed 0.255 / 0.281 / 0.289; latent swapped 0.327 / 0.366 / 0.377;
  age/sex condition swapped (own latent) 0.268 / 0.303 / 0.311.
  full own 0.161 / 0.137 / 0.124; latent zeroed 0.210 / 0.228 / 0.236; latent swapped 0.245 / 0.263 / 0.276;
  condition swapped 0.187 / 0.185 / 0.191. Every one of the 100 subjects fits worse at full scale (epoch 30).
  Age + sex is unique for all 100 pilot subjects (468 of 497 at full scale).
- Pilot network, test-time fitting of the 5 test subjects (training per-subject budget): Dice 0.162 / 0.331 / 0.361 /
  0.380 / 0.384 at test-time epoch 10 / 15 / 20 / 30 / 40; MAE 1.19 mm; volume ratio 0.49 — versus 0.921 on its own
  training subjects. Full A4 network on the same test subjects: 0.373 (volume ratio 2.5).
- Pilot network, its own 6 training subjects REFITTED from scratch (latent 0, identity transform, network frozen,
  same procedure as the test subjects): Dice 0.201 / 0.798 / 0.928 / 0.937 / 0.938 at test-time epoch 10 / 15 / 20 / 30 /
  40 (MAE 0.53 mm, component ratio 1.07). Same procedure, unseen subjects: 0.38. Recurrence (3 mm) of the pilot:
  train vs train 0.313, test vs train 0.40, refit vs train 0.343.
  => the stored subjects are recalled from their age/sex condition plus a freshly fitted latent; new subjects are not.
- Conclusion (fact at this scale): the pilot's training fit is memorization. Its network uses the age/sex condition
  (a unique key for 100 subjects) together with the latent to store each subject in the shared weights; swapping
  only the condition destroys the fit as much as removing the latent. That capacity does not transfer to new
  subjects (test Dice 0.38, the same as the full model). At 497 subjects the network cannot memorize, so training
  and test fits agree (0.39 / 0.37). The 'gap' is therefore not a loss of generalizable fit quality at scale; the
  generalizable fit quality is ~0.37–0.40 in both.
- Consequence for Priority 1: recovering 0.92 on training subjects at full scale would require memorization, which
  would not help new subjects. The ablations are therefore ranked by TEST-time fit (5 test subjects) and trained fits
  on the 6 fixed training subjects together; sg_s100nocond tests the condition route causally.

Scale-gap screening ablations (submitted; frozen 73d5426): reference sg_r20 = A4 on 497 subjects, 20 epochs (10
alignment + 10 refinement), 3 GPUs, horizon = run. One factor each vs sg_r20: latent grid 6^3 (sg_latent6), hidden
width 1536 (sg_width1536), 40 epochs (sg_r40), latent lr 2e-3 (sg_latentlr). Subject series (1 GPU, 20 epochs,
the 6 diagnostic subjects always included via data.subset_include): 100 (sg_s100), 250 (sg_s250); 100 with the
condition zeroed (sg_s100nocond); 100 on 3 GPUs = 15 subjects per step (sg_s100g3). All measured with post_ablation.sh
(0c on the 6 fixed subjects, recurrence, test-time fitting with --train-subjects fixed, inspect_run incl. MoE).

### Probabilistic vessel-graph atlas, step 1, on full A4 final (job 46889, CPU 2.5 min; screening evidence)

vatl_sdf/graph_atlas.py (commit 08c27c1, tests/test_graph_atlas.py). Large vessels = graph radius >= 1.2 mm at both
edge endpoints; trained transforms; skeleton of {c_r >= tau} -> population graph; edge occurrence = fraction of
subjects covering >= 50% of the edge points within r. Train = first 100 training subjects with leave-one-out edge
probabilities (the population skeleton itself is not rebuilt without them: slightly optimistic skeleton coverage).

| r / tau | map max | mask vox / comps / largest | skeleton: nodes / edges / length | edge p median / frac >= 0.5 |
| --- | ---: | --- | --- | --- |
| 2 mm / 0.1 | 0.62 | 9.2 k / 3 / 0.85 | 15 / 13 / 283 mm | 0.24 / 0.23 |
| 2 mm / 0.25 | 0.62 | 2.5 k / 3 / 0.47 | 12 / 10 / 130 mm | 0.41 / 0.20 |
| 3 mm / 0.1 | 0.89 | 21.1 k / 2 / 1.00 | 17 / 16 / 362 mm | 0.49 / 0.38 |
| 3 mm / 0.25 | 0.89 | 8.5 k / 4 / 0.86 | 13 / 11 / 234 mm | 0.59 / 0.73 |

Held-out evaluation, medians (test learned / test centroid-only / train LOO learned / train LOO centroid-only):
- r 3 mm, tau 0.1: centreline points within 3 mm of the population skeleton 0.55 / 0.19 / 0.58 / 0.22; points where
  c_3 >= 0.25: 0.71 / 0.35 / 0.69 / 0.30; edge Brier skill 0.18 / 0.12 / 0.22 / -0.03; edge log-likelihood
  -0.60 / -0.64 / -0.58 / -0.74 (constant -0.69); probability-weighted edge recall 0.67 / 0.34 / 0.64 / 0.22.
- r 2 mm, tau 0.1: skeleton coverage 0.32 / 0.07 / 0.34 / 0.09; c_2 >= 0.25: 0.41 / 0.13 / 0.37 / 0.10; edge Brier
  skill 0.22 / -0.28 / 0.12 / -0.28.
- Facts: test subjects score like leave-one-out training subjects (no generalization gap in alignment of large
  vessels); learned alignment beats centroid-only on every measure; the 1 mm centreline map was empty (max < 0.25),
  at 2/3 mm it reaches 0.62/0.89. The population graph is small (13–16 edges, 0.3 m of centreline): only the most
  recurrent large vessels survive the threshold. Step (2) (node/edge atlas) not started yet; decision after the
  connectivity/radius atlas.

### Connectivity, radius and node/edge atlases on full A4 final (jobs 46908/46909, CPU; screening evidence)

vatl_sdf/vessel_attribute_atlas.py (commits 9d8fd5f, 1af803e; tests/test_vessel_attribute_atlas.py with synthetic
tubes: graph-connected across a mask gap, mask-connected with a missing graph edge, radius = -SDF). Population graphs
from the step-1 graph atlas (r 2 mm and 3 mm, tau 0.1). Node presence = large-vessel centreline within r; graph
criterion = path in the subject's full graph (no repair) <= 3 x straight-line distance (min 2 mm); mask criterion =
same 26-component of SDF < 0 (anchor voxel or nearest vessel voxel within 1 mm). Train = first 100 subjects LOO.

Large-vessel selection (per-subject medians, 497 subjects): used = graph radius >= 1.2 mm at both edge endpoints.
Of 14,924 densified centreline points, 884 are graph-selected and 1059 would be selected by -SDF >= 1.2 mm; Jaccard
0.71. Graph-selected points are never outside the mask; -SDF is 0.15 mm below the graph radius on them (median).
80% of centreline points (12,003) have floor graph radius (0.8 mm); their -SDF median is 0.50 mm and 85% lie inside
the mask. Floor-radius points enter the -SDF selection in only 0.4% of cases.

Connectivity (per population edge; medians over edges, 2 mm / 3 mm populations):
- P(both nodes present) 0.10 / 0.25 (per edge 0.04–0.21 / 0.06–0.63).
- P(connected | both): graph 0.98 / 0.97, mask 1.00 / 1.00. P(connected) graph 0.10 / 0.23, mask 0.10 / 0.23.
- Mask-connected but graph-disconnected (given both): 0.01 / 0.02 median; max 0.08 (2 mm) and 0.36 (one 3 mm edge,
  n = 118). Graph-connected but mask-disconnected: 0 for every edge.
- Facts: at the scale of the population edges (large vessels, ~20 mm edges) the two criteria agree for >= 92% of
  subjects on every edge except one; the graph fragmentation documented in Phase 0b (51 components per subject)
  sits mostly in small vessels and barely affects these large-vessel edges. Conditional connectivity is near 1, so
  its likelihood is uninformative (log-lik given both ~ -0.01 to -0.06 for test and train alike); the informative
  quantity is presence.
- Held-out vs LOO (3 mm): edges with both nodes present per subject 7 (test learned) vs 4 (train LOO) vs 1 (test
  centroid-only); unconditional log-likelihood graph -0.46 / -0.46, mask -0.45 / -0.49; coverage of observed
  connected edges by P(connected) >= 0.5: graph 0.25 / 0.25, mask 0.43 / 0.38.

Radius (-SDF primary, graph radius secondary; scale det(A)^(1/3) median 1.00, p5–p95 0.96–1.04):
- Population-node radius medians 1.2–2.3 mm (-SDF), IQR 0.15–0.6 mm, 50–425 contributing subjects per node at
  3 mm; graph radius 0.1–0.2 mm larger. Voxel maps: 38 k voxels, 3.9 k with >= 10 subjects
  (Radius_{sdf,sdf_scaled,graph}_{median,q25,q75}.nii.gz, Radius_subject_count.nii.gz).
- Held-out error (3 mm; test learned / train LOO): -SDF abs error 0.17 / 0.14 mm, |robust z| 0.69 / 0.64, fraction of
  nodes within the 5–95% band 1.00 / 0.91; scaled -SDF 0.13 / 0.14 mm; graph radius 0.11 / 0.11 mm. At 2 mm the test
  -SDF |z| is 1.35 vs 0.69 (fewer matched nodes, n = 5 subjects). Centroid-only control matches fewer nodes
  (different node sets), so its radius errors are not like-for-like.
- Not done: age weighting of the radius statistics (option not implemented).

Node/edge atlas (step 2, built from the same anchors): node occurrence 0.10–0.86 (3 mm), mean anchor position and
covariance per node, edge connection probability for both criteria. LIMITATION (fact): the node anchor is by
construction the nearest large-vessel point within r of the population node, so the positional spread is capped by
r (RMS 1.3 mm at 2 mm, 1.9 mm at 3 mm) and 'inside the 95% ellipsoid' is trivially 1.0 for test and train. A real
positional covariance needs node correspondence that is not defined by the radius window (e.g. junction matching in
each subject's graph); not implemented. Median anchor distance to the node mean: test learned 1.79 mm, train LOO
1.73 mm, test centroid-only 2.21 mm (3 mm population).
