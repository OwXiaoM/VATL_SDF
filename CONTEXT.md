# Conversation context and restart point

Last updated: 2026-10-30.

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
