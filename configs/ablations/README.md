# Controlled vessel ablations

These six full configs use the original baseline settings, with a common stop
after epoch index 199 (200 epochs). The cosine scheduler retains its original
1,000-epoch horizon. Each variant changes one factor from `00_baseline`.
Nothing is submitted automatically.

| Run | Change from matched baseline | Question |
| --- | --- | --- |
| `00_baseline` | None | Matched reference at 200 epochs |
| `01_sdf_weight_1` | SDF weight 0.1 → 1.0 | Does stronger SDF supervision preserve small branches? |
| `02_sdf_only` | MRA weight 1.0 → 0.0 | Does removing intensity supervision help SDF fitting? |
| `03_all_experts` | MoE k 2 → 4 | Does removing hard expert selection reduce breaks? |
| `04_surface_band_1mm` | Surface band 2 → 1 mm | Does concentrating samples nearer the boundary help? |
| `05_latent_grid_5` | Latent shape [256,3,3,3] → [256,5,5,5] | Does more subject-specific spatial capacity help? |

Run from the repository root, one command at a time, waiting for each job to
finish before submitting the next:

```bash
sbatch slurm/train_ablation_4gpu.sh 00_baseline
sbatch slurm/train_ablation_4gpu.sh 01_sdf_weight_1
sbatch slurm/train_ablation_4gpu.sh 02_sdf_only
sbatch slurm/train_ablation_4gpu.sh 03_all_experts
sbatch slurm/train_ablation_4gpu.sh 04_surface_band_1mm
sbatch slurm/train_ablation_4gpu.sh 05_latent_grid_5
```

All start from scratch, use seed 42, the same subjects, 250,000 points per task
per subject, learning rates, and export settings. Compare equal completed epochs,
not equal wall time. All-expert routing and the larger latent grid cost more;
the latter also changes parameter count and initialization, so it is a capacity
experiment, not a parameter-matched comparison. Repeat promising results with
additional matched seeds before treating small improvements as reliable.

## SDF-only semantics

`02_sdf_only.yaml` sets `loss.mra_weight: 0.0`, retaining `loss.sdf_weight: 0.1`
and `loss.moe_weight: 0.01`. MRA targets contribute zero gradient to the total
loss. Keeping the SDF weight fixed isolates removal of intensity supervision.
The existing architecture, MRA loading/sampling, and two output channels remain;
this is SDF-only supervision, not an SDF-only data loader. MRA coordinates still
participate in MoE balancing. Ignore its MRA images and MRA reconstruction loss:
the MRA head is untrained. MRA files must still be present.

For a subsequent practical SDF-only run with unit SDF weight (two changes from
baseline), use:

```bash
sbatch slurm/train_ablation_4gpu.sh 02_sdf_only --set loss.sdf_weight=1.0 run_name=sdf_only_weight1
```

## Training duration and continuation

The original run's saved history ended near 194 epochs. Use the matched
200-epoch suite for screening; 200 epochs is not evidence of convergence. To
run the full 1,000-epoch schedule, apply the same override to every compared run:

```bash
sbatch slurm/train_ablation_4gpu.sh 00_baseline --set training.stop_after_epoch=null
```

A 24-hour allocation may end before the stopping point. Continue each run from
its own latest checkpoint, keeping the same config and GPU count:

```bash
sbatch slurm/train_ablation_4gpu.sh 00_baseline --set resume=/absolute/path/to/checkpoint.pt
```

To continue beyond the common 200-epoch stop, add
`training.stop_after_epoch=null` to that same `--set` list. Resume requires
unchanged data, sampling, model, loss, optimizer, SDF settings, and seed.
Do not resume an ablation from the original baseline checkpoint.

## Separate reconstruction, atlas averaging, and voxelization

Before retraining, diagnose the existing checkpoint. Inside a GPU allocation:

```bash
source slurm/environment.sh
bash slurm/diagnose_vessels.sh /absolute/path/to/checkpoint.pt /absolute/path/to/new_diagnostics IXI002-Guys-0828
```

This exports the subject on its native grid and the same age-20 male atlas at
1 mm and 0.5 mm. The finer grid has eight times as many voxels and larger memory
and runtime requirements. It can reveal undersampling but cannot recover absent
geometry. Native subject and atlas exports have different grids; do not compare
them voxel by voxel. If subjects reconstruct well but atlases fragment, latent
averaging and anatomical correspondence deserve attention before more training.

Evaluate several fixed training subjects spanning age and sex using
`python -m mra_sdf_only.evaluate --help`. Use the native reference MRA/SDF,
`--config` for reference intensity normalization, and `--exported-mask` for
export consistency. For SDF-only runs, assess only SDF/mask metrics. Training
subject reconstructions measure fitting, not held-out generalization.

Compare zero-level Dice, vessel volume, visible branch recovery, and component
sizes/counts with each subject's reference, at matched spacing. Component count
alone is insufficient: false bridges can improve it. Never compare raw
normalized SDF loss as a substitute for topology. Atlas connectivity has no
single subject ground truth. Small positive threshold offsets are diagnostic
only; thickening and false bridges can masquerade as recovery.

These experiments do not add a topology loss or guarantee connectivity. If
individual fitting improves but atlas breaks persist, the next experiment is
an atlas construction/alignment change. If individual fitting remains fragile,
a separate sign/centerline supervision experiment is warranted.
