# SDF + graph configs (no MRA intensity)

`00_sdf_baseline.yaml` (= `configs/config.yaml`) is the matched baseline. Each numbered config changes
exactly one factor from it:

| Config | Change vs 00 |
| --- | --- |
| 01_centerline_sdf | extra SDF samples along mask-supported graph edges, target = reference SDF (default centreline term) |
| 02_centerline_hinge | hinge at snapped supported nodes: relu(pred - ceiling) |
| 03_edge_hinge | hinge along supported edges between snapped nodes |
| 04_all_graph_terms | 01 + 02 + 03 (the one combined variant) |
| 05_radius_regression | \|pred + r\| at supported nodes with graph radius r >= 0.9 mm (never floor nodes) |
| 06_eikonal | eikonal penalty on the SDF gradient in mm |

Graph support rule (Phase 0a, user decision): a node is supported if within 0.4 mm of the voxelized
vessel mask (label 1); `graph.support_strict: true` switches to the nearest-voxel rule.
Phase 4 field heads live in the `graph_fields` section (all disabled in 00).

## Commands

All jobs: run from a frozen worktree so queued jobs never import live edits:

```bash
C=$(git rev-parse --short HEAD); git worktree add --detach /users/zhangx/VATL_SDF_frozen/$C $C
export VATL_CODE_DIR=/users/zhangx/VATL_SDF_frozen/$C
O=/medias/medical/caravel/zhangx/training_outputs/sdf_only
```

Full run (DDP; all compared full runs must use the same GPU count, currently 3 L40S), then the
continuation (exits immediately if the run already passed `stop_after_epoch`):

```bash
J=$(sbatch --parsable --gres=gpu:l40s:3 --export=ALL $VATL_CODE_DIR/slurm/train_sdf.sh configs/sdf/00_sdf_baseline.yaml --set output_root=$O/runs/00_sdf_baseline_full)
sbatch --dependency=afterany:$J --gres=gpu:l40s:3 --export=ALL $VATL_CODE_DIR/slurm/continue_sdf.sh configs/sdf/00_sdf_baseline.yaml $O/runs/00_sdf_baseline_full
```

Screening pilot (1 GPU, first 100 training subjects, 40k steps, seed 42, then diagnostics):

```bash
sbatch --export=ALL -o $O/pilots/<date>_p2_<name>/job_%j.log $VATL_CODE_DIR/slurm/pilot_sdf.sh \
  configs/sdf/<name>.yaml $O/pilots/<date>_p2_<name> 100 40000
python -m analysis.phase2.summarize_pilots --output $O/pilots/p2_summary.tsv
```

Analyses of a finished full run (0c diagnostics, population maps, centreline density):

```bash
sbatch --export=ALL -o $O/analysis/<dir>/job_%j.log $VATL_CODE_DIR/slurm/post_baseline.sh $O/runs/<run> $O/analysis/<dir>
```

Pilot results are screening evidence (100 subjects, one seed), not conclusions.
