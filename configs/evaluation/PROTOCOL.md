# Individual validation protocol after split correction

The [README](README.md) gives the authoritative 497/55 split, fresh-start training
and evaluation commands. Only a newly trained, train-only SDF checkpoint is valid
for evaluating the existing 55 validation subjects. Original 552-subject models
remain preserved but are ineligible for this validation protocol.

Selection uses the validation YAML list order; `validation_cohort.json` freezes
the first five IDs (N=3 takes its prefix), source hashes, FOV and settings.
Training and evaluation share strict nested split validation. The original split
file is never edited. The CLI flag `--num-test-subjects` denotes the requested
number of validation subjects here; there is no separate final test cohort.

## Two separate inference protocols

- **`zero_shot_atlas`**: decode the training-age-weighted mean latent, conditioned
  on the test subject's known age and sex. The mean uses the existing age-only
  Gaussian/top-N weighting; sex enters the decoder. Rigid transform is identity.
  No test SDF values determine the latent or transform. This is a conditioned
  population atlas mapped onto the native centered coordinate grid, not an
  individualized zero-shot reconstruction or GT-registered atlas.
- **`test_time_sdf_fit`**: initialize from that same latent, allocate a **new**
  latent grid and a new identity rigid transform, then fit them to explicitly
  observed test SDF values. This is **test-time SDF fitting**, not zero-shot
  prediction. Every pretrained parameter, including the modulator, decoder,
  routing parameters, training latents and rigid table, is frozen. Hashes before
  and after verify unchanged state. Only the external subject parameters enter
  Adam; no population model optimization occurs.

The model's `latent_override` bypasses its stored rigid transformations. The
wrapper therefore applies the existing `rigid_transform` explicitly **before**
latent interpolation and decoding. Coordinates reuse `native_grid` exactly:
`(affine @ voxel - affine @ (shape/2)) / (world_bbox_mm/2)`. XYZ rotations are
radians and translation is mm. Optimization can be disabled for rigid parameters
with `optimize_rigid: false` in a versioned settings file. No unseen subject is
assigned a training-table index for latent/transform lookup.

## Observations and held-out evaluation

All defaults are fixed in `sdf_only.yaml`, not chosen from test metrics. A stable
SHA256 of base seed and subject ID determines the per-subject seed independently
of N and processing order. Native integer voxel coordinates are grouped into
8-voxel blocks. A documented integer spatial hash assigns blocks to five folds:
fold 0 is the adaptation region; the other four folds are evaluation regions.
Coordinates alone determine this partition: no SDF sign, vessel mask or intensity
is used. Up to 100,000 distinct adaptation voxels are chosen with the fixed seed.
Their flat indices are saved. All other voxels in adaptation blocks remain
excluded from primary evaluation, even if not selected as observations.

Evaluation support is the four held-out folds with a one-voxel 26-neighbor guard
removed at observation/FOV boundaries. The same support is used for zero-shot
and fitted protocols. It is saved as a native-grid volume. Fit uses only saved
observation coordinates and clipped SDF targets; its function never receives
held-out values. Defaults: 500 fixed Adam steps, batches of 4096 drawn from the
fixed observations, latent/rigid LR 5e-4/1e-4, latent deviation and scaled rigid
priors weighted 1e-4 each, rotations bounded ±0.35 rad and translations ±10 mm.
There is no evaluation-based early stopping, best-checkpoint selection or tuning.
Adaptation traces are labeled objectives, never evaluation scores.

## Native outputs and metric interpretation

SDF references must have a finite, nonsingular native affine and explicit mm
units. SDF sign is negative inside; exports are predicted SDF in mm and the exact
binary threshold `SDF < 0`. Shape, affine, qform/sform and spacing are preserved
and verified on reload. Valid FOV is determined only by the original centered
coordinates within [-1,1]. Outside that FOV, SDF is NaN, mask is zero, and no
voxels are scored. No automatic resampling or threshold offsets are applied.
The default reference mask is `reference SDF < 0`. An optional **binary**
`VesselMask` TSV column must agree with that sign and share the same physical
grid; a multi-tissue `Seg` column is not silently treated as a vessel mask.

Two metric scopes are saved separately:

- **`heldout_spatial_folds` (primary):** voxel Dice, precision, recall, evaluated
  vessel volumes, raw SDF MAE in mm, doubly clipped SDF MAE, and MAE where the
  reference lies within the training truncation band. These are reference-based
  test evaluations, not training losses. The metric clipping/band radius is fixed by `metric_truncation_mm: 5.0`,
  independently of a future checkpoint's training truncation. Raw field errors far beyond truncation
  are not evidence of learned exact distances there.
- **`full_fov_descriptive`:** dense-field scores and component counts/absolute
  largest sizes under **both 6 and 26 adjacency** for prediction and reference.
  These include observed adaptation regions for fitting and cannot be called
  held-out reconstruction/topology accuracy. We do not punch observation holes
  in masks for component counting: that would create artificial fragments.

Surface metrics in each scope use 6-neighbor foreground boundary voxel centers,
extracted from full valid masks **before** restricting evaluation support.
Boundary points at the physical FOV edge are excluded. Distances use physical
world coordinates (KD trees, including arbitrary nonsingular affines). Reported
ASSD is the average of the two directional mean distances; HD95 is the 95th
percentile of concatenated directional distances; surface Dice is the fraction
of both boundary-point sets within the fixed 1 mm tolerance. Held-out surface
scores compare only surfaces within held-out support and exclude its boundary
halo; they are partial-support discrete surface diagnostics, not whole-anatomy
area-weighted mesh metrics. No artificial surfaces are created at observation
holes. The full-FOV scope gives complementary descriptive geometry.

Empty-mask Dice: both empty = 1, one empty = 0. Precision/recall are null when
their denominators vanish. Empty eligible surfaces yield null distance/surface
Dice and an explicit status. Cohort summaries give mean and **sample SD
(ddof=1)** using equal subject weights, with `n_defined` and `n_total`; undefined
values are not replaced by zero. With fewer than two defined values, SD is null.
All aggregate statistics are separated by protocol and scope.

Each subject/protocol directory has native SDF/mask, metrics, and `comparison.png`
(GT green, prediction red, white overlap; matched slices and projections).
Displays cover the full valid FOV and therefore include adaptation regions.
Slice locations use the GT-mask median for display only, never optimization.
Root CSV/JSON metrics and separate `<protocol>_metrics.csv/json` files include
per-subject scores and protocol-separated summaries; `summary.csv` stores means,
SDs and denominators. Provenance includes checkpoint path/hash, source/config
hashes, software versions, seeds, condition, reference paths/hashes, observation
indices, masks and fitted subject parameters. Checkpoint bytes are checked again
at completion. No modification of the checkpoint is needed.

## Execution and reproducibility

Follow the corrected training, checkpoint preflight and N=3/N=5 commands in the
[README](README.md). All evaluation invocations must supply a new train-only
checkpoint and the same `validation_cohort.json` manifest.

The script requests one GPU for evaluation only; nothing is submitted by setup.
Fixed seeds, disabled TF32, deterministic cuDNN and fixed settings are recorded.
**CUDA `grid_sample` backward is not bitwise deterministic**: the GPU script
explicitly uses `--determinism seeded`, warns on nondeterministic operations, and
records this limitation. Do not claim bitwise reproducibility across GPU runs or
hardware/software versions. Strict determinism is the CLI default on CPU;
`--device cpu --determinism strict` is available for exact regression checks.
Strict CUDA fitting raises on unsupported deterministic operations rather than
silently changing behavior. Full-size GPU performance is not yet validated.
Use `--protocols zero_shot_atlas` or `--protocols test_time_sdf_fit` to run just
one protocol; they remain separate in all records.

Lightweight validation:

```bash
source slurm/environment.sh
python -m unittest -v tests.test_individual
bash -n slurm/evaluate_sdf_only.sh
```

Synthetic tests exercise both N=3/N=5 protocols, identical first-three outputs,
cohort contamination/empty-split rejection, disjoint observations, native
coordinates, rigid mapping, pretrained-state freezing, deterministic external
latent optimization, physical surface distances, empty masks and aggregation.
These tests do not establish real-subject performance or convergence of the
500-step test-time fitting protocol. Future experiments must preserve this
cohort/protocol and keep the 55 validation subjects out of population training.
