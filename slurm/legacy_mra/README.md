# Legacy MRA + SDF launchers

These scripts drive the two-head MRA/SDF code and the configs in
`configs/ablations/legacy_mra/`. They only work on a checkout of tag
`pre-sdf-only-refactor` (package `mra_sdf_only`), e.g. to export old checkpoints
that contain an MRA head:

    git worktree add ../VATL_SDF_legacy pre-sdf-only-refactor
