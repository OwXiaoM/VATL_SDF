# Project handoff

Read `CONTEXT.md` before continuing work. It records the conversation through
2026-10-25, observed results, hypotheses, implemented ablations, and restart steps.
Read `configs/ablations/README.md` for exact experiment commands and caveats.

## Current objective

Use **02_sdf_only** as the baseline for all subsequent experiments. Do not
investigate intensity supervision or compare intensity-related ablations.
Corrected splitting is implemented: train only the 497 IDs from
`/leonardo/home/userexternal/xzhang01/data/v1/subject_ids_label_t1_fast.yaml`,
then evaluate the first 3/5 of its 55 validation IDs in YAML order. The original
split file is unchanged. Existing 552-subject checkpoints are contaminated for
this validation cohort and must never initialize the corrected run. See
`configs/evaluation/README.md`; use `slurm/train_sdf_only_split.sh` for a fresh
run, with unchanged SDF-only training hyperparameters. No jobs were submitted.

## Working conventions

- Preserve the original completed run and its outputs.
- Keep ablations controlled: each named variant changes one factor from its
  matched baseline. Start variants from scratch; do not resume them from the
  original baseline checkpoint.
- Distinguish measured evidence from hypotheses. The cause of fragmentation has
  not been established.
- The user intends to launch training one experiment at a time. Do not interpret
  this handoff as a request to submit jobs automatically.
- SDF-only currently means zero intensity loss, not removing the MRA loader,
  coordinate samples, or output head. Explain that distinction when relevant.
- Update `CONTEXT.md` when new jobs, results, conclusions, or decisions arrive.

## Environment

Repository: `/leonardo/home/userexternal/xzhang01/VATL_SDF`.
Use `source slurm/environment.sh` for Python and required HPC modules. Activating
the virtualenv alone did not provide a complete runtime in this session.
A local Git repository was initialized on 2026-10-30; GitHub upload used the connected app. Local SSH authentication was unavailable.
