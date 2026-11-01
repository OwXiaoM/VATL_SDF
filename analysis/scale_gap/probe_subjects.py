"""Per-subject fit probe across checkpoints (scale-gap diagnosis; inference only).

For each checkpoint of each run and each probe subject (training subjects common to the runs, same index in both
frames): normalized-SDF L1 (the training metric, truncation sdf.truncation_mm) on a fixed training-distribution
sample, and the same with the subject's inputs swapped, to see what carries subject identity:
  own        own latent, own transform, own age/sex condition (the training fit);
  zero_lat   latent 0 (template), own transform and condition;
  swap_lat   latent of another probe subject (fixed derangement), own transform and condition;
  swap_cond  own latent and transform, age/sex condition of that other subject;
plus per-subject latent norm. Samples are drawn once per subject (seeded) and reused for all checkpoints.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vatl_sdf.config import condition_vector
from vatl_sdf.dataset import load_subject, sample_sdf
from vatl_sdf.losses import normalized_sdf
from vatl_sdf.model import SDFModel


def probe_samples(frame, config, indices, points, seed=0):
    out = {}
    for index in indices:
        subject = load_subject(frame.iloc[index], config)
        result = sample_sdf(subject, config, np.random.default_rng(seed + index))
        pick = np.random.default_rng(seed + 10_000 + index).choice(len(result['sdf_coords']), points, replace=False)
        out[index] = (result['sdf_coords'][pick], result['sdf_target_mm'][pick])
    return out


@torch.no_grad()
def l1(model, config, coords, target_mm, transform_index, latent, condition, device):
    x = torch.as_tensor(coords, device=device)
    ids = torch.full((len(x),), transform_index, device=device, dtype=torch.long)
    x = model.transform_coords(x, ids)
    cond = torch.as_tensor(condition, device=device).reshape(1, 2).expand(len(x), -1)
    pred, _ = model(x, ids, cond, latent_override=latent[None])
    target = normalized_sdf(torch.as_tensor(target_mm, device=device), config['sdf']['truncation_mm']).reshape(-1)
    return (pred[:, 0] - target).abs().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, help='name=run_directory')
    p.add_argument('--output', required=True); p.add_argument('--subjects', type=int, default=100)
    p.add_argument('--points', type=int, default=20000); p.add_argument('--device', default='cuda')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    runs = dict(r.split('=', 1) for r in a.runs)
    frames, first = {}, {}
    for name, directory in runs.items():
        paths = sorted(Path(directory).glob('checkpoint_step_*.pt'))
        state = torch.load(paths[0], map_location='cpu', weights_only=False)
        frames[name], first[name] = pd.DataFrame(state['metadata']), state
    reference = next(iter(frames.values()))
    indices = list(range(min(a.subjects, *(len(f) for f in frames.values()))))
    for name, f in frames.items():
        if f.subject_id.iloc[indices].tolist() != reference.subject_id.iloc[indices].tolist():
            raise ValueError(f'Probe subjects differ between runs ({name})')
    config = next(iter(first.values()))['config']
    samples = probe_samples(reference, config, indices, a.points)
    partner = {i: indices[(k + 1) % len(indices)] for k, i in enumerate(indices)}  # derangement
    rows = []
    for name, directory in runs.items():
        for path in sorted(Path(directory).glob('checkpoint_step_*.pt')):
            state = torch.load(path, map_location='cpu', weights_only=False)
            config, frame = state['config'], frames[name]
            model = SDFModel(config, len(frame)).to(a.device); model.load_state_dict(state['model']); model.eval()
            epoch = state['position']['epoch']
            for i in indices:
                row = frame.iloc[i]; other = frame.iloc[partner[i]]
                own_c = condition_vector(config, row.scan_age, row.sex); other_c = condition_vector(config, other.scan_age, other.sex)
                coords, target = samples[i]
                z = model.latents[i]
                rows.append({'run': name, 'step': state['step'], 'epoch': epoch, 'updates_per_subject': epoch * 50,
                             'subject_index': i, 'subject_id': row.subject_id, 'latent_norm': z.norm().item(),
                             'own': l1(model, config, coords, target, i, z, own_c, a.device),
                             'zero_lat': l1(model, config, coords, target, i, torch.zeros_like(z), own_c, a.device),
                             'swap_lat': l1(model, config, coords, target, i, model.latents[partner[i]], own_c, a.device),
                             'swap_cond': l1(model, config, coords, target, i, z, other_c, a.device)})
            table = pd.DataFrame(rows)
            print(table[(table.run == name) & (table.step == state['step'])][['own', 'zero_lat', 'swap_lat', 'swap_cond', 'latent_norm']]
                  .median().round(4).to_dict() | {'run': name, 'epoch': epoch}, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'per_subject.tsv', sep='\t', index=False)
    summary = table.groupby(['run', 'epoch'])[['own', 'zero_lat', 'swap_lat', 'swap_cond', 'latent_norm']].median()
    summary.to_csv(out / 'summary.tsv', sep='\t')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
