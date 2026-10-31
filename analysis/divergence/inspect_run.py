"""What blows up first in a diverging run? Per checkpoint: latent norms, transform magnitudes, module weight
norms, Adam second-moment state, MoE routing (expert importance, gate entropy, top-1 share) and the fraction
of negative SDF predictions on a fixed probe set (training SDF samples of a few subjects, own latents and
transforms). From losses.jsonl: 1k-step means and the first step at which each logged term leaves its
reference band (mean +- 4 std of the reference window).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vatl_sdf.dataset import SDFGraphDataset
from vatl_sdf.layers import MoESineLayer
from vatl_sdf.model import SDFModel


def probe_set(config, frame, subjects, points, seed=0):
    data = SDFGraphDataset(config, frame)
    coords, ids, conds = [], [], []
    for index in subjects:
        item = data[index]
        pick = np.random.default_rng(seed + index).choice(len(item['sdf_coords']), points, replace=False)
        coords.append(item['sdf_coords'][pick]); ids.append(np.full(points, index)); conds.append(np.repeat(item['condition'][None], points, 0))
    return [torch.as_tensor(np.concatenate(x)) for x in (coords, ids, conds)]


def checkpoint_stats(state, model, probe, device):
    m = state['model']
    z = m['latents'].flatten(1)
    t = m['transforms']
    out = {'step': state['step'], 'epoch': state['position']['epoch'],
           'latent_norm_median': z.norm(dim=1).median().item(), 'latent_norm_max': z.norm(dim=1).max().item(),
           'latent_abs_max': z.abs().max().item(), 'latent_mean_norm': z.mean(0).norm().item(),
           'rotation_deg_median': torch.rad2deg(t[:, :3].norm(dim=1)).median().item(),
           'translation_mm_median': t[:, 3:].norm(dim=1).median().item()}
    for name, value in m.items():
        if value.ndim >= 2 and ('weight' in name) and 'latents' not in name:
            out[f'norm/{name}'] = value.norm().item()
    groups = state['optimizer']['state']
    for gi, group in enumerate(state['optimizer']['param_groups']):
        sq = [groups[p]['exp_avg_sq'] for p in group['params'] if p in groups]
        if sq:
            out[f'adam_sq_max/group{gi}'] = max(s.max().item() for s in sq)
            out[f'adam_sq_mean/group{gi}'] = float(np.mean([s.mean().item() for s in sq]))
        out[f'lr/group{gi}'] = group['lr']
    model.load_state_dict(m); model.eval()
    logits = {}
    hooks = [layer.gate.register_forward_hook(lambda mod, inp, outp, k=k: logits.setdefault(k, []).append(outp.detach()))
             for k, layer in enumerate(l for l in model.backbone if isinstance(l, MoESineLayer))]
    coords, ids, conds = (x.to(device) for x in probe)
    with torch.no_grad():
        preds = torch.cat([model(coords[s:s + 50000], ids[s:s + 50000], conds[s:s + 50000])[0] for s in range(0, len(coords), 50000)])
    for h in hooks:
        h.remove()
    out['probe_negative_fraction'] = (preds < 0).float().mean().item()
    out['probe_pred_mean'] = preds.mean().item()
    for k, chunks in logits.items():
        g = torch.cat(chunks)
        soft = g.softmax(-1)
        importance = soft.mean(0)
        out[f'moe{k}_importance_max'] = importance.max().item()
        out[f'moe{k}_importance_min'] = importance.min().item()
        out[f'moe{k}_entropy'] = (-(soft * soft.clamp_min(1e-12).log()).sum(-1)).mean().item()
        out[f'moe{k}_top1_share_max'] = torch.bincount(g.argmax(-1), minlength=g.shape[1]).float().div(len(g)).max().item()
        out[f'moe{k}_logit_abs_mean'] = g.abs().mean().item()
    return out


def loss_onsets(losses, reference, window=1000, sigmas=4.0):
    frame = pd.DataFrame(losses)
    keys = [k for k in frame if k.endswith('_raw')]
    binned = frame.groupby(frame.step // window)[keys].mean()
    binned.index = binned.index * window
    ref = binned.loc[reference[0]:reference[1]]
    onset = {}
    for k in keys:
        lo, hi = ref[k].mean() - sigmas * ref[k].std(), ref[k].mean() + sigmas * ref[k].std()
        after = binned.loc[reference[1]:]
        bad = after[(after[k] > hi) | (after[k] < lo)]
        onset[k] = int(bad.index[0]) if len(bad) else None
    return binned, onset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='run directory with checkpoint_step_*.pt and losses.jsonl')
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--subjects', type=int, nargs='*', default=[0, 100, 200, 300, 400])
    p.add_argument('--points', type=int, default=20000)
    p.add_argument('--reference-steps', type=int, nargs=2, default=[20000, 60000])
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    run = Path(a.run)
    losses = [json.loads(l) for l in (run / 'losses.jsonl').read_text().splitlines()]
    binned, onset = loss_onsets(losses, a.reference_steps)
    binned.to_csv(out / 'losses_1k.tsv', sep='\t')
    paths = sorted(run.glob('checkpoint_step_*.pt'))
    first = torch.load(paths[0], map_location='cpu', weights_only=False)
    config, frame = first['config'], pd.DataFrame(first['metadata'])
    probe = probe_set(config, frame, a.subjects, a.points)
    model = SDFModel(config, len(frame)).to(a.device)
    rows = []
    for path in paths:
        state = torch.load(path, map_location='cpu', weights_only=False)
        rows.append(checkpoint_stats(state, model, probe, a.device))
        print(json.dumps({k: v for k, v in rows[-1].items() if not k.startswith(('norm/', 'adam'))}), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'checkpoints.tsv', sep='\t', index=False)
    # Relative change of every checkpoint statistic vs the reference checkpoint closest to the window end.
    ref_row = table.iloc[(table.step - a.reference_steps[1]).abs().argmin()]
    numeric = table.select_dtypes('number').drop(columns=['step', 'epoch'])
    relative = (numeric / ref_row[numeric.columns].replace(0, np.nan) - 1)
    first_departure = {}
    for c in relative:
        bad = table.step[(relative[c].abs() > 0.5) & (table.step > a.reference_steps[1])]
        first_departure[c] = int(bad.iloc[0]) if len(bad) else None
    summary = {'loss_onset_step': onset, 'checkpoint_first_50pct_departure_step': dict(sorted(first_departure.items(), key=lambda kv: (kv[1] is None, kv[1]))),
               'reference_steps': a.reference_steps}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
