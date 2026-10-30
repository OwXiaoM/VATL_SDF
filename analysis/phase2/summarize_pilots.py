"""Aggregate Phase 2 pilot diagnostics into one table (screening evidence: 100 subjects, 40k steps, 1 seed).

Per pilot: medians over the 6 fixed training subjects of zero-level Dice, band MAE, volume ratio,
reference-relative connectivity (component ratio for >= 10-voxel pieces, largest-fraction difference),
centreline inside fraction by radius bin (prediction; reference shown once), 1 mm atlas components,
and the mean SDF L1 over the last 2000 training steps.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(directory):
    diag = json.loads((directory / 'diagnostics' / 'diagnostics.json').read_text())
    losses = [json.loads(l) for l in next((directory / 'runs').glob('*/losses.jsonl')).read_text().splitlines()]
    rows = []
    for s in diag['subjects']:
        g = s['grid_x1']; c = g['connectivity']; cl = s['centerline']
        rows.append({'dice': g['zero_level_dice'], 'band_mae_mm': g['band_mae_mm'], 'volume_ratio': g['volume_ratio_pred_over_ref'],
                     'comp_ratio_ge10': c['component_ratio_ge_min'], 'comp_ratio_all': c['component_ratio'],
                     'largest_frac_diff': c['largest_fraction_difference'],
                     **{f'inside_{k}': cl[k]['pred_inside'] for k in ('floor', '0.90-1.20', '1.20-1.60', '>=1.60')},
                     'inside_given_ref': cl['pred_inside_given_ref_inside'],
                     **{f'ref_inside_{k}': cl[k]['ref_inside'] for k in ('floor',)}})
    table = pd.DataFrame(rows)
    out = table.median().to_dict()
    atlas = diag.get('atlas', {}).get('1mm', {})
    out.update({'atlas_c26': atlas.get('c26', {}).get('count'), 'atlas_largest': atlas.get('c26', {}).get('largest_frac'),
                'atlas_voxels': atlas.get('voxels'), 'sdf_l1_last2000': float(np.mean([r['sdf_raw'] for r in losses[-2000:]])),
                'steps': losses[-1]['step'],
                'hours': (losses[-1]['time'] - losses[0]['time']) / 3600 if 'time' in losses[0] else None})
    for key in losses[-1]:
        if key.endswith('_raw') and key not in ('sdf_raw', 'moe_raw', 'transform_raw'):
            out[key + '_last2000'] = float(np.mean([r[key] for r in losses[-2000:]]))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default='/medias/medical/caravel/zhangx/training_outputs/sdf_only/pilots')
    p.add_argument('--prefix', default='20261030_p2_'); p.add_argument('--output')
    a = p.parse_args()
    rows = {}
    for d in sorted(Path(a.root).glob(a.prefix + '*')):
        if (d / 'diagnostics' / 'diagnostics.json').exists() and 'atlas' in json.loads((d / 'diagnostics' / 'diagnostics.json').read_text()):
            rows[d.name[len(a.prefix):]] = summarize(d)
    table = pd.DataFrame(rows).T
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(table.round(3).T.to_string())
    if a.output:
        table.to_csv(a.output, sep='\t')


if __name__ == '__main__':
    main()
