"""Detection vs perturbation magnitude from a deviation.json with several manifests (Step C); CPU, reads only.

Rows: learned, centroid (uniform weighting). Columns: radius factor (thin, removal = 0), local stenosis factor,
displacement (mm). Magnitude 'none' = the unperturbed refit of the same (subject, edge) pairs (baseline columns).
Series 'default' (old any, Step B, subject level), 'radius' (radius_absz vs radius_allr, Step E1; population- and
null-referenced element detection) or 'f' (population-referenced: radius_absz, seg_radius, radius_allr and the Step F
families segmatch_med, segmatch_q25, presence3).
Usage: python analysis/deviation/plot_dose_response.py DEVIATION_DIR [OUTPUT_PNG] [default|radius|f]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [('Thinning / removal', 'radius factor', [(1., None), (0.5, 'thin0'), (0.3, 'thin0.3'), (0., 'remove0')]),
          ('Local stenosis (4 mm, tapered)', 'radius factor', [(1., None), (0.5, 'stenosis0.5'), (0.3, 'stenosis0.3')]),
          ('Displacement', 'mm', [(0., None), (2., 'displace2'), (3., 'displace3'), (5., 'displace5')])]
SERIES_RADIUS = [('population_radius_absz', 'radius_absz (pop.)', '#2a78d6'), ('null_radius_absz', 'radius_absz (null)', '#eb6834'),
                 ('population_radius_allr', 'radius_allr (pop.)', '#1baf7a'), ('null_radius_allr', 'radius_allr (null)', '#eda100')]
SERIES_F = [('population_radius_absz', 'radius_absz', '#2a78d6'), ('population_seg_radius', 'seg_radius (B)', '#eb6834'),
            ('population_radius_allr', 'radius_allr (E1)', '#1baf7a'), ('population_segmatch_med', 'segmatch_med (F1)', '#eda100'),
            ('population_segmatch_q25', 'segmatch_q25 (F1)', '#e87ba4'), ('population_presence3', 'presence3 (F2)', '#6250d6')]
SERIES = [('population_any', 'element, old families (pop.)', '#2a78d6'),
          ('null_any', 'element, old families (null)', '#eb6834'),
          ('population_any_new', 'element, segment families (pop.)', '#1baf7a'),
          ('subject_detection', 'subject, primary min-p', '#eda100'),
          ('subject_detection_ext', 'subject, extended min-p', '#6250d6')]


def value(e, key, baseline):
    if not baseline:
        return e[key]
    return e['baseline_' + key] if key.startswith(('population', 'null')) else e[key.replace('subject_detection', 'baseline_subject_detection')]


def main():
    run = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / 'dose_response.png'
    series = {'radius': SERIES_RADIUS, 'f': SERIES_F}.get(sys.argv[3] if len(sys.argv) > 3 else 'default', SERIES)
    d = json.loads((run / 'deviation.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for i, method in enumerate(('learned', 'centroid')):
        groups = d['sensitivity'][f'{method}/uniform']['by_perturbation']
        any_label = next(iter(groups.values()))
        for j, (title, xlabel, points) in enumerate(PANELS):
            ax = axes[i, j]
            for key, name, colour in series:
                xs = [x for x, _ in points]
                ys = [value(groups[lab] if lab else any_label, key, lab is None) for _, lab in points]
                ax.plot(xs, ys, '-o', color=colour, lw=2, ms=6, label=name)
            ax.axhline(0.05, color='#52514e', lw=0.8, ls=':')
            ax.set_title(f'{method}: {title}', fontsize=10)
            ax.set_xlabel(xlabel); ax.grid(alpha=0.25, lw=0.5)
            if j < 2:
                ax.invert_xaxis()
            if j == 0:
                ax.set_ylabel('detection rate per copy')
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('Detection vs magnitude (83 copies per point; leftmost = unperturbed refit of the same pairs; '
                 'dotted = 0.05). Screening evidence.', fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    main()
