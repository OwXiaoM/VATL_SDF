"""Stage G0.1 / G0.2: theoretical z vs BH threshold and variance decomposition (CPU; reads a --save-elements run).

Statistics: radius_absz (node: large-vessel node radius sdf), radius_allr (node), segmatch_med, segmatch_q25 (edge);
methods learned and centroid. Target elements of a copy = mapped_nodes / mapped_edges (deviation.json).
G0.1  delta = copy value - unperturbed validation value of the same element; sigma = robust SD (IQR / 1.349) of the
      training reference (pop) or of the null-300 queries (null); z_theory = delta / sigma, per copy the maximum |z|
      over its target elements; z_needed = Phi^-1(1 - q / (2 m)), m = finite elements of the family (median over null
      queries); discrete floor 2 / (n_ref + 1); predicted detection = share of copies with max |z| >= z_needed vs the
      observed BH flags on the same elements.
G0.2  sigma_fit = pooled within-subject SD over the validation fit and its replicates (valrep*); sigma_total =
      sigma_null; sigma_between = sqrt(max(sigma_total^2 - sigma_fit^2, 0)); projected detection if sigma_between
      were reduced to f (0.5, 0.25, 0): sigma' = sqrt(sigma_fit^2 + (f sigma_between)^2) (projection, not a statistic).
Usage: python analysis/deviation/detectability.py DEVIATION_DIR --output JSON
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

LABELS = ('thin0', 'thin0.3', 'remove0', 'stenosis0.5', 'stenosis0.3', 'displace2', 'displace3', 'displace5')
# statistic: (query array, train array, population p array, reject family, element kind)
STATISTICS = {'radius_absz': ('query_node_radius', 'train_node_radius', 'query_pop_p_radius', 'radius_absz', 'nodes'),
              'radius_allr': ('query_node_radius_allr', 'train_node_radius_allr', 'query_pop_p_radius_allr', 'radius_allr', 'nodes'),
              'segmatch_med': ('query_segrad_match_med', 'train_segrad_match_med', 'query_pop_p_segmatch_med', 'segmatch_med', 'edges'),
              'segmatch_q25': ('query_segrad_match_q25', 'train_segrad_match_q25', 'query_pop_p_segmatch_q25', 'segmatch_q25', 'edges')}
APRIORI = {'segmatch_q25_apriori': ('query_segrad_match_q25', 'train_segrad_match_q25', 'query_pop_p_segmatch_q25_apriori',
                                    'segmatch_q25_apriori', 'edges'),
           'segmatch_med_apriori': ('query_segrad_match_med', 'train_segrad_match_med', 'query_pop_p_segmatch_med_apriori',
                                    'segmatch_med_apriori', 'edges')}   # Stage P1 (present when the dump has them)
FRACTIONS = (0.5, 0.25, 0.)


def robust_sd(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return float((q3 - q1) / 1.349)


def z_needed(q, m):
    return float(norm.ppf(1. - q / (2. * m)))


def decompose(total, fit):
    return float(np.sqrt(max(total ** 2 - fit ** 2, 0.)))


def pooled_within_sd(values):
    """values (subjects, fits), NaN = missing: sqrt of the mean within-subject variance (subjects with >= 2 fits)."""
    var = [np.nanvar(v, ddof=1) for v in values if np.isfinite(v).sum() >= 2]
    return float(np.sqrt(np.mean(var))) if var else np.nan


def quart(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return None
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return [float(med), float(q3 - q1)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--output', required=True)
    a = p.parse_args()
    run = Path(a.run)
    d = json.loads((run / 'deviation.json').read_text())
    q = d['settings']['q']
    pop_key = f"r{d['settings']['radii_mm'][0]:g}mm_tau{d['settings']['tau']:g}"
    report = {'run': str(run), 'q': q, 'g01': {}, 'g02': {}}
    for method in ('learned', 'centroid'):
        z = np.load(run / f'elements_{method}.npz', allow_pickle=False)
        g = lambda name: z[f'{pop_key}/{name}']
        keys = list(z['query_keys']); sets = z['query_sets']; row = {k: i for i, k in enumerate(keys)}
        null = np.flatnonzero(sets == 'null')
        copies = d['sensitivity'][f'{method}/uniform']['copies']
        report['g01'][method], report['g02'][method] = {}, {}
        stats = dict(STATISTICS, **{k: v for k, v in APRIORI.items() if f'{pop_key}/{v[2]}' in z})
        for stat, (qa, ta, pa, fam, kind) in stats.items():
            Q = g(qa)[..., 0] if qa == 'query_node_radius' else g(qa)
            T = g(ta)[..., 0] if ta == 'train_node_radius' else g(ta)
            if stat == 'radius_absz' or stat in APRIORI:   # tested elements only (finite population p), as the family
                Q = np.where(np.isfinite(g(pa)), Q, np.nan)
            sig_pop = np.array([robust_sd(T[:, e]) for e in range(T.shape[1])])
            sig_null = np.array([robust_sd(Q[null, e]) for e in range(Q.shape[1])])
            n_pop = np.isfinite(T).sum(0); n_null = np.isfinite(Q[null]).sum(0)
            m = float(np.median(np.isfinite(g(pa)[null]).sum(1)))
            zn = z_needed(q, m)
            rej = {'population': g(f'query_reject_population_{fam}'), 'null': g(f'query_reject_{fam}')}
            # G0.2 fit noise: validation + replicate fits per subject
            subjects = sorted({s for s, st in zip(z['query_subjects'], sets) if st == 'validation'})
            fits = {s: [row[f'validation/{s}']] + [row[k] for k in keys if k.startswith('valrep') and k.split('/', 1)[1] == s] for s in subjects}
            sig_fit = np.array([pooled_within_sd(np.array([Q[idx, e] for idx in fits.values()])) for e in range(Q.shape[1])])
            per_label = {}
            target_elements = set()
            for lab in LABELS:
                sel = [c for c in copies if c['label'] == lab]
                if not sel:
                    continue
                zmax = {'population': [], 'null': []}; proj = {f: [] for f in FRACTIONS}; obs = {'population': [], 'null': []}
                deltas = {}
                for c in sel:
                    i, b = row[f"perturbed/{c['copy_id']}"], row[f"validation/{c['subject_id']}"]
                    el = c['mapped_nodes'] if kind == 'nodes' else c['mapped_edges']
                    target_elements.update(el)
                    dl = Q[i][el] - Q[b][el]
                    deltas.setdefault(c['population_edge'], []).extend(dl[np.isfinite(dl)].tolist())
                    for ref, sg in (('population', sig_pop), ('null', sig_null)):
                        zz = np.abs(dl / sg[el])
                        zmax[ref].append(float(np.nanmax(zz)) if np.isfinite(zz).any() else 0.)
                        obs[ref].append(bool(rej[ref][i][el].any()) if el else False)
                    for f in FRACTIONS:
                        sp = np.sqrt(sig_fit[el] ** 2 + (f * np.array([decompose(t, s) for t, s in zip(sig_null[el], sig_fit[el])])) ** 2)
                        zz = np.abs(dl / np.where(sp > 0, sp, np.nan))
                        proj[f].append(float(np.nanmax(zz)) if np.isfinite(zz).any() else 0.)
                per_label[lab] = {
                    'copies': len(sel),
                    **{f'predicted_{ref}': float(np.mean(np.array(zmax[ref]) >= zn)) for ref in zmax},
                    **{f'observed_{ref}': float(np.mean(obs[ref])) for ref in obs},
                    **{f'median_max_abs_z_{ref}': float(np.median(zmax[ref])) for ref in zmax},
                    'delta_by_edge': {int(e): quart(v) for e, v in deltas.items()},
                    'delta_all': quart([x for v in deltas.values() for x in v]),
                    'projection_sigma_between_fraction': {str(f): float(np.mean(np.array(proj[f]) >= zn)) for f in FRACTIONS}}
            agree = {ref: sum(abs(v[f'predicted_{ref}'] - v[f'observed_{ref}']) <= 0.05 for v in per_label.values()) for ref in ('population', 'null')}
            els = sorted(target_elements)
            report['g01'][method][stat] = {
                'm_elements': m, 'z_needed': zn, 'bh_rank1_p': q / m, 'labels': per_label,
                'labels_with_agreement_le_0.05': agree, 'usable_for_ranking': all(v >= 6 for v in agree.values()),
                'elements': {int(e): {'sigma_pop': float(sig_pop[e]), 'sigma_null': float(sig_null[e]), 'n_pop': int(n_pop[e]),
                                      'n_null': int(n_null[e]), 'floor_pop': 2. / (n_pop[e] + 1), 'floor_binds_pop': bool(2. / (n_pop[e] + 1) > q / m),
                                      'reference_median_pop': float(np.nanmedian(T[:, e])) if n_pop[e] else None} for e in els}}
            report['g02'][method][stat] = {'elements': {int(e): {'sigma_total': float(sig_null[e]), 'sigma_fit': float(sig_fit[e]),
                                                                 'sigma_between': decompose(sig_null[e], sig_fit[e]) if np.isfinite(sig_fit[e]) else None,
                                                                 'fit_share_of_variance': float(sig_fit[e] ** 2 / sig_null[e] ** 2) if sig_null[e] > 0 else None}
                                                   for e in els}}
    Path(a.output).write_text(json.dumps(report, indent=1))
    print(a.output)


if __name__ == '__main__':
    main()
