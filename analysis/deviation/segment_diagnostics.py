"""Step D0 diagnostics of the segment families and the node-radius test (CPU; reads a deviation run with --save-elements).

Inputs: DEVIATION_DIR/deviation.json (copies with mapped edges / nodes) and elements_<method>.npz (uniform weighting).
Report (DEVIATION_DIR/d0_report.json):
  floor     per target edge and method: fraction of training subjects (trained transforms) with seg_radius <= 0.1 mm
            and coverage <= 0.05 (and exactly 0), vs 1 - training presence; smallest achievable population p of a
            query with value 0 (seg_radius two-sided radius tail, coverage lower tail; uniform weights, all training
            subjects); BH rank-1 threshold q / m (m = finite elements of the family, median over null queries);
  offset    per target edge and over all edges: training (trained transforms) vs null-query (refitted) distributions
            (median, IQR), paired shift on the null subjects (refitted - trained, same subject), and the fraction of
            null queries with population p <= alpha;
  nodes     per copy (thin / stenosis / removal) and its unperturbed validation baseline: mapped target nodes tested
            by radius_absz (finite population p) vs floor-excluded (graph radius <= 0.801 mm) vs absent (no
            large-vessel centreline within r) vs untested (finite radius, < 3 population subjects);
  nodes_allr the same for radius_allr (Step E1; absent = no all-radius graph point within r; no floor rule).
Usage: python analysis/deviation/segment_diagnostics.py DEVIATION_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

from vatl_sdf.deviation import radius_deviation, tail_p

SEG_TOL_MM = 0.1      # seg_radius "about 0": below one tenth of a voxel
COVER_TOL = 0.05      # coverage "about 0": under 5% of the path samples
NODE_LABELS = ('thin0', 'thin0.3', 'stenosis0.5', 'stenosis0.3', 'remove0')


def zero_fraction(col, tol):
    col = col[np.isfinite(col)]
    return float((col <= tol).mean())


def min_population_p(col, family):
    """Population p of a query value 0 (the smallest possible value) against col, uniform weights."""
    dist = col[np.isfinite(col)][:, None]
    w = np.ones(len(dist))
    if family == 'seg_radius':
        return float(radius_deviation(np.array([0.]), dist, w)[0, 3])
    return float(tail_p(np.array([0.]), dist, w, 'lower')[0])


def bh_rank1(q, p):
    """BH threshold for a single discovery among the finite elements of one family."""
    return q / int(np.isfinite(p).sum())


def node_categories(node_radius, floor, tested):
    out = np.full(len(node_radius), 'untested', dtype=object)
    out[~np.isfinite(node_radius[:, 0])] = 'absent'
    out[floor.astype(bool)] = 'floor'
    out[tested.astype(bool)] = 'tested'
    return out


def quartiles(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return None, None
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return float(med), float(q3 - q1)


def distribution_summary(train, null, paired=None):
    tm, ti = quartiles(train); nm, ni = quartiles(null)
    out = {'train_median': tm, 'train_iqr': ti, 'null_median': nm, 'null_iqr': ni}
    if paired is not None:
        out['paired_shift_median'], out['paired_shift_iqr'] = quartiles(paired)
    return out


def node_report(copies, row_of, radius, floor, pop_p):
    """Per label: mapped target node categories (copy vs baseline), transitions, copies losing a tested node and the
    radius change on nodes tested in both. radius (queries, nodes), floor (queries, nodes), pop_p (queries, nodes)."""
    nodes = {}
    for label in NODE_LABELS:
        counts = {'copy': {}, 'baseline': {}}; transitions = {}; lost = 0; dr = []
        sel = [c for c in copies if c['label'] == label]
        for c in sel:
            i, b = row_of[f"perturbed/{c['copy_id']}"], row_of[f"validation/{c['subject_id']}"]
            nn = c['mapped_nodes']
            cat = {}
            for name, j in (('copy', i), ('baseline', b)):
                cat[name] = node_categories(radius[j][nn][:, None], floor[j][nn], np.isfinite(pop_p[j][nn]))
                for x in cat[name]:
                    counts[name][x] = counts[name].get(x, 0) + 1
            for x, y in zip(cat['baseline'], cat['copy']):
                transitions[f'{x}->{y}'] = transitions.get(f'{x}->{y}', 0) + 1
            lost += bool(((cat['baseline'] == 'tested') & (cat['copy'] != 'tested')).any())
            both = (cat['baseline'] == 'tested') & (cat['copy'] == 'tested')
            dr += list(radius[i][nn][both] - radius[b][nn][both])
        nodes[label] = {'copies': len(sel), 'node_counts': counts, 'transitions': transitions,
                        'copies_losing_a_tested_node': lost,
                        'radius_change_tested_both_mm': quartiles(np.array(dr)) if dr else None}
    return nodes


def main():
    run = Path(sys.argv[1])
    d = json.loads((run / 'deviation.json').read_text())
    alpha, q = d['settings']['alpha'], d['settings']['q']
    pop_key = f"r{d['settings']['radii_mm'][0]:g}mm_tau{d['settings']['tau']:g}"
    targets = {t['population_edge']: t['p_edge'] for t in d['perturbations']['targets']}
    report = {'tolerances': {'seg_radius_mm': SEG_TOL_MM, 'coverage': COVER_TOL}, 'population': pop_key, 'alpha': alpha, 'q': q,
              'floor': {}, 'offset': {}, 'nodes': {}, 'nodes_allr': {}}
    for method in ('learned', 'centroid'):
        z = np.load(run / f'elements_{method}.npz', allow_pickle=False)
        keys = list(z['query_keys']); sets = z['query_sets']; subjects = z['query_subjects']
        train_ids = list(z['train_ids']); row_of = {k: i for i, k in enumerate(keys)}
        null = np.flatnonzero(sets == 'null')
        g = lambda name: z[f'{pop_key}/{name}']
        copies = d['sensitivity'][f'{method}/uniform']['copies']
        mapped = {}
        for c in copies:
            mapped.setdefault(c['population_edge'], c['mapped_edges'])
        m_seg = float(np.median([np.isfinite(g('query_pop_p_seg_radius')[i]).sum() for i in null]))
        m_cov = float(np.median([np.isfinite(g('query_pop_p_coverage')[i]).sum() for i in null]))
        floor = {'bh_rank1_threshold': {'seg_radius': q / m_seg, 'coverage': q / m_cov, 'elements_seg_radius': m_seg, 'elements_coverage': m_cov},
                 'targets': {}}
        for edge, p_edge in targets.items():
            rows = []
            for k in mapped.get(edge, []):
                seg, cov = g('train_seg_radius')[:, k], g('train_coverage')[:, k]
                cop = [c for c in copies if c['population_edge'] == edge]
                obs_seg = [g('query_pop_p_seg_radius')[row_of[f"perturbed/{c['copy_id']}"], k] for c in cop if c['label'] in ('thin0.3', 'remove0')]
                rows.append({'population_edge_method': int(k), 'one_minus_presence': float(1 - g('train_presence')[:, k].mean()),
                             'seg_radius_le_tol': zero_fraction(seg, SEG_TOL_MM), 'seg_radius_eq0': zero_fraction(seg, 0.),
                             'coverage_le_tol': zero_fraction(cov, COVER_TOL), 'coverage_eq0': zero_fraction(cov, 0.),
                             'seg_radius_median_present': float(np.median(seg[g('train_presence')[:, k].astype(bool)]))
                             if g('train_presence')[:, k].any() else None,
                             'min_population_p_seg_radius': min_population_p(seg, 'seg_radius'),
                             'min_population_p_coverage': min_population_p(cov, 'coverage'),
                             'observed_min_p_seg_radius_thin0.3_remove0': float(np.nanmin(obs_seg)) if obs_seg else None})
            floor['targets'][int(edge)] = {'p_edge_manifest': p_edge, 'mapped': rows}
        report['floor'][method] = floor
        # offset: training (trained transforms) vs null queries (refitted), paired on the same subjects
        train_row = {s: i for i, s in enumerate(train_ids)}
        paired_rows = [train_row[s] for s in subjects[null]]
        off_tr, off_null = g('train_offset'), g('query_offset')[null]
        shift = off_null - off_tr[paired_rows]
        p_off = g('query_pop_p_offset')[null]
        per_edge = {}
        for edge in targets:
            for k in mapped.get(edge, []):
                per_edge[f'{edge}->{k}'] = {**distribution_summary(off_tr[:, k], off_null[:, k], shift[:, k]),
                                            'null_fraction_pop_p_le_alpha': float(np.mean(p_off[:, k] <= alpha))}
        med_tr, med_null = np.nanmedian(off_tr, 0), np.nanmedian(off_null, 0)
        report['offset'][method] = {
            'targets': per_edge,
            'all_edges': {'edges': int(off_tr.shape[1]), 'fraction_edges_null_median_gt_train': float(np.mean(med_null > med_tr)),
                          'median_of_edge_median_shift': float(np.nanmedian(np.nanmedian(shift, 0))),
                          'median_of_edge_paired_shift_iqr': float(np.nanmedian([quartiles(shift[:, k])[1] or 0. for k in range(shift.shape[1])])),
                          'null_element_fraction_pop_p_le_alpha': float(np.nanmean(p_off <= alpha)),
                          'train_median_of_edge_median': float(np.nanmedian(med_tr)), 'null_median_of_edge_median': float(np.nanmedian(med_null))}}
        # node-radius tests on the mapped target nodes: copy vs unperturbed baseline (same subject)
        report['nodes'][method] = node_report(copies, row_of, g('query_node_radius')[:, :, 0], g('query_floor'), g('query_pop_p_radius'))
        if f'{pop_key}/query_node_radius_allr' in z:
            report['nodes_allr'][method] = node_report(copies, row_of, g('query_node_radius_allr'),
                                                       np.zeros_like(g('query_floor')), g('query_pop_p_radius_allr'))
    (run / 'd0_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
