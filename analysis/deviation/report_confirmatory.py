"""P1 confirmatory run: regression check and pre-registered report (CPU, reads deviation outputs only).

  regression NEW REF   every field of REF/deviation.json (except settings.output / settings.sets / runtime_s) and every
                       array of REF/elements_<method>.npz must be identical in NEW; prints the number of diffs.
  report RUN --output JSON
                       pre-registered criteria (CONTEXT, Stage P1 confirmatory, Step 1), judged as written:
                         primary   learned / r3mm_tau0.1 / uniform / segmatch_q25_apriori, null-referenced (elements_bh)
                         C1        sensitivity thin0.3 >= 0.25
                         C2        sensitivity stenosis0.3 >= 0.30
                         C3        FPR (subjects with >= 1 BH discovery) <= 0.06 on null_loo AND on validation
                       secondary (reported, never judged): population-referenced FPR and sensitivity, all 8 labels,
                       unperturbed baselines, per-edge table, segmatch_med_apriori, centroid, families, discrete p
                       floor vs q/m, Wilson 95% intervals (descriptive).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

POP = 'r3mm_tau0.1'
PRIMARY = ('learned', 'segmatch_q25_apriori')
SECONDARY = [('learned', 'segmatch_med_apriori'), ('centroid', 'segmatch_q25_apriori'), ('centroid', 'segmatch_med_apriori')]
LABELS = ('thin0', 'thin0.3', 'remove0', 'stenosis0.5', 'stenosis0.3', 'displace2', 'displace3', 'displace5')
STAT_ARRAYS = {'segmatch_q25_apriori': 'segrad_match_q25', 'segmatch_med_apriori': 'segrad_match_med'}
IGNORED = {('settings', 'output'), ('settings', 'sets'), ('runtime_s',)}


def diff(a, b, path=()):
    """Paths where nested JSON values a and b differ (NaN equals NaN)."""
    if path in IGNORED:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b), key=str) for p in
                (diff(a[k], b[k], path + (k,)) if k in a and k in b else [path + (k,)])]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [path]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diff(x, y, path + (i,))]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [path]


def regression(new, ref):
    out = diff(json.loads((Path(ref) / 'deviation.json').read_text()), json.loads((Path(new) / 'deviation.json').read_text()))
    for method in ('learned', 'centroid'):
        a, b = np.load(Path(ref) / f'elements_{method}.npz'), np.load(Path(new) / f'elements_{method}.npz')
        out += [(f'elements_{method}', k) for k in sorted(set(a.files) | set(b.files))
                if k not in a.files or k not in b.files or not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == 'f')]
    return out


def wilson(p, n, z=1.959964):
    if not n:
        return None
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0., centre - half), min(1., centre + half)]


def floors(run, method, fam, q, family):
    """Discrete p floors of the family's elements (population 2 / (n_train + 1), null 1 / (n_null + 1)) vs q / m."""
    z = np.load(Path(run) / f'elements_{method}.npz', allow_pickle=False)
    g = lambda name: z[f'{POP}/{name}']
    stat = STAT_ARRAYS[fam]
    train, query, sets = g(f'train_{stat}'), g(f'query_{stat}'), z['query_sets']
    null = sets == 'null'
    pop_p = g(f'query_pop_p_{fam}')
    m = float(np.median(np.isfinite(pop_p[null]).sum(1)))
    rows = {int(e): {'n_train': int(np.isfinite(train[:, e]).sum()), 'n_null': int(np.isfinite(query[null, e]).sum())} for e in family}
    for r in rows.values():
        r.update(floor_population=2. / (r['n_train'] + 1), floor_null=1. / (r['n_null'] + 1))
    bh1 = q / m if m else None
    return {'m_elements': m, 'bh_rank1_p': bh1, 'elements': rows,
            'floor_binds_population': [e for e, r in rows.items() if bh1 is not None and r['floor_population'] > bh1],
            'floor_binds_null': [e for e, r in rows.items() if bh1 is not None and r['floor_null'] > bh1]}


def rows_for(d, method, fam):
    s = d['sensitivity'][f'{method}/uniform']['by_perturbation']
    f = d['fpr'][f'{method}/uniform']
    fpr = {}
    for g in ('null_loo', 'validation'):
        n = f[g]['subjects']
        for ref, kind in (('null', 'elements_bh'), ('population', 'elements_bh_population')):
            v = f[g][kind][POP][fam]['subjects_with_discovery']
            fpr[f'{g}/{ref}'] = {'fpr': v, 'subjects': n, 'wilson95': wilson(v, n)}
    labels = {}
    for lab in LABELS:
        if lab not in s:
            continue
        e = s[lab]
        n = e['copies']
        labels[lab] = {'copies': n, 'subjects': e['subjects'],
                       **{f'{ref}': e[f'{ref}_{fam}'] for ref in ('null', 'population')},
                       **{f'{ref}_wilson95': wilson(e[f'{ref}_{fam}'], n) for ref in ('null', 'population')},
                       **{f'baseline_{ref}': e[f'baseline_{ref}_{fam}'] for ref in ('null', 'population')},
                       'per_edge': {k: {'copies': v['copies'], 'p_edge': v['p_edge'], 'null': v[f'null_{fam}'],
                                        'population': v[f'population_{fam}']} for k, v in e['per_edge'].items()}}
    return {'fpr': fpr, 'labels': labels}


def report(run):
    d = json.loads((Path(run) / 'deviation.json').read_text())
    q = d['settings']['q']
    method, fam = PRIMARY
    primary = rows_for(d, method, fam)
    sens = primary['labels']
    criteria = {
        'C1_null_thin0.3_ge_0.25': {'value': sens['thin0.3']['null'], 'pass': sens['thin0.3']['null'] >= 0.25},
        'C2_null_stenosis0.3_ge_0.30': {'value': sens['stenosis0.3']['null'], 'pass': sens['stenosis0.3']['null'] >= 0.30},
        'C3_null_fpr_le_0.06': {'value': {g: primary['fpr'][f'{g}/null']['fpr'] for g in ('null_loo', 'validation')},
                                'pass': all(primary['fpr'][f'{g}/null']['fpr'] <= 0.06 for g in ('null_loo', 'validation'))}}
    out = {'run': str(run), 'label': 'CONFIRMATORY RESULT (P1 design frozen; pre-registered criteria)',
           'primary': f'{method}/{POP}/uniform/{fam}, null-referenced', 'criteria': criteria,
           'all_pass': all(c['pass'] for c in criteria.values()),
           'families': d['apriori_families'], 'perturbations': d['perturbations'],
           'rows': {f'{m}/{f}': rows_for(d, m, f) for m, f in [PRIMARY] + SECONDARY},
           'floors': {f'{m}/{f}': floors(run, m, f, q, d['apriori_families'][f'{m}/{POP}'])
                      for m, f in [PRIMARY] + SECONDARY if d['apriori_families'][f'{m}/{POP}']}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('regression'); r.add_argument('new'); r.add_argument('ref')
    s = sub.add_parser('report'); s.add_argument('run'); s.add_argument('--output', required=True)
    a = p.parse_args()
    if a.mode == 'regression':
        out = regression(a.new, a.ref)
        print(json.dumps({'diffs': len(out), 'first': [list(map(str, x)) for x in out[:20]]}, indent=1))
        return
    out = report(a.run)
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({'label': out['label'], 'criteria': out['criteria'], 'all_pass': out['all_pass']}, indent=1))


if __name__ == '__main__':
    main()
