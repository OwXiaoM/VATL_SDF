"""Stage G0.4: bottleneck decision table from the G0.1/G0.2 report and the G0.3 phantom (CPU, reads JSON only).

Rules (CONTEXT Stage G0, Step 0; learned, population reference): target radius r* = median manifest segment radius of the
label's copies (phantom row at the nearest 0.2 mm radius, straight and curved averaged); s = phantom slope, SD = phantom
measurement SD, delta_phantom = phantom measured change at r*;
  dilution            |median delta| < 0.5 |delta_phantom| and s >= 0.7
  resolution limit    s < 0.5 or SD >= 0.5 sigma_pop
  between-subject     |median delta| >= 0.5 |delta_phantom|, sigma_between >= 0.8 sigma_total, median |z| < z_needed
  unclear             none of these (removal and displacement: not modelled by the phantom)
A phantom rule that yields no value for the perturbed tube (radius_absz: thinned edges leave the large-vessel
selection) counts as dilution by the measurement rule. The p-floor column (G0.1) is reported beside the class.
Usage: python analysis/deviation/decision_table.py G0_DETECTABILITY_JSON G0_PHANTOM_JSON DEVIATION_DIR OUTPUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

RULE = {'radius_absz': 'node_absz', 'radius_allr': 'node_allr', 'segmatch_med': 'segmatch_med', 'segmatch_q25': 'segmatch_q25'}
PHANTOM_LABEL = {'thin0': 'thin0.5', 'thin0.3': 'thin0.3', 'stenosis0.5': 'stenosis0.5', 'stenosis0.3': 'stenosis0.3'}


def phantom_at(phantom, rule, radius, cond):
    key = f'{np.clip(np.round(radius / 0.2) * 0.2, 0.8, 3.0):.1f}'
    rows = [phantom['radii'][f'{shape}/r{key}'][rule] for shape in ('straight', 'curved')]
    cond_rows = [r[cond] for r in rows]
    if any(c['slope'] is None for c in cond_rows):
        return None
    return {'radius_row_mm': float(key), 'slope': float(np.mean([c['slope'] for c in cond_rows])),
            'delta_phantom': float(np.mean([c['measured_change_median'] for c in cond_rows])),
            'sd': float(np.mean([r['sd'] for r in rows if r['sd'] is not None])), 'bias': float(np.mean([r['bias'] for r in rows]))}


def classify(delta, ph, sigma_pop, sigma_between, sigma_total, z_med, z_need):
    if ph is None:
        return 'dilution (measurement rule gives no value for the perturbed vessel)'
    if abs(delta) < 0.5 * abs(ph['delta_phantom']) and ph['slope'] >= 0.7:
        return 'dilution'
    if ph['slope'] < 0.5 or ph['sd'] >= 0.5 * sigma_pop:
        return 'resolution limit'
    if abs(delta) >= 0.5 * abs(ph['delta_phantom']) and sigma_between >= 0.8 * sigma_total and z_med < z_need:
        return 'between-subject variance'
    return 'unclear'


def main():
    g0, phantom = json.loads(Path(sys.argv[1]).read_text()), json.loads(Path(sys.argv[2]).read_text())
    d = json.loads((Path(sys.argv[3]) / 'deviation.json').read_text())
    copies = d['sensitivity']['learned/uniform']['copies']
    table = {}
    for stat, rule in RULE.items():
        s1 = g0['g01']['learned'][stat]; s2 = g0['g02']['learned'][stat]['elements']
        sig_pop = float(np.median([e['sigma_pop'] for e in s1['elements'].values()]))
        sig_tot = float(np.median([e['sigma_total'] for e in s2.values()]))
        sig_bet = float(np.median([e['sigma_between'] for e in s2.values() if e['sigma_between'] is not None]))
        floor_binds = sum(e['floor_binds_pop'] for e in s1['elements'].values()), len(s1['elements'])
        table[stat] = {'sigma_pop_median': sig_pop, 'sigma_total_median': sig_tot, 'sigma_between_median': sig_bet,
                       'z_needed': s1['z_needed'], 'p_floor_binds_elements': list(floor_binds), 'labels': {}}
        for lab, e in s1['labels'].items():
            delta = e['delta_all'][0] if e['delta_all'] else 0.
            r_star = float(np.median([c['segment_radius_mm'] for c in copies if c['label'] == lab]))
            ph = phantom_at(phantom, rule, r_star, PHANTOM_LABEL[lab]) if lab in PHANTOM_LABEL else None
            cls = classify(delta, ph, sig_pop, sig_bet, sig_tot, e['median_max_abs_z_population'], s1['z_needed']) \
                if lab in PHANTOM_LABEL else 'unclear (not modelled by the phantom)'
            table[stat]['labels'][lab] = {'target_radius_mm': r_star, 'median_delta': delta, 'phantom': ph,
                                          'delta_ratio_to_phantom': (delta / ph['delta_phantom']) if ph and ph['delta_phantom'] else None,
                                          'median_abs_z': e['median_max_abs_z_population'], 'observed': e['observed_population'],
                                          'class': cls}
    Path(sys.argv[4]).write_text(json.dumps(table, indent=1))
    for stat, v in table.items():
        print(stat, {lab: (x['class'], None if x['delta_ratio_to_phantom'] is None else round(x['delta_ratio_to_phantom'], 2))
                     for lab, x in v['labels'].items()})


if __name__ == '__main__':
    main()
