"""Individual deviation maps of the validation subjects (Step E2; CPU; reads a deviation run with --save-elements).

Learned alignment, uniform weighting, primary population. Per validation subject: axial / coronal / sagittal
projections in template space of
  - the population graph, each edge coloured by status: present and expected (p_e >= 0.5), absent but expected
    (colour = defect class radius_threshold / graph_defect / displaced / absent, labelled with p_e and the population
    p = 1 - p_e in the axial view), present though unexpected (p_e < 0.5); absent and unexpected edges faint;
  - the subject's mapped large-vessel and all-radius centrelines on a faint MIP of its mapped vessel mask;
  - population nodes coloured by the radius robust z (radius_absz, symmetric limits), BH discoveries marked
    (population-referenced: ring, null-referenced: plus; edge families at the edge midpoint: square / cross);
  - a side panel: id, age, primary scores with empirical p, family min-p decision, BH discoveries per family,
    absent expected edges with defect class.
Also a 20-subject montage, an absent-but-expected summary per population edge (validation by defect class vs the
null-300 rate), copy-vs-baseline examples for the example subject (one copy per label) and a self-contained index.html.
Usage: python analysis/deviation/plot_deviations.py DEVIATION_DIR OUTPUT_DIR
"""
import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFECTS = ('radius_threshold', 'graph_defect', 'displaced', 'absent')   # codes 0-3 in query_defects
STATUS_COLOUR = {'expected': '#9a9a96', 'unexpected': '#2a78d6', 'radius_threshold': '#eda100', 'graph_defect': '#e34948',
                 'displaced': '#6250d6', 'absent': '#e87ba4'}
LARGE_COLOUR, ALL_COLOUR = '#1baf7a', '#9085e9'
CROP_MARGIN_MM = 15.   # views cropped to the population graph's bounding box plus this margin
Z_LIMIT = 4.
VIEWS = (('axial', (0, 1), 2), ('coronal', (0, 2), 1), ('sagittal', (1, 2), 0))   # name, plotted axes, MIP axis
FAMILIES = ('edge_surprise', 'conn_graph', 'conn_mask', 'radius_absz', 'radius_allr')
NODE_FAMILIES = ('radius_absz', 'radius_allr')


class Dump:
    """Accessors on elements_learned.npz for one population."""

    def __init__(self, path, pop_key):
        self.z = np.load(path, allow_pickle=False)
        self.pop = pop_key
        self.row = {k: i for i, k in enumerate(self.z['query_keys'])}
        self.geo = {k: i for i, k in enumerate(self.z['geometry_keys'])}
        off = self.z[f'{pop_key}/pop_path_offsets']; paths = self.z[f'{pop_key}/pop_paths']
        self.paths = [paths[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        self.nodes = self.z[f'{pop_key}/pop_nodes_mm']; self.edges = self.z[f'{pop_key}/pop_edges']
        affine, shape = self.z['grid_affine'], self.z['grid_shape']
        self.extent = {a: (affine[a, 3], affine[a, 3] + shape[a] * affine[a, a]) for a in range(3)}
        allp = np.vstack(self.paths + [self.nodes])
        self.crop = {a: (max(allp[:, a].min() - CROP_MARGIN_MM, self.extent[a][0]), min(allp[:, a].max() + CROP_MARGIN_MM, self.extent[a][1]))
                     for a in range(3)}

    def in_crop(self, pts):
        return np.all([(pts[:, a] >= self.crop[a][0]) & (pts[:, a] <= self.crop[a][1]) for a in range(3)], axis=0)

    def q(self, name, key):
        return self.z[f'{self.pop}/query_{name}'][self.row[key]]

    def lines(self, key, kind):
        i = self.geo[key]; off = self.z[f'geometry_{kind}_offsets']
        return self.z[f'geometry_{kind}'][off[i]:off[i + 1]]

    def mip(self, key, view):
        return self.z[f'geometry_mip_{view}'][self.geo[key]]

    def status(self, key):
        """Per edge: 'expected', 'unexpected', a defect class, or None (absent and not expected)."""
        pe, present, defects = self.q('edge_p', key), self.q('presence', key).astype(bool), self.q('defects', key)
        out = []
        for k in range(len(pe)):
            if present[k]:
                out.append('expected' if pe[k] >= 0.5 else 'unexpected')
            elif pe[k] >= 0.5:
                out.append(DEFECTS[defects[k]] if defects[k] >= 0 else 'absent')
            else:
                out.append(None)
        return out

    def discoveries(self, key):
        """{family: {'population': ids, 'null': ids}} (edges or nodes)."""
        return {f: {'population': np.flatnonzero(self.q(f'reject_population_{f}', key)).tolist(),
                    'null': np.flatnonzero(self.q(f'reject_{f}', key)).tolist()} for f in FAMILIES}


def draw_projection(ax, d, key, view, highlight=None, annotate=True):
    name, (i, j), axis = view
    mip = d.mip(key, name).astype(float)
    ax.imshow(mip.T, origin='lower', cmap='Greys', alpha=0.13, vmin=0, vmax=1, interpolation='nearest',
              extent=[*d.extent[i], *d.extent[j]])
    status, pe = d.status(key), d.q('edge_p', key)
    for k, path in enumerate(d.paths):
        s = status[k]
        if s is None:
            ax.plot(path[:, i], path[:, j], color='#c3c2b7', lw=0.6, ls=':', zorder=1)
            continue
        lw = 2.4 if s not in ('expected',) else 1.6
        ax.plot(path[:, i], path[:, j], color=STATUS_COLOUR[s], lw=lw, zorder=2, solid_capstyle='round')
        if annotate and s in DEFECTS and name == 'axial':
            mid = path[len(path) // 2]
            ax.annotate(f'e{k} p_e {pe[k]:.2f}\npop p {1 - pe[k]:.2f}', (mid[i], mid[j]), fontsize=5.5, color='#0b0b0b',
                        xytext=(3, 3), textcoords='offset points', zorder=6)
    if highlight is not None:
        for k in highlight:
            ax.plot(d.paths[k][:, i], d.paths[k][:, j], color='#0b0b0b', lw=4.5, alpha=0.25, zorder=1.5)
    large, alll = d.lines(key, 'large'), d.lines(key, 'all')
    large, alll = large[d.in_crop(large)], alll[d.in_crop(alll)]
    ax.scatter(alll[:, i], alll[:, j], s=0.6, color=ALL_COLOUR, alpha=0.45, lw=0, zorder=1.8, rasterized=True)
    ax.scatter(large[:, i], large[:, j], s=1.2, color=LARGE_COLOUR, alpha=0.8, lw=0, zorder=3.5, rasterized=True)
    z = d.q('radius_z', key)
    ok = np.isfinite(z)
    sc = ax.scatter(d.nodes[ok, i], d.nodes[ok, j], c=np.clip(z[ok], -Z_LIMIT, Z_LIMIT), cmap='RdBu_r', vmin=-Z_LIMIT, vmax=Z_LIMIT,
                    s=26, edgecolors='#fcfcfb', linewidths=0.8, zorder=4)
    ax.scatter(d.nodes[~ok, i], d.nodes[~ok, j], s=10, facecolors='none', edgecolors='#9a9a96', linewidths=0.6, zorder=4)
    disc = d.discoveries(key)
    for f in FAMILIES:
        for ref, marker, size in (('population', 'o', 110), ('null', '+', 90)):
            ids = disc[f][ref]
            if not ids:
                continue
            if f in NODE_FAMILIES:
                pts = d.nodes[ids]
                mk = marker if f == 'radius_absz' else ('D' if ref == 'population' else 'x')
            else:
                pts = np.array([d.paths[k][len(d.paths[k]) // 2] for k in ids])
                mk = 's' if ref == 'population' else 'x'
            ax.scatter(pts[:, i], pts[:, j], s=size, marker=mk, facecolors='none' if mk in 'osD' else '#0b0b0b',
                       edgecolors='#0b0b0b', linewidths=1.1, zorder=5)
    ax.set_xlim(*d.crop[i]); ax.set_ylim(*d.crop[j]); ax.set_aspect('equal')
    ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    return sc


def legend_handles():
    h = [Line2D([], [], color=STATUS_COLOUR['expected'], lw=1.6, label='present, expected'),
         Line2D([], [], color=STATUS_COLOUR['unexpected'], lw=2.4, label='present, unexpected (p_e < 0.5)')]
    h += [Line2D([], [], color=STATUS_COLOUR[c], lw=2.4, label=f'absent, expected: {c}') for c in DEFECTS]
    h += [Line2D([], [], color='#c3c2b7', lw=0.8, ls=':', label='absent, not expected'),
          Line2D([], [], marker='.', ls='', color=LARGE_COLOUR, label='subject large-vessel centreline'),
          Line2D([], [], marker='.', ls='', color=ALL_COLOUR, label='subject all-radius centreline'),
          Line2D([], [], marker='o', ls='', mfc='none', mec='#0b0b0b', label='BH radius_absz (population)'),
          Line2D([], [], marker='+', ls='', color='#0b0b0b', label='BH radius_absz (null)'),
          Line2D([], [], marker='D', ls='', mfc='none', mec='#0b0b0b', label='BH radius_allr (pop.; x = null)'),
          Line2D([], [], marker='s', ls='', mfc='none', mec='#0b0b0b', label='BH edge families (pop.; x = null)')]
    return h


def side_text(d, key, report, alpha):
    lines = []
    if report:
        lines += [f"{report['subject_id']}  age {report['age']:.0f}", '',
                  f"family min-p p = {report['p_family_minp']:.3f} -> {'FLAGGED' if report['p_family_minp'] <= alpha else 'not flagged'}",
                  'primary scores (empirical p):']
        lines += [f"  {s[:24]:24s} {report['p'][s]:.3f}" for s in report['primary_scores']]
    disc = d.discoveries(key)
    lines += ['', 'BH discoveries (population | null):']
    for f in FAMILIES:
        unit = 'n' if f in NODE_FAMILIES else 'e'
        pop_ids = ','.join(f'{unit}{k}' for k in disc[f]['population']) or '-'
        null_ids = ','.join(f'{unit}{k}' for k in disc[f]['null']) or '-'
        lines.append(f'  {f:13s} {pop_ids} | {null_ids}')
    status, pe = d.status(key), d.q('edge_p', key)
    absent = [(k, s) for k, s in enumerate(status) if s in DEFECTS]
    lines += ['', f'absent expected edges ({len(absent)}):']
    lines += [f'  e{k:<3d} p_e {pe[k]:.2f}  pop p {1 - pe[k]:.2f}  {s}' for k, s in absent] or ['  none']
    return '\n'.join(lines)


def subject_figure(d, key, report, alpha, path, title=None, highlight=None):
    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.05)
    sc = None
    for c, view in enumerate(VIEWS):
        sc = draw_projection(fig.add_subplot(gs[0, c]), d, key, view, highlight)
    ax = fig.add_subplot(gs[0, 3]); ax.axis('off')
    ax.text(0, 1, side_text(d, key, report, alpha), va='top', ha='left', family='monospace', fontsize=6.6, transform=ax.transAxes)
    cax = fig.add_axes([0.08, 0.08, 0.18, 0.02])
    fig.colorbar(sc, cax=cax, orientation='horizontal').set_label('node radius robust z (radius_absz)', fontsize=7)
    cax.tick_params(labelsize=6)
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle(title or f'{key}  (learned, uniform, {d.pop}; screening evidence)', fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


def montage(paths, out, cols=4):
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2.2))
    for ax in np.ravel(axes):
        ax.axis('off')
    for ax, p in zip(np.ravel(axes), paths):
        ax.imshow(plt.imread(p)); ax.set_title(p.stem.replace('subject_', ''), fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def absent_summary(d, val_keys, null_keys, out):
    """Per population edge: validation fraction absent-but-expected by defect class (stacked) vs the null-300 rate."""
    n_edges = len(d.edges)
    counts = np.zeros((len(DEFECTS), n_edges)); null_rate = np.zeros(n_edges)
    for k in val_keys:
        for e, s in enumerate(d.status(k)):
            if s in DEFECTS:
                counts[DEFECTS.index(s), e] += 1
    for k in null_keys:
        null_rate += np.array([s in DEFECTS for s in d.status(k)], float)
    null_rate /= max(len(null_keys), 1)
    show = np.flatnonzero((counts.sum(0) > 0) | (null_rate > 0))
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(show) + 2), 3.6))
    x = np.arange(len(show)); bottom = np.zeros(len(show))
    for c, name in enumerate(DEFECTS):
        v = counts[c, show] / max(len(val_keys), 1)
        ax.bar(x, v, bottom=bottom, color=STATUS_COLOUR[name], width=0.7, label=f'validation: {name}', edgecolor='#fcfcfb', lw=1)
        bottom += v
    ax.scatter(x, null_rate[show], marker='_', s=260, color='#0b0b0b', lw=2, zorder=3, label='null 300 (all classes)')
    ax.set_xticks(x); ax.set_xticklabels([f'e{e}\n{d.z[f"{d.pop}/pop_edge_probability"][e]:.2f}' for e in show], fontsize=6)
    ax.set_xlabel('population edge (p_e)', fontsize=8); ax.set_ylabel('fraction absent but expected', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    ax.set_title(f'Absent-but-expected edges: {len(val_keys)} validation subjects vs null {len(null_keys)}\n(learned, uniform; screening evidence)', fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def embed(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run / 'deviation.json').read_text())
    s = summary['settings']; alpha = s['alpha']
    pop_key = f"r{s['radii_mm'][0]:g}mm_tau{s['tau']:g}"
    d = Dump(run / 'elements_learned.npz', pop_key)
    val_keys = [k for k in d.z['geometry_keys'] if k.startswith('validation/')]
    null_keys = [k for k, st in zip(d.z['query_keys'], d.z['query_sets']) if st == 'null']
    pages = []
    for key in val_keys:
        sid = key.split('/', 1)[1]
        rp = run / 'subjects' / sid / 'report_learned_uniform.json'
        report = json.loads(rp.read_text()) if rp.is_file() else None
        path = out / f'subject_{sid}.png'
        subject_figure(d, key, report, alpha, path)
        pages.append((sid, path))
    montage([p for _, p in pages], out / 'montage.png')
    absent_summary(d, val_keys, null_keys, out / 'absent_expected_summary.png')
    examples = []
    example = str(d.z['example_subject'])
    if example:
        copies = {c['copy_id']: c for c in summary['sensitivity']['learned/uniform']['copies']}
        base = f'validation/{example}'
        for key in [k for k in d.z['geometry_keys'] if k.startswith('perturbed/')]:
            c = copies[key.split('/', 1)[1]]
            path = out / f"example_{c['label']}.png"
            fig_paths = []
            for k, tag in ((base, 'unperturbed baseline'), (key, f"copy {c['label']} (target edge {c['population_edge']})")):
                fp = out / f"_tmp_{tag.split()[0]}.png"
                subject_figure(d, k, None, alpha, fp, title=f'{example}: {tag}', highlight=c['mapped_edges'])
                fig_paths.append(fp)
            fig, axes = plt.subplots(2, 1, figsize=(15, 10.6))
            for ax, fp in zip(axes, fig_paths):
                ax.imshow(plt.imread(fp)); ax.axis('off'); fp.unlink()
            fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
            examples.append((c['label'], path))
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Deviation maps</title>',
            '<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b}img{max-width:100%;height:auto}'
            'h2{margin-top:2em}</style></head><body>',
            f'<h1>Deviation maps: {len(val_keys)} validation subjects</h1>',
            f'<p>Learned alignment, uniform weighting, population {pop_key}; model step {summary["step"]}; run {run}. '
            'Screening evidence. Node colour = radius robust z (symmetric limits &plusmn;4); rings / plus = BH discoveries '
            '(population- / null-referenced).</p>',
            f'<h2>Absent-but-expected edges</h2><img src="{embed(out / "absent_expected_summary.png")}">',
            f'<h2>Montage</h2><img src="{embed(out / "montage.png")}">']
    html += [f'<h2>{sid}</h2><img src="{embed(p)}">' for sid, p in pages]
    if examples:
        html.append(f'<h2>Perturbation examples ({example}; first copy per label, illustration only)</h2>')
        html += [f'<h3>{lab}</h3><img src="{embed(p)}">' for lab, p in examples]
    html.append('</body></html>')
    (out / 'index.html').write_text('\n'.join(html))
    print(out / 'index.html')


if __name__ == '__main__':
    main()
