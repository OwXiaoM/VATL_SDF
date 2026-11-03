"""Deviation-statistics stage (CPU): subject sets, refitting, per-subject decodings."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from analysis.deviation.subject_sets import make_sets
from tests import test_generalization  # module import: keeps FittingTests out of this module's test discovery


class SubjectSetTests(unittest.TestCase):
    def test_disjoint_seeded_and_excluding(self):
        train = [f't{i}' for i in range(30)]
        a = make_sets(train, ['v0', 'v1'], {'null': 10, 'calibration': 5, 'ants_template': 8}, 0, ['t0', 't1'])
        b = make_sets(train, ['v0', 'v1'], {'null': 10, 'calibration': 5, 'ants_template': 8}, 0, ['t0', 't1'])
        self.assertEqual(a, b)
        groups = [set(a[k]) for k in ('null', 'calibration', 'ants_template')]
        self.assertEqual(sum(map(len, groups)), len(set().union(*groups)))
        self.assertFalse({'t0', 't1'} & set().union(*groups))
        with self.assertRaises(ValueError):
            make_sets(train, ['t3'], {'null': 2}, 0, [])
        with self.assertRaises(ValueError):
            make_sets(train, [], {'null': 40}, 0, [])


class RefitAndDecodeTests(unittest.TestCase):
    setUpClass = classmethod(test_generalization.FittingTests.setUpClass.__func__)   # tiny trained checkpoint (s0 train, s1 validation)
    tearDownClass = classmethod(test_generalization.FittingTests.tearDownClass.__func__)

    def test_refit_subjects_and_per_subject_decodings(self):
        from analysis.deviation import refit_subjects
        from vatl_sdf import probabilistic_atlas
        sets = Path(self.temp.name) / 'sets.json'
        sets.write_text(json.dumps({'null': ['s0'], 'calibration': [], 'validation': ['s1']}))
        out = Path(self.temp.name) / 'refit'
        for name in ('null', 'validation'):
            with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', name,
                                                 '--output', str(out), '--align-epochs', '1', '--total-epochs', '2',
                                                 '--steps-per-epoch', '2', '--device', 'cpu']):
                refit_subjects.main()
        fitted = torch.load(out / 'fitted_validation_00.pt', weights_only=False)
        self.assertEqual(fitted['subject_ids'], ['s1'])
        self.assertEqual(fitted['A'].shape, (1, 3, 3))
        self.assertEqual(torch.load(out / 'fitted_null_00.pt', weights_only=False)['subject_ids'], ['s0'])
        with self.assertRaises(FileExistsError):
            with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', 'null',
                                                 '--output', str(out), '--total-epochs', '2', '--device', 'cpu']):
                refit_subjects.main()
        # replicate refits of the validation subjects: own seed, own file name, never picked up as 'validation'
        from vatl_sdf.deviation import load_fitted
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', 'valrep1',
                                             '--output', str(out), '--align-epochs', '1', '--total-epochs', '2',
                                             '--steps-per-epoch', '2', '--device', 'cpu']):
            refit_subjects.main()
        rep = torch.load(out / 'fitted_valrep1_00.pt', weights_only=False)
        self.assertEqual(rep['subject_ids'], ['s1'])
        self.assertFalse(np.allclose(rep['t_mm'], fitted['t_mm']) and np.allclose(rep['A'], fitted['A']))  # other sampler seed
        np.testing.assert_array_equal(load_fitted(out, 'validation')['s1'][1], np.asarray(fitted['t_mm'][0], float))
        self.assertEqual(set(load_fitted(out, 'valrep1')), {'s1'})
        self.assertEqual(len(list(out.glob('fitted_validation_*.pt'))), 1)
        shards = Path(self.temp.name) / 'decode'
        for k in range(2):
            with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(shards),
                                                 '--shard', str(k), '--num-shards', '2', '--device', 'cpu', '--chunk-size', '64',
                                                 '--save-subjects']):
                probabilistic_atlas.main()
        total = sum(np.load(f)['hard'] for f in sorted(shards.glob('shard_*.npz')))
        packed = [np.load(f) for f in sorted(shards.glob('subjects_*.npz'))]
        n = int(np.prod(packed[0]['shape']))
        decoded = sum(np.unpackbits(p['packed'], axis=1, count=n).sum(0) for p in packed if len(p['packed']))
        np.testing.assert_array_equal(decoded, total)
        self.assertEqual([s for p in packed for s in p['subject_ids']], ['s0'])


if __name__ == '__main__':
    unittest.main()


def cohort_config(directory, n_train=6, n_val=2):
    """tiny_config with n_train + n_val subjects whose vessel (sphere + 3-node graph) is shifted per subject."""
    import pickle
    import networkx as nx
    import nibabel as nib
    import pandas as pd
    import yaml
    from tests.test_pipeline import AFFINE
    from vatl_sdf.graph_coords import voxel_to_world
    c = test_generalization.tiny_config(directory)
    rows = []
    world = voxel_to_world(AFFINE, np.indices((8, 8, 8)).reshape(3, -1).T)
    for i in range(n_train + n_val):
        shift = np.array([0.3 * (i % 3), 0.2 * (i % 2), 0.])
        centre = voxel_to_world(AFFINE, np.array([[4., 4., 4.]]))[0] + shift
        sdf = (np.linalg.norm(world - centre, axis=1) - 1.5 - 0.1 * (i % 2)).reshape(8, 8, 8).astype(np.float32)
        nib.save(nib.Nifti1Image(sdf, AFFINE), Path(directory) / f'SDF{i}.nii.gz')
        graph = nx.Graph()
        for n, v in enumerate(([4, 4, 4], [5, 4, 4], [4, 5, 4], [4, 4, 5])):
            graph.add_node(n, pos=voxel_to_world(AFFINE, np.array(v, float)) + shift, radius=np.float32(1.5 if n < 3 or i % 2 else 0.8))
        graph.add_edges_from([(0, 1), (0, 2), (0, 3)], weight=1.)
        with open(Path(directory) / f'Graph{i}.pkl', 'wb') as f:
            pickle.dump({'graph': graph}, f)
        rows.append({'subject_id': f's{i}', 'scan_age': 20 + 5 * i, 'sex': 'MF'[i % 2], 'MRA': 'unused_never_read.nii.gz',
                     'SDF': str(Path(directory) / f'SDF{i}.nii.gz'), 'Graph': str(Path(directory) / f'Graph{i}.pkl')})
    pd.DataFrame(rows).to_csv(c['data']['tsv'], sep='\t', index=False)
    split = Path(directory) / 'splits_cohort.yaml'
    split.write_text(yaml.safe_dump({'train': [f's{i}' for i in range(n_train)], 'val': [f's{i}' for i in range(n_train, n_train + n_val)], 'test': []}))
    c['data'].update(subject_ids=str(split), split_key=None, expected_split_counts={'train': n_train, 'val': n_val, 'test': 0})
    return c


def tube_cohort_config(directory, n_train=6, n_val=2, size=24):
    """Cohort of tubes along x (radius 1.5 mm) with a vertical side branch, shifted per subject; bbox 20 mm."""
    import pickle
    import networkx as nx
    import nibabel as nib
    import pandas as pd
    import yaml
    from vatl_sdf.graph_coords import voxel_to_world
    c = test_generalization.tiny_config(directory)
    c['data']['world_bbox_mm'] = [20, 20, 20]
    affine = np.diag([1., 1., 1., 1.]); affine[:3, 3] = -size / 2
    world = voxel_to_world(affine, np.indices((size,) * 3).reshape(3, -1).T)
    rows = []
    for i in range(n_train + n_val):
        dy, dz = 0.4 * (i % 3) - 0.4, 0.3 * (i % 2)
        line = np.linalg.norm(world[:, 1:] - [dy, dz], axis=1) - 1.5
        branch = np.where(world[:, 2] >= dz, np.linalg.norm(world[:, :2] - [0., dy], axis=1) - 1.5, 9.)
        sdf = np.minimum(line, branch).reshape((size,) * 3).astype(np.float32)
        nib.save(nib.Nifti1Image(sdf, affine), Path(directory) / f'SDF{i}.nii.gz')
        g = nx.Graph()
        xs = np.arange(-8., 9.)
        for n, x in enumerate(xs):
            g.add_node(n, pos=np.array([x, dy, dz]), radius=np.float32(1.5))
        for m, z in enumerate(np.arange(1., 9.)):
            g.add_node(len(xs) + m, pos=np.array([0., dy, dz + z]), radius=np.float32(1.5))
        g.add_edges_from([(n, n + 1) for n in range(len(xs) - 1)] + [(8, len(xs))] + [(len(xs) + m, len(xs) + m + 1) for m in range(7)])
        with open(Path(directory) / f'Graph{i}.pkl', 'wb') as f:
            pickle.dump({'graph': g}, f)
        rows.append({'subject_id': f's{i}', 'scan_age': 20 + 5 * i, 'sex': 'MF'[i % 2], 'MRA': 'unused_never_read.nii.gz',
                     'SDF': str(Path(directory) / f'SDF{i}.nii.gz'), 'Graph': str(Path(directory) / f'Graph{i}.pkl')})
    pd.DataFrame(rows).to_csv(c['data']['tsv'], sep='\t', index=False)
    split = Path(directory) / 'splits_cohort.yaml'
    split.write_text(yaml.safe_dump({'train': [f's{i}' for i in range(n_train)], 'val': [f's{i}' for i in range(n_train, n_train + n_val)], 'test': []}))
    c['data'].update(subject_ids=str(split), split_key=None, expected_split_counts={'train': n_train, 'val': n_val, 'test': 0})
    return c


class DeviationPipelineTests(unittest.TestCase):
    cohort = staticmethod(cohort_config)

    @classmethod
    def setUpClass(cls):
        from tests.test_alignment import normalized, staged
        from vatl_sdf.train import train
        cls.temp = tempfile.TemporaryDirectory()
        c = normalized(staged(cls.cohort(cls.temp.name), epochs=1), 'affine')
        c['graph']['centerline_sdf']['enabled'] = True
        c['graph_fields']['centerline_distance']['enabled'] = True
        c['training'].update(max_steps=None, epochs=2, stop_after_epoch=None)
        c['optimizer']['scheduler_epochs'] = 'run'
        torch.manual_seed(0)
        cls.checkpoint_path = sorted(Path(train(c)).glob('checkpoint_step_*.pt'))[-1]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_end_to_end(self):
        from analysis.deviation import refit_subjects
        from vatl_sdf import deviation, probabilistic_atlas
        d = Path(self.temp.name)
        sets = d / 'sets.json'
        sets.write_text(json.dumps({'null': ['s0', 's1', 's2'], 'calibration': ['s3', 's4'], 'ants_template': ['s5'], 'validation': ['s6', 's7']}))
        for name in ('null', 'calibration', 'validation'):
            with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', name,
                                                 '--output', str(d / 'refit'), '--align-epochs', '1', '--total-epochs', '2',
                                                 '--steps-per-epoch', '1', '--device', 'cpu']):
                refit_subjects.main()
        with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'dec'),
                                             '--shard', '0', '--num-shards', '1', '--device', 'cpu', '--chunk-size', '64', '--save-subjects']):
            probabilistic_atlas.main()
        out = d / 'deviation'
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--decodings', str(d / 'dec'),
                                             '--refit', str(d / 'refit'), '--output', str(out), '--min-skeleton-voxels', '1',
                                             '--age-sigmas', '7', '--smooth-mm', '0', '1', '--recalibration', 'none', 'isotonic',
                                             '--workers', '1']):
            deviation.main()
        summary = json.loads((out / 'deviation.json').read_text())
        self.assertEqual(set(summary['fpr']), {f'{m}/{w}' for m in ('decoder', 'learned', 'centroid') for w in ('uniform', 'age_sigma7')})
        fpr = summary['fpr']['learned/uniform']
        self.assertEqual((fpr['null_loo']['subjects'], fpr['validation']['subjects']), (3, 2))
        self.assertIn('affine_d2', fpr['validation']['per_score'])
        self.assertIn('primary_family_minp', fpr['validation']); self.assertIn('elements_bh_population', fpr['validation'])
        self.assertNotIn('affine_d2', summary['fpr']['centroid/uniform']['validation']['per_score'])
        scores = pd.read_csv(out / 'scores.tsv', sep='\t', keep_default_na=False, na_values=[''])  # 'null' is a set name
        p = scores.p.dropna()
        self.assertTrue(((p > 0) & (p <= 1)).all())
        self.assertEqual(set(scores.set), {'null', 'validation'})       # calibration subjects never scored
        self.assertTrue((out / 'subjects' / 's6' / 'VoxelSurprise_learned.nii.gz').is_file())
        self.assertFalse((out / 'subjects' / 's0' / 'VoxelSurprise_learned.nii.gz').exists())  # maps: validation only
        report = json.loads((out / 'subjects' / 's6' / 'report_learned_uniform.json').read_text())
        self.assertIn('r3mm_tau0.1', report['populations'])


    def test_refit_validation_by_explicit_ids(self):
        """Confirmatory cohort: the set's validation list (not the first N YAML val IDs) is refitted, in list order."""
        from analysis.deviation import refit_subjects
        d = Path(self.temp.name) / 'explicit'
        d.mkdir()
        for ids, expect in ((['s7'], ['s7']), (['s7', 's6'], ['s7', 's6']), (['s0'], ValueError), (['s99'], ValueError)):
            sets = d / 'sets.json'
            sets.write_text(json.dumps({'null': ['s0'], 'calibration': [], 'validation': ids}))
            out = d / f'refit_{"_".join(ids)}'
            argv = ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', 'validation', '--output', str(out),
                    '--align-epochs', '1', '--total-epochs', '1', '--steps-per-epoch', '1', '--device', 'cpu']
            with mock.patch.object(sys, 'argv', argv):
                if expect is ValueError:
                    with self.assertRaises(ValueError):
                        refit_subjects.main()
                    continue
                refit_subjects.main()
            self.assertEqual(torch.load(out / 'fitted_validation_00.pt', weights_only=False)['subject_ids'], expect)


class PerturbationPipelineTests(DeviationPipelineTests):
    cohort = staticmethod(tube_cohort_config)

    def test_end_to_end(self):  # covered by DeviationPipelineTests
        pass

    def test_refit_validation_by_explicit_ids(self):  # covered by DeviationPipelineTests
        pass

    def test_perturbation_pipeline(self):
        from analysis.deviation import perturbation_manifest, refit_subjects
        from vatl_sdf import deviation, graph_atlas, probabilistic_atlas
        d = Path(self.temp.name) / 'pert'
        d.mkdir()
        sets = d / 'sets.json'
        sets.write_text(json.dumps({'null': ['s0', 's1', 's2'], 'calibration': ['s3', 's4'], 'ants_template': ['s5'], 'validation': ['s6', 's7']}))
        common = ['--checkpoint', str(self.checkpoint_path), '--sets', str(sets)]
        for name in ('null', 'calibration', 'validation'):
            with mock.patch.object(sys, 'argv', ['x', *common, '--set', name, '--output', str(d / 'refit'), '--align-epochs', '1',
                                                 '--total-epochs', '2', '--steps-per-epoch', '1', '--device', 'cpu']):
                refit_subjects.main()
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'graph'),
                                             '--fitted', str(d / 'refit' / 'fitted_validation_00.pt'), '--radii-mm', '3', '2',
                                             '--thresholds', '0.1', '--min-skeleton-voxels', '1', '--workers', '1']):
            graph_atlas.main()
        manifest = d / 'manifest.json'
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--graph-atlas', str(d / 'graph'),
                                             '--sets', str(sets), '--refit', str(d / 'refit'), '--output', str(manifest),
                                             '--radius-mm', '100', '--min-p', '0', '--min-c', '0', '--min-radius-mm', '0']):
            perturbation_manifest.main()
        m = json.loads(manifest.read_text())
        self.assertGreater(len(m['copies']), 0)
        self.assertEqual(m['copies'][0]['kind'], 'remove')                   # priority order
        with mock.patch.object(sys, 'argv', ['x', *common, '--set', 'perturbed', '--perturbations', str(manifest), '--chunk-size', '3',
                                             '--output', str(d / 'refit'), '--align-epochs', '1', '--total-epochs', '2',
                                             '--steps-per-epoch', '1', '--device', 'cpu']):
            refit_subjects.main()                                             # first chunk only: partial copies are allowed
        with mock.patch.object(sys, 'argv', ['x', *common, '--set', 'valrep1', '--output', str(d / 'refit'), '--align-epochs', '1',
                                             '--total-epochs', '2', '--steps-per-epoch', '1', '--device', 'cpu']):
            refit_subjects.main()
        with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'dec'),
                                             '--shard', '0', '--num-shards', '1', '--device', 'cpu', '--chunk-size', '64', '--save-subjects']):
            probabilistic_atlas.main()
        out = d / 'deviation'
        with mock.patch.object(sys, 'argv', ['x', *common, '--decodings', str(d / 'dec'), '--refit', str(d / 'refit'), '--output', str(out),
                                             '--min-skeleton-voxels', '1', '--radii-mm', '3', '2', '--age-sigmas', '7',
                                             '--smooth-mm', '0', '--recalibration', 'none', '--perturbations', str(manifest), '--workers', '1',
                                             '--replicate-sets', 'valrep1', '--save-elements']):
            deviation.main()
        summary = json.loads((out / 'deviation.json').read_text())
        dump = np.load(out / 'elements_learned.npz', allow_pickle=False)
        keys = list(dump['query_keys'])
        self.assertIn('null/s0', keys); self.assertIn('valrep1/s6', keys)
        self.assertTrue(any(k.startswith('perturbed/') for k in keys))
        n_train = len(dump['train_ids'])
        for stat in ('seg_radius', 'offset', 'coverage'):
            self.assertEqual(dump[f'r3mm_tau0.1/train_{stat}'].shape[0], n_train)
            self.assertEqual(dump[f'r3mm_tau0.1/query_{stat}'].shape[0], len(keys))
        self.assertEqual(dump['r3mm_tau0.1/train_node_radius'].shape[-1], 2)
        self.assertEqual(dump['r3mm_tau0.1/train_presence'].shape, dump['r3mm_tau0.1/train_coverage'].shape)
        self.assertEqual(dump['r3mm_tau0.1/query_node_radius'].shape[:2], (len(keys), dump['r3mm_tau0.1/train_node_radius'].shape[1]))
        for name in ('query_floor', 'query_pop_p_radius', 'query_pop_p_seg_radius', 'query_pop_p_offset', 'query_pop_p_coverage', 'query_presence'):
            self.assertEqual(dump[f'r3mm_tau0.1/{name}'].shape[0], len(keys))
        # code fact used by D0/D1: presence == (coverage >= edge_cover) on the same points and radius
        cov, pres = dump['r3mm_tau0.1/query_coverage'], dump['r3mm_tau0.1/query_presence']
        np.testing.assert_array_equal(pres.astype(bool), cov >= 0.5)
        self.assertEqual(dump['r3mm_tau0.1/query_node_radius_allr'].shape, dump['r3mm_tau0.1/query_node_radius'].shape[:2])
        self.assertEqual(dump['r3mm_tau0.1/train_node_radius_allr'].shape[0], n_train)
        for name in ('query_segrad_match_med', 'query_presence3', 'query_pop_p_segmatch_med'):
            self.assertEqual(dump[f'r3mm_tau0.1/{name}'].shape[0], len(keys))
        self.assertEqual(dump['r3mm_tau0.1/train_segrad_match_q25'].shape[0], n_train)
        fam_ = np.asarray(summary['apriori_families']['learned/r3mm_tau0.1'], int)
        outside = np.setdiff1d(np.arange(len(dump['r3mm_tau0.1/pop_edges'])), fam_)
        self.assertTrue(np.isnan(dump['r3mm_tau0.1/query_pop_p_segmatch_q25_apriori'][:, outside]).all())   # not tested outside F
        np.testing.assert_array_equal(fam_, np.flatnonzero(dump['r3mm_tau0.1/pop_edge_probability'] >= 0.5))
        for name in ('query_edge_p', 'query_defects', 'query_radius_z', 'query_radius_allr_z', 'query_pop_p_radius_allr',
                     'query_reject_population_radius_allr', 'query_reject_radius_absz', 'query_reject_population_edge_surprise'):
            self.assertEqual(dump[f'r3mm_tau0.1/{name}'].shape[0], len(keys))
        pop_edges = dump['r3mm_tau0.1/pop_edges']
        self.assertEqual(len(dump['r3mm_tau0.1/pop_path_offsets']), len(pop_edges) + 1)
        geo = list(dump['geometry_keys'])
        self.assertIn('validation/s6', geo); self.assertIn('validation/s7', geo)
        self.assertEqual(len(dump['geometry_large_offsets']), len(geo) + 1)
        self.assertEqual(dump['geometry_mip_axial'].shape[0], len(geo))
        self.assertTrue((out / 'elements_centroid.npz').is_file())
        self.assertFalse((out / 'elements_decoder.npz').exists())
        from analysis.deviation import segment_diagnostics
        with mock.patch.object(sys, 'argv', ['x', str(out)]):
            segment_diagnostics.main()
        d0 = json.loads((out / 'd0_report.json').read_text())
        self.assertEqual(set(d0['nodes']['learned']['remove0']['node_counts']), {'copy', 'baseline'})
        self.assertIn('all_edges', d0['offset']['centroid'])
        self.assertAlmostEqual(d0['offset']['centroid']['all_edges']['median_of_edge_median_shift'], 0.)   # centroid: no refit
        self.assertEqual(set(d0['nodes_allr']['learned']['remove0']['node_counts']), {'copy', 'baseline'})   # E1 node diagnostic
        self.assertEqual(set(d0['segments_match']['learned']['remove0']['edge_counts']), {'copy', 'baseline'})   # F segment diagnostic
        from analysis.deviation import segment_zero_diagnostics
        with mock.patch.object(sys, 'argv', ['x', str(out), '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'f0.json'),
                                             '--workers', '1', '--max-seg-radius', '1e9', '--any-presence']):
            segment_zero_diagnostics.main()
        f0 = json.loads((d / 'f0.json').read_text())
        from analysis.deviation import detectability
        with mock.patch.object(sys, 'argv', ['x', str(out), '--output', str(d / 'g0.json')]):
            detectability.main()
        g0 = json.loads((d / 'g0.json').read_text())
        self.assertIn('remove0', g0['g01']['learned']['segmatch_med']['labels'])
        self.assertIn('segmatch_q25_apriori', g0['g01']['learned'])
        self.assertIn('sigma_fit', next(iter(g0['g02']['learned']['radius_absz']['elements'].values())))
        self.assertTrue(f0['targets']); self.assertIn('classes', next(iter(f0['targets'].values())))
        from analysis.deviation import plot_deviations
        figs = d / 'figures'
        with mock.patch.object(sys, 'argv', ['x', str(out), str(figs)]):
            plot_deviations.main()
        for name in ('subject_s6.png', 'subject_s7.png', 'montage.png', 'absent_expected_summary.png', 'index.html'):
            self.assertTrue((figs / name).is_file(), name)
        self.assertIn('data:image/png;base64', (figs / 'index.html').read_text())
        self.assertEqual(summary['perturbations']['copies_scored'], min(3, len(m['copies'])))
        sens = summary['sensitivity']['learned/uniform']
        self.assertIn('remove0', sens['by_perturbation'])
        self.assertIn('population_any', sens['by_perturbation']['remove0'])
        self.assertIn('subject_detection', summary['sensitivity']['decoder/uniform']['by_perturbation']['remove0'])
        self.assertNotIn('learned/age_sigma7', summary['sensitivity'])       # perturbed copies: uniform weighting only
        row = sens['copies'][0]
        for key in ('population_seg_radius', 'null_offset', 'population_coverage', 'population_any_new', 'transform_rms_mm',
                    'replicate_subject_detected', 'replicate_population_any', 'subject_detected_ext'):
            self.assertIn(key, row)
        self.assertIn('label', row)
        val = summary['fpr']['learned/uniform']['validation']
        self.assertIn('seg_radius', val['elements_bh_population']['r3mm_tau0.1']); self.assertIn('any_new', val['elements_bh']['r3mm_tau0.1'])
        self.assertIn('primary_family_minp_ext', val)
        for fam in ('seg_radius_cond', 'coverage_cond', 'seg_radius_condpresent', 'coverage_condpresent', 'any_new_cond', 'any_new_condpresent'):
            self.assertIn(fam, val['elements_bh_population']['r3mm_tau0.1']); self.assertIn(fam, val['elements_bh']['r3mm_tau0.1'])
        self.assertIn('discovery_fraction_absent_query', val['elements_bh_population']['r3mm_tau0.1']['seg_radius_cond'])
        self.assertIn('primary_family_minp_cond', val)
        for key in ('population_seg_radius_cond', 'null_coverage_cond', 'population_any_new_cond', 'null_any_new_condpresent',
                    'subject_detected_cond', 'baseline_subject_detected_cond', 'replicate_subject_detected_cond', 'target_present_after'):
            self.assertIn(key, row)
        self.assertIn('subject_detection_cond', sens['by_perturbation']['remove0'])
        self.assertIn('population_any_new_cond', sens['by_perturbation']['remove0']['per_edge'][str(row['population_edge'])])
        self.assertEqual(row['population_any_all'], row['population_any'] or row['population_any_new'])   # any_all: old + Step B only
        self.assertIn('radius_allr', val['elements_bh_population']['r3mm_tau0.1']); self.assertIn('any_allr', val['elements_bh']['r3mm_tau0.1'])
        for key in ('population_radius_allr', 'null_radius_allr', 'population_any_allr', 'baseline_population_radius_allr'):
            self.assertIn(key, row)
        self.assertIn('population_radius_allr', sens['by_perturbation']['remove0']['per_edge'][str(row['population_edge'])])
        for fam in ('segmatch_med', 'segmatch_q25', 'presence3', 'any_f'):
            self.assertIn(fam, val['elements_bh_population']['r3mm_tau0.1']); self.assertIn(fam, val['elements_bh']['r3mm_tau0.1'])
        for key in ('population_segmatch_med', 'null_segmatch_q25', 'population_presence3', 'population_any_f', 'target_matched_after'):
            self.assertIn(key, row)
        for fam in ('segmatch_q25_apriori', 'segmatch_med_apriori'):                                   # P1
            self.assertIn(fam, val['elements_bh_population']['r3mm_tau0.1']); self.assertIn(fam, val['elements_bh']['r3mm_tau0.1'])
            self.assertIn(f'population_{fam}', row); self.assertIn(f'null_{fam}', row)
        self.assertIn('learned/r3mm_tau0.1', summary['apriori_families'])
        rep = summary['replicates']['learned/uniform']
        self.assertEqual(rep['valrep1']['fpr']['subjects'], 2)
        self.assertEqual(len(rep['transform_shift_mm']['valrep1']), 2)
        self.assertIn('uncov_0.25_uncal', rep['score_variability'])
        self.assertEqual(set(scores_sets(out)), {'null', 'validation'})       # replicates / copies not in scores.tsv
        self.assertFalse(any(p.name.startswith('s6__') for p in (out / 'subjects').iterdir()))  # no reports for copies


def scores_sets(out):
    return pd.read_csv(out / 'scores.tsv', sep='\t', keep_default_na=False, na_values=['']).set.unique()


class DetectabilityTests(unittest.TestCase):
    """analysis/deviation/detectability.py and phantom.py (Stage G0)."""

    def test_helpers(self):
        from analysis.deviation.detectability import decompose, pooled_within_sd, robust_sd, z_needed
        self.assertAlmostEqual(robust_sd(np.array([0., 1., 2., 3., 4.])), 2. / 1.349)
        self.assertAlmostEqual(z_needed(0.05, 1), 1.959964, places=5)
        self.assertGreater(z_needed(0.05, 16), 2.9)
        self.assertAlmostEqual(decompose(5., 3.), 4.); self.assertEqual(decompose(1., 2.), 0.)
        v = np.array([[1., 3.], [10., 12.], [np.nan, 5.]])
        self.assertAlmostEqual(pooled_within_sd(v), np.sqrt(2.))          # var 2 and 2; single-fit subject ignored

    def test_phantom_axis_aligned(self):
        from analysis.deviation.phantom import measure, tube
        path = np.c_[np.linspace(0., 20., 41), np.zeros(41), np.zeros(41)]
        for spacing, tol in (((0.9375, 0.9375, 0.8), 0.6), ((0.25, 0.25, 0.25), 0.3)):
            sdf, affine, graph = tube(path, np.full(len(path), 2.), spacing, np.zeros(3))
            m = measure(sdf, affine, graph, path)
            self.assertLess(abs(m['segmatch_med'] - 2.), tol, spacing)
            self.assertLess(abs(m['node_allr'] - 2.), tol, spacing)
        fine = (0.25, 0.25, 0.25)
        thin = measure(*tube(path, np.full(len(path), 1.), fine, np.zeros(3)), path)
        base = measure(*tube(path, np.full(len(path), 2.), fine, np.zeros(3)), path)
        slope = (thin['segmatch_med'] - base['segmatch_med']) / (1. - 2.)
        self.assertTrue(0.7 <= slope <= 1.3, slope)
        self.assertTrue(np.isnan(thin['node_absz']))                       # radius_absz rule: graph radius 1 < 1.2
        gone = measure(*tube(path, np.full(len(path), 0.2), (0.9375, 0.9375, 0.8), np.zeros(3)), path)   # nearest centre 0.49 mm off-axis
        self.assertTrue(all(np.isnan(v) for v in gone.values()))           # vessel vanished on the grid

    def test_phantom_curved_and_stenosis(self):
        from analysis.deviation.phantom import centreline, measure, profile, tube
        rng = np.random.default_rng(0)
        path = centreline('curved', rng)
        self.assertAlmostEqual(np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), 20., delta=0.2)
        r = profile(path, 2., 'stenosis', 0.3)
        self.assertAlmostEqual(r.min(), 0.6, places=6); self.assertAlmostEqual(r[0], 2.)
        m = measure(*tube(path, r, (0.25, 0.25, 0.25), np.zeros(3)), path)
        self.assertLess(m['segmatch_q25'], m['segmatch_med'])


    def test_phantom_summary(self):
        from analysis.deviation import phantom
        res = [phantom.placement((2.0, k, s)) for k in ('straight', 'curved') for s in range(2)]
        with mock.patch.object(phantom, 'RADII', np.array([2.0])):
            rep = phantom.summarise(res)
        e = rep['straight/r2.0']['segmatch_med']
        self.assertLess(abs(e['bias']), 0.6); self.assertIn('slope', e['thin0.5'])
        self.assertAlmostEqual(e['thin0.5']['true_change_median'], -1.0)


    def test_decision_rules(self):
        from analysis.deviation.decision_table import classify
        ph = {'slope': 0.9, 'delta_phantom': -0.7, 'sd': 0.05}
        self.assertEqual(classify(-0.2, ph, 0.25, 0.24, 0.25, 2.5, 2.9), 'dilution')
        self.assertEqual(classify(-0.6, ph, 0.25, 0.24, 0.25, 2.5, 2.9), 'between-subject variance')
        self.assertEqual(classify(-0.6, dict(ph, slope=0.4), 0.25, 0.24, 0.25, 2.5, 2.9), 'resolution limit')
        self.assertEqual(classify(-0.6, ph, 0.25, 0.1, 0.25, 2.5, 2.9), 'unclear')
        self.assertTrue(classify(0., None, 0.25, 0.24, 0.25, 0., 2.9).startswith('dilution'))


class SegmentDiagnosticsTests(unittest.TestCase):
    """analysis/deviation/segment_diagnostics.py (Step D0)."""

    def test_floor_p_and_zero_fraction(self):
        from analysis.deviation.segment_diagnostics import bh_rank1, min_population_p, zero_fraction
        col = np.array([0., 0., 0.05, 1., 1.5, 2., 2.5, 3.])
        self.assertAlmostEqual(zero_fraction(col, 0.1), 3 / 8)
        self.assertAlmostEqual(min_population_p(col, 'seg_radius'), min(1., 2 * (2 + 1) / (8 + 1)))   # v = 0, two-sided (2 zeros)
        self.assertAlmostEqual(min_population_p(col, 'coverage'), (2 + 1) / (8 + 1))                 # v = 0, lower tail (ties)
        self.assertAlmostEqual(min_population_p(np.array([1., 2., 3.]), 'coverage'), 1 / 4)
        self.assertAlmostEqual(bh_rank1(0.05, np.array([0.3, np.nan, 0.2, 0.9])), 0.05 / 3)

    def test_node_categories(self):
        from analysis.deviation.segment_diagnostics import node_categories
        node_r = np.array([[1.5, 1.4], [np.nan, np.nan], [0.5, 0.7], [1.2, 1.3]])
        floor = np.array([False, False, True, False]); tested = np.array([True, False, False, False])
        self.assertEqual(list(node_categories(node_r, floor, tested)), ['tested', 'absent', 'floor', 'untested'])

    def test_large_vessel_points_never_floor(self):
        """Graph radius of large-vessel points is interpolated between endpoint radii >= 1.2 mm: never <= floor."""
        from vatl_sdf.vessel_attribute_atlas import FLOOR_MM, centreline_samples
        graph = {'pos': np.array([[0., 0, 0], [5, 0, 0], [9, 0, 0]]), 'radius': np.array([1.2, 2.0, 0.5]),
                 'edges': np.array([[0, 1], [1, 2]])}
        points, eid, r = centreline_samples(graph)
        large = (graph['radius'][graph['edges'][eid]] >= 1.2).all(1)
        self.assertTrue(large.any()); self.assertFalse((r[large] <= FLOOR_MM).any())

    def test_paired_offset_shift(self):
        from analysis.deviation.segment_diagnostics import distribution_summary
        s = distribution_summary(np.array([1., 2., 3., 4.]), np.array([2., 3., 4., 5.]), paired=np.array([1., 1., 1., 1.]))
        self.assertAlmostEqual(s['train_median'], 2.5); self.assertAlmostEqual(s['null_median'], 3.5)
        self.assertAlmostEqual(s['paired_shift_median'], 1.)
        self.assertAlmostEqual(s['train_iqr'], 1.5)


class DeviationStatisticsTests(unittest.TestCase):
    def test_weights_and_smoothing(self):
        from vatl_sdf.deviation import ess, kernel_weights, smoothed
        w = kernel_weights([20, 30, 40], 30, None, exclude=1)
        np.testing.assert_array_equal(w, [1, 0, 1])
        self.assertAlmostEqual(ess(w), 2.)
        self.assertAlmostEqual(float(smoothed(1., w.sum(), w)), (1 + 1) / (2 + 2))   # uniform: (k + 1) / (n + 2)
        w7 = kernel_weights([20, 30, 40], 30, 7.)
        self.assertLess(ess(w7), 3.); self.assertAlmostEqual(w7[1], 1.)

    def test_stack_loo_matches_weighted(self):
        from vatl_sdf.deviation import Stack, kernel_weights
        rng = np.random.default_rng(0)
        y = rng.uniform(size=(5, 40)) < 0.3; v = rng.uniform(size=(5, 40)) < 0.9
        support = np.flatnonzero(y.any(0))
        stack = Stack([np.packbits(r) for r in y], [np.packbits(r) for r in v], support)
        w = kernel_weights(np.zeros(5), 0., None, exclude=2)
        np.testing.assert_allclose(stack.values(w, uniform_exclude=2), stack.values(w), atol=1e-6)
        num, den = (y & v)[[0, 1, 3, 4]][:, support].sum(0), v[[0, 1, 3, 4]][:, support].sum(0)
        np.testing.assert_allclose(stack.values(w), np.where(den > 0, num / np.maximum(den, 1), 0), atol=1e-6)

    def test_pvalues_bh_holm(self):
        from vatl_sdf.deviation import benjamini_hochberg, empirical_p, holm, radius_deviation
        self.assertAlmostEqual(empirical_p(5., [1, 2, 3, 6]), 2 / 5)
        self.assertAlmostEqual(empirical_p(10., [1, 2, 3, 6]), 1 / 5)
        self.assertTrue(np.isnan(empirical_p(np.nan, [1.])))
        np.testing.assert_array_equal(benjamini_hochberg([0.001, 0.02, 0.04, 0.5, np.nan], 0.05), [True, True, False, False, False])
        np.testing.assert_array_equal(holm([0.01, 0.03, 0.2], 0.05), [True, False, False])
        dist = np.array([[1.], [2.], [3.], [4.], [np.nan]])
        out = radius_deviation(np.array([2.5]), dist, np.ones(5))
        self.assertAlmostEqual(out[0, 0], 0.); self.assertAlmostEqual(out[0, 2], 0.5)
        self.assertAlmostEqual(out[0, 3], 1.)                               # median value: two-sided p = 1
        far = radius_deviation(np.array([9.]), dist, np.ones(5))
        self.assertAlmostEqual(far[0, 3], 2 * 1 / 5)                        # (0 + c) / (4 + c), c = 1

    def test_tail_p(self):
        from vatl_sdf.deviation import tail_p
        dist = np.array([[1.], [2.], [3.], [4.], [np.nan]])
        np.testing.assert_allclose(tail_p(np.array([9.]), dist, np.ones(5), 'upper'), [1 / 5])     # (0 + c) / (4 + c)
        np.testing.assert_allclose(tail_p(np.array([0.]), dist, np.ones(5), 'lower'), [1 / 5])
        np.testing.assert_allclose(tail_p(np.array([3.]), dist, np.ones(5), 'upper'), [3 / 5])
        self.assertTrue(np.isnan(tail_p(np.array([np.nan]), dist, np.ones(5), 'upper')[0]))

    def test_interior_samples(self):
        from vatl_sdf.deviation import interior_samples
        path = np.c_[np.arange(0., 21.), np.zeros(21), np.zeros(21)]
        idx = interior_samples(path, path[0], path[-1])
        np.testing.assert_array_equal(idx, np.arange(3, 18))                 # 3 mm margin
        short = path[:5]
        np.testing.assert_array_equal(interior_samples(short, short[0], short[-1]), [1, 2, 3])   # margin |a-b|/4 = 1
        self.assertEqual(len(interior_samples(path[:2], path[0], path[1])), 1)   # fallback: farthest sample

    def test_segment_statistics(self):
        from vatl_sdf.deviation import segment_statistics
        from vatl_sdf.perturbation import perturb
        import pickle
        from vatl_sdf.graph_coords import edge_samples, load_graph
        sdf, affine, g = tube_case()
        with tempfile.TemporaryDirectory() as t:
            with open(Path(t) / 'g.pkl', 'wb') as f:
                pickle.dump({'graph': g}, f)
            graph = load_graph(str(Path(t) / 'g.pkl'))
        centre = np.zeros(3)
        pop = {'edge_points_mm': [np.c_[np.arange(15., 26.), np.full(11, 12.), np.full(11, 12.)]], 'edges': [(0, 1)],
               'nodes_mm': np.array([[15., 12., 12.], [25., 12., 12.], [20., 12., 12.], [100., 100., 100.]]), 'radius_mm': 3.}

        def stats(s, gr):
            large = edge_samples(gr['pos'], gr['edges'][(gr['radius'][gr['edges']] >= 1.2).all(1)], 0.4)[0]
            alll, eid = edge_samples(gr['pos'], gr['edges'], 0.4)
            return segment_statistics(s, affine, centre, np.eye(3), np.zeros(3), large, alll, pop, all_eid=eid)
        base = stats(sdf, graph)
        self.assertAlmostEqual(base['seg_radius'][0], 2., places=1)
        self.assertAlmostEqual(base['offset'][0], 0., places=6); self.assertAlmostEqual(base['coverage'][0], 1.)
        self.assertAlmostEqual(base['node_radius_allr'][2], 2., places=1)   # all-radius node radius (E1)
        self.assertTrue(np.isnan(base['node_radius_allr'][3]))               # no graph point within r -> NaN
        self.assertAlmostEqual(base['segrad_match_med'][0], 2., places=1)    # F1: own matched segment interior
        self.assertEqual(base['presence3'][0], 2)                            # F2: large
        edges = [int(k) for k, (a, b) in enumerate(graph['edges']) if 10 <= a <= 20 and 10 <= b <= 20]
        thin = stats(*perturb(sdf, affine, graph, edges, 'thin'))
        self.assertAlmostEqual(thin['seg_radius'][0], 1., places=1)
        self.assertAlmostEqual(thin['node_radius_allr'][2], 1., places=1)   # thinned points still contribute
        self.assertAlmostEqual(thin['segrad_match_med'][0], 1., places=1)
        self.assertLessEqual(thin['segrad_match_q25'][0], thin['segrad_match_med'][0])
        self.assertEqual(thin['presence3'][0], 2 if thin['coverage'][0] >= 0.5 else 1)   # short tube: large lines reach in from both ends
        alll, eid = edge_samples(graph['pos'], graph['edges'], 0.4)
        small = segment_statistics(sdf, affine, centre, np.eye(3), np.zeros(3), np.zeros((0, 3)), alll, pop, all_eid=eid)
        self.assertEqual(small['presence3'][0], 1)                           # all-radius covered, no large-vessel line: small
        self.assertAlmostEqual(thin['offset'][0], 0., places=6)            # all-radius centreline unmoved
        self.assertLess(thin['coverage'][0], 0.8)                          # radius 1 < 1.2: interior off the large-vessel centreline
        moved = stats(*perturb(sdf, affine, graph, edges, 'displace', 3., [0., 0., 1.]))
        self.assertGreater(moved['offset'][0], 2.)
        gone = stats(*perturb(sdf, affine, graph, edges, 'remove'))
        self.assertLess(gone['node_radius_allr'][2], 1.)                     # unclipped: remaining points outside the vessel
        expected = 2 if gone['coverage'][0] >= 0.5 else (1 if gone['coverage_all'][0] >= 0.5 else 0)
        self.assertEqual(gone['presence3'][0], expected)                     # category rule (short tube: ends still covered)
        self.assertFalse(gone['segrad_match_med'][0] > 1.)                   # NaN (no match) or side-branch root outside the vessel
        self.assertLess(gone['seg_radius'][0], 0.5); self.assertLess(gone['coverage'][0], 0.8)   # (small branch root keeps offset low)

    def test_conditional_segment_p(self):
        from vatl_sdf.deviation import conditional_segment_p, radius_deviation
        seg_tr = np.array([[0.], [0.], [0.], [2.], [2.1], [1.9], [2.2], [1.8]])
        cov_tr = np.array([[0.], [0.1], [0.2], [1.], [0.9], [0.8], [1.], [0.7]])
        present = cov_tr >= 0.5
        w = np.ones(8)
        out = conditional_segment_p(np.array([0.5]), seg_tr, np.array([0.3]), cov_tr, present, np.array([False]), w)
        uncond = radius_deviation(np.array([0.5]), seg_tr, w)[0, 3]
        self.assertAlmostEqual(out['pop_p_seg_radius_cond'][0], 2 * 1 / 6)        # (0 + c) / (5 + c) two-sided
        self.assertGreater(uncond, out['pop_p_seg_radius_cond'][0])
        self.assertAlmostEqual(out['pop_p_coverage_cond'][0], 1 / 6)               # absent query: smallest attainable p
        self.assertTrue(np.isnan(out['pop_p_seg_radius_condpresent'][0]))          # present-only row: absent query -> NaN
        self.assertTrue(np.isnan(out['seg_radius_condpresent_absz'][0]))
        self.assertLess(out['seg_radius_cond_z'][0], -5)
        kept = conditional_segment_p(np.array([1.9]), seg_tr, np.array([0.9]), cov_tr, present, np.array([True]), w)
        self.assertAlmostEqual(kept['pop_p_seg_radius_condpresent'][0], kept['pop_p_seg_radius_cond'][0])
        self.assertAlmostEqual(kept['coverage_condpresent_deficit'][0], 0.1)

    def test_conditional_null_p(self):
        from vatl_sdf.deviation import element_null_p
        nulls = np.array([[1.], [2.], [3.], [9.]]); present = np.array([[True], [True], [True], [False]])
        np.testing.assert_allclose(element_null_p(np.array([5.]), nulls), [2 / 5])
        np.testing.assert_allclose(element_null_p(np.array([5.]), nulls, present), [1 / 4])   # absent null subject left out

    def test_masked_mips(self):
        from vatl_sdf.deviation import masked_mips
        y = np.zeros((4, 3, 2), bool); y[3, :, :] = True; y[0, 0, 0] = True       # x = 3 slab: extrapolated outside the FOV
        valid = np.ones_like(y); valid[3] = False
        mips = masked_mips(np.packbits(y.ravel()), np.packbits(valid.ravel()), y.shape)
        self.assertEqual(mips['axial'].shape, (4, 3)); self.assertEqual(mips['sagittal'].shape, (3, 2))
        self.assertEqual(int(mips['axial'].sum()), 1); self.assertTrue(mips['axial'][0, 0])

    def test_matched_interior(self):
        from vatl_sdf.deviation import matched_edges, matched_interior
        from vatl_sdf.graph_coords import edge_samples
        pos = np.array([[float(i), 0., 0.] for i in range(21)] + [[10., 8., 0.]])
        edges = np.array([[i, i + 1] for i in range(20)] + [[10, 21]])
        pts, eid = edge_samples(pos, edges, 0.4)
        path = np.c_[np.arange(5., 15.1, 0.5), np.zeros(21), np.zeros(21)]
        m = matched_edges(pts, eid, path, 3.)
        self.assertEqual(set(m), set(range(2, 18)))                        # >= 50% of samples within 3 mm; side branch 0-8 mm: < 0.5
        keep, m2 = matched_interior(pts, eid, path, path[0], path[-1], 3.)
        np.testing.assert_array_equal(m, m2)
        d = np.minimum(np.linalg.norm(pts[keep] - path[0], axis=1), np.linalg.norm(pts[keep] - path[-1], axis=1))
        self.assertTrue((d >= 2.5 - 1e-9).all())                           # margin min(3, 10 / 4)
        far, none = matched_interior(pts + [0., 20., 0.], eid, path, path[0], path[-1], 3.)
        self.assertEqual(len(far), 0); self.assertEqual(len(none), 0)

    def test_categorical_p(self):
        from vatl_sdf.deviation import categorical_p, smoothed_categories
        cats = np.array([[2], [2], [2], [2], [2], [2], [1], [0]])
        pi = smoothed_categories(cats, np.ones(8))
        np.testing.assert_allclose(pi[:, 0], [(1 + 1 / 3) / 9, (1 + 1 / 3) / 9, (6 + 1 / 3) / 9])
        np.testing.assert_allclose(pi.sum(0), [1.])
        self.assertAlmostEqual(categorical_p(pi, np.array([1]))[0], 2 * (1 + 1 / 3) / 9)    # ties: both rare categories
        self.assertAlmostEqual(categorical_p(pi, np.array([2]))[0], 1.)

    def test_zero_causes(self):
        from analysis.deviation.segment_zero_diagnostics import zero_cause
        d = np.array([5., 5., 1., 1., 0.5]); raw = np.array([2., 2., -0.3, 1.5, 1.4])
        self.assertEqual(zero_cause(d, raw, 3.), ('no_point', 2, 1))      # 2 no-point zeros vs 1 clipped
        self.assertEqual(zero_cause(np.array([1., 1., 1.]), np.array([-1., -1., 2.]), 3.), ('clipped', 0, 2))
        self.assertEqual(zero_cause(np.array([1., 1.]), np.array([1., 2.]), 3.), ('other', 0, 0))

    def test_transform_shift(self):
        from vatl_sdf.deviation import transform_shift
        rng = np.random.default_rng(0)
        A = np.eye(3) * 1.1; t = np.array([1., 2., 3.]); native = rng.normal(size=(50, 3))
        lines = native @ A.T + t
        self.assertAlmostEqual(transform_shift(lines, A, t, A, t + [0., 0., 2.]), 2.)
        self.assertAlmostEqual(transform_shift(lines, A, t, A, t), 0.)

    def test_outcome_p(self):
        from vatl_sdf.deviation import outcome_p
        np.testing.assert_allclose(outcome_p(np.array([0.9, 0.9, 0.2, 0.2]), np.array([False, True, True, False])), [0.1, 1., 0.2, 1.])


def tube_case():
    """Straight tube along x (radius 2 mm) in a 1 mm grid, graph = chain of nodes every mm plus a side branch."""
    import networkx as nx
    shape = (40, 24, 24); affine = np.eye(4)
    world = np.indices(shape).reshape(3, -1).T.astype(float)
    sdf = (np.linalg.norm(world[:, 1:] - 12., axis=1) - 2.).reshape(shape).astype(np.float32)
    g = nx.Graph()
    for i in range(5, 36):
        g.add_node(i - 5, pos=np.array([i, 12., 12.]), radius=2.)
    g.add_node(31, pos=np.array([20., 18., 12.]), radius=1.)
    g.add_edges_from([(k, k + 1) for k in range(30)] + [(15, 31)])
    from vatl_sdf.graph_coords import load_graph
    import pickle
    return sdf, affine, g


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        import pickle
        from vatl_sdf.graph_coords import load_graph
        self.temp = tempfile.TemporaryDirectory()
        self.sdf, self.affine, g = tube_case()
        path = Path(self.temp.name) / 'g.pkl'
        with open(path, 'wb') as f:
            pickle.dump({'graph': g}, f)
        self.graph = load_graph(str(path)); self.graph_path = str(path)
        # segment = edges between nodes 10..20 (x = 15..25)
        self.edges = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if 10 <= a <= 20 and 10 <= b <= 20]

    def tearDown(self):
        self.temp.cleanup()

    def test_remove(self):
        from vatl_sdf.perturbation import perturb
        sdf, g = perturb(self.sdf, self.affine, self.graph, self.edges, 'remove')
        self.assertGreater(sdf[20, 12, 12], 0.)                 # segment cleared
        self.assertLess(sdf[8, 12, 12], 0.); self.assertLess(sdf[32, 12, 12], 0.)   # rest of the tube intact
        np.testing.assert_array_equal(sdf[:, :, :2], self.sdf[:, :, :2])            # far away unchanged
        self.assertEqual(len(g['edges']), len(self.graph['edges']) - len(self.edges))
        self.assertTrue(np.all(np.bincount(g['edges'].ravel(), minlength=len(g['pos'])) > 0))  # no orphan nodes

    def test_thin(self):
        from vatl_sdf.perturbation import perturb
        tail = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if a >= 21 and b >= 21 and max(a, b) <= 30]  # x = 26..35
        sdf, g = perturb(self.sdf, self.affine, self.graph, tail, 'thin')
        self.assertAlmostEqual(float(sdf[32, 12, 12]), -1., places=1)  # radius 1 mm away from the attachment
        self.assertGreater(sdf[32, 13, 13], 0.)                       # 1.41 mm off axis: outside
        self.assertTrue((sdf[22:36, 12, 12] < 0).all())               # still connected through the attachment (x = 26)
        np.testing.assert_allclose(g['radius'][22:31], 1.); self.assertEqual(g['radius'][21], 2.)  # boundary node kept
        sdf, g = perturb(self.sdf, self.affine, self.graph, self.edges, 'thin')
        np.testing.assert_allclose(g['radius'][range(11, 20)], 1.)    # node 15 joins only a small branch: not an attachment
        self.assertEqual(g['radius'][10], 2.); self.assertEqual(g['radius'][20], 2.)  # attachments to the large tube kept
        self.assertLess(sdf[20, 12, 12], -0.5); self.assertGreater(sdf[20, 13, 13], 0.)  # thinned through the small-branch junction

    def test_thin_factor(self):
        from vatl_sdf.perturbation import perturb
        tail = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if a >= 21 and b >= 21 and max(a, b) <= 30]
        default, _ = perturb(self.sdf, self.affine, self.graph, tail, 'thin')
        half, _ = perturb(self.sdf, self.affine, self.graph, tail, 'thin', factor=0.5)
        np.testing.assert_array_equal(default, half)                  # legacy copies (no factor) regenerate identically
        sdf, g = perturb(self.sdf, self.affine, self.graph, tail, 'thin', factor=0.3)
        self.assertAlmostEqual(float(sdf[32, 12, 12]), -0.6, places=1)
        self.assertGreater(sdf[32, 13, 12], 0.)
        np.testing.assert_allclose(g['radius'][22:31], 0.6)

    def test_stenosis(self):
        from vatl_sdf.perturbation import perturb, stenosis_profile
        np.testing.assert_allclose(stenosis_profile(np.array([0., 2., 3., 4., 9.]), 0.5), [0.5, 0.5, 0.75, 1., 1.])
        sdf, g = perturb(self.sdf, self.affine, self.graph, self.edges, 'stenosis', factor=0.5)   # x = 15..25, centre x = 20
        self.assertAlmostEqual(float(sdf[20, 12, 12]), -1., places=1)   # plateau: radius 1 mm
        self.assertGreater(sdf[20, 13, 13], 0.)
        self.assertAlmostEqual(float(sdf[17, 12, 12]), -1.5, places=1)  # taper (d = 3 mm): radius 1.5 mm
        self.assertAlmostEqual(float(sdf[14, 12, 12]), -2., places=1)   # outside the 8 mm zone: original
        self.assertTrue((sdf[5:36, 12, 12] < 0).all())                  # lumen continuous
        np.testing.assert_array_equal(sdf[:9], self.sdf[:9])            # far away unchanged
        self.assertAlmostEqual(float(g['radius'][15]), 1.); self.assertAlmostEqual(float(g['radius'][12]), 1.5)
        self.assertAlmostEqual(float(g['radius'][10]), 2.); self.assertEqual(len(g['edges']), len(self.graph['edges']))
        np.testing.assert_array_equal(g['pos'], self.graph['pos'])

    def test_kinds_and_labels(self):
        from vatl_sdf.perturbation import copy_label, parse_kinds
        legacy = parse_kinds(['remove', 'thin', 'displace:3', 'displace:5'])
        self.assertEqual([copy_label(k) for k in legacy], ['remove0', 'thin0', 'displace3', 'displace5'])  # unchanged ids
        new = parse_kinds(['thin:0.3', 'stenosis:0.5', 'stenosis:0.3', 'displace:2'])
        self.assertEqual([copy_label(k) for k in new], ['thin0.3', 'stenosis0.5', 'stenosis0.3', 'displace2'])
        self.assertEqual(new[0], {'kind': 'thin', 'magnitude': 0., 'factor': 0.3})
        self.assertEqual(new[3], {'kind': 'displace', 'magnitude': 2., 'factor': None})
        with self.assertRaises(ValueError):
            parse_kinds(['bend:2'])

    def test_displace(self):
        from vatl_sdf.perturbation import perturb
        tail = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if a >= 21 and b >= 21 and max(a, b) <= 30]  # x = 26..35
        sdf, g = perturb(self.sdf, self.affine, self.graph, tail, 'displace', 3., [0., 0., 1.])
        self.assertGreater(sdf[33, 12, 12], 0.)                  # original axis vacated (shift 3 > radius 2)
        self.assertLess(sdf[33, 12, 15], 0.)                     # shifted tube present
        self.assertLess(sdf[24, 12, 12], 0.)                     # tapered to 0 at the attachment (node 21, x = 26)
        moved = np.linalg.norm(g['pos'] - self.graph['pos'], axis=1)
        self.assertLessEqual(moved.max(), 3. + 1e-9); self.assertEqual(moved[21], 0.); self.assertAlmostEqual(moved[29], 3.)

    def test_virtual_source_roundtrip(self):
        import nibabel as nib
        from vatl_sdf import perturbation, sources
        from vatl_sdf.graph_coords import load_graph
        nib.save(nib.Nifti1Image(self.sdf, self.affine), Path(self.temp.name) / 's.nii.gz')
        manifest = Path(self.temp.name) / 'm.json'
        manifest.write_text(json.dumps({'copies': [{'copy_id': 'c0', 'scan_age': 30., 'sex': 'F', 'source_sdf': str(Path(self.temp.name) / 's.nii.gz'),
                                                     'source_graph': self.graph_path, 'matched_edges': self.edges, 'kind': 'remove',
                                                     'magnitude': 0., 'direction': None}]}))
        perturbation.configure(manifest)
        image = sources.load_image('perturb:c0')
        expected, g = perturbation.perturb(self.sdf, self.affine, self.graph, self.edges, 'remove')
        np.testing.assert_array_equal(np.asarray(image.dataobj), expected)
        self.assertEqual(len(load_graph('perturb:c0')['edges']), len(g['edges']))
        second = Path(self.temp.name) / 'm2.json'   # a second manifest (Step C) is added to the first
        second.write_text(json.dumps({'copies': [{'copy_id': 'c1', 'scan_age': 30., 'sex': 'F', 'source_sdf': str(Path(self.temp.name) / 's.nii.gz'),
                                                   'source_graph': self.graph_path, 'matched_edges': self.edges, 'kind': 'stenosis',
                                                   'magnitude': 0., 'factor': 0.5, 'direction': None}]}))
        perturbation.configure(second, reset=False)
        expected, _ = perturbation.perturb(self.sdf, self.affine, self.graph, self.edges, 'stenosis', factor=0.5)
        np.testing.assert_array_equal(np.asarray(sources.load_image('perturb:c1').dataobj), expected)
        self.assertIsNotNone(sources.load_image('perturb:c0'))
        self.assertTrue(np.allclose(nib.load(Path(self.temp.name) / 's.nii.gz').get_fdata(), self.sdf))  # source untouched

    def test_selection(self):
        from vatl_sdf.perturbation import select_targets, subject_segments, willis_centre
        pop = {'edge_points_mm': [np.c_[np.arange(15., 26.), np.full(11, 12.), np.full(11, 12.)],
                                  np.c_[np.full(15, 20.), np.arange(3., 18.), np.full(15, 30.)]],
               'edge_probability': np.array([0.9, 0.3]), 'radius_mm': 3., 'edge_cover': 0.5}
        cmap = np.zeros((40, 24, 40)); cmap[15:26, 12, 12] = 0.8
        centre, chosen, c, _ = willis_centre(pop, cmap, np.eye(4), 0.5)
        self.assertEqual(chosen, [0]); np.testing.assert_allclose(centre, [20., 12., 12.])
        self.assertEqual(select_targets(pop, centre, 25., 0.5), [0])
        segs = subject_segments(self.graph, np.zeros(3), np.eye(3), np.zeros(3), pop, [0], 3., 0.5)
        self.assertTrue(segs[0]['present'] and segs[0]['eligible'])
        self.assertAlmostEqual(segs[0]['segment_radius_mm'], 2.)
        thin_graph = dict(self.graph, radius=np.full(len(self.graph['radius']), 1.3))
        self.assertFalse(subject_segments(thin_graph, np.zeros(3), np.eye(3), np.zeros(3), pop, [0], 3., 0.5)[0]['eligible'])


class ConfirmatoryCohortTests(unittest.TestCase):
    """P1 confirmatory run: cohort file and validation selection on the real split (reads the YAML / TSV only)."""
    SPLIT = '/medias/medical/data/Atlas_training/v1/subject_ids_label_t1_fast.yaml'
    KEY = 'mra_atlas_synthseg_label_t1_fast.subject_ids'

    def setUp(self):
        from vatl_sdf.config import load_config
        from vatl_sdf.splits import read_splits
        self.splits = read_splits(self.SPLIT, self.KEY, {'train': 497, 'val': 55, 'test': 0})
        self.config = load_config(Path('configs/sdf/21_a4_full_latent6.yaml'))
        self.confirm = json.loads(Path('configs/evaluation/deviation_sets_confirm.json').read_text())
        self.p1 = json.loads(Path('configs/evaluation/deviation_sets_null300.json').read_text())

    def test_cohort_file(self):
        self.assertEqual(self.confirm['validation'], self.splits['val'][20:55])
        self.assertEqual((self.confirm['validation'][0], self.confirm['validation'][-1]), ('IXI417-Guys-0939', 'IXI137-HH-1472'))
        self.assertFalse(set(self.confirm['validation']) & set(self.p1['validation']))
        self.assertFalse(set(self.confirm['validation']) & set(self.splits['train']))
        for key in set(self.p1) - {'validation', 'note'}:
            self.assertEqual(self.confirm[key], self.p1[key], key)

    def test_validation_selection(self):
        from analysis.deviation.refit_subjects import set_frame
        from analysis.generalization.evaluate_generalization import test_frame
        frame = pd.DataFrame({'subject_id': self.splits['train']})
        old = set_frame(self.config, frame, self.p1, 'validation')
        ref = test_frame(self.config, self.splits['train'], '', 20)            # P1 refits: first 20 YAML val IDs
        pd.testing.assert_frame_equal(old, ref)
        new = set_frame(self.config, frame, self.confirm, 'validation')
        self.assertEqual(new.subject_id.tolist(), self.splits['val'][20:55])
        self.assertEqual(set_frame(self.config, frame, self.confirm, 'valrep1').subject_id.tolist(), self.splits['val'][20:55])

    def test_copy_labels_match_p1(self):
        from vatl_sdf.perturbation import copy_label, parse_kinds
        kinds = ['remove', 'thin', 'displace:3', 'displace:5', 'thin:0.3', 'stenosis:0.5', 'stenosis:0.3', 'displace:2']
        self.assertEqual(sorted(copy_label(k) for k in parse_kinds(kinds)),
                         sorted(['thin0', 'thin0.3', 'remove0', 'stenosis0.5', 'stenosis0.3', 'displace2', 'displace3', 'displace5']))


class ConfirmatoryReportTests(unittest.TestCase):
    def test_diff_and_wilson(self):
        from analysis.deviation.report_confirmatory import diff, wilson
        a = {'x': [1, float('nan'), {'y': 2}], 'runtime_s': {'total': 1}, 'settings': {'output': 'a', 'q': 0.05}}
        self.assertEqual(diff(a, json.loads(json.dumps(a))), [])
        b = {'x': [1, float('nan'), {'y': 3}], 'runtime_s': {'total': 9}, 'settings': {'output': 'b', 'q': 0.05}}
        self.assertEqual(diff(a, b), [('x', 2, 'y')])                      # output path and runtime ignored
        self.assertEqual(diff(a, dict(b, z=1, x=a['x'])), [('z',)])
        lo, hi = wilson(0.0, 35)
        self.assertEqual(lo, 0.); self.assertAlmostEqual(hi, 0.0989, places=3)
        self.assertIsNone(wilson(0., 0))
