"""Connectivity and radius atlases (CPU, synthetic subject)."""
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np

from vatl_sdf.vessel_attribute_atlas import connectivity_stats, radius_errors, score_connectivity, subject_attributes

SHAPE = (30, 13, 13)


def write_subject(directory, name, graph_gap_edge, mask_gap):
    """Tube along x at (y, z) = (6, 6), radius 2 mm (identity affine, 1 mm voxels). mask_gap: tube interrupted at
    x in (13, 16) (caps of radius 2 end at 13 and 16). graph: nodes every 1 mm at radius 1.5 mm; graph_gap_edge: edge 7-8 removed (inside the tube)."""
    affine = np.eye(4)
    grid = np.indices(SHAPE).reshape(3, -1).T.astype(float)
    segments = [(2, 11), (18, 27)] if mask_gap else [(2, 27)]
    dist = np.full(len(grid), np.inf)
    for lo, hi in segments:
        x = np.clip(grid[:, 0], lo, hi)
        dist = np.minimum(dist, np.linalg.norm(grid - np.c_[x, np.full(len(x), 6.), np.full(len(x), 6.)], axis=1))
    sdf = (dist - 2.).reshape(SHAPE).astype(np.float32)
    image = nib.Nifti1Image(sdf, affine); image.header.set_xyzt_units('mm')
    nib.save(image, Path(directory) / f'{name}.nii.gz')
    g = nx.Graph()
    for k, x in enumerate(range(2, 28)):
        g.add_node(k, pos=np.array([x, 6., 6.]), radius=np.float32(1.5))
    for k in range(25):
        if not (graph_gap_edge and k == 7):
            g.add_edge(k, k + 1)
    with open(Path(directory) / f'{name}.pkl', 'wb') as f:
        pickle.dump({'graph': g}, f)
    return {'subject_id': name, 'SDF': str(Path(directory) / f'{name}.nii.gz'), 'Graph': str(Path(directory) / f'{name}.pkl')}


class SubjectAttributeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        centre = np.asarray(SHAPE, float) / 2.
        to_t = lambda x: np.array([x, 6., 6.]) - centre
        # population: edge 0 across the mask gap (x 5 -> 22), edge 1 inside the first tube piece (x 4 -> 12)
        self.population = {'r2': {'radius_mm': 2.0, 'nodes_mm': np.stack([to_t(5), to_t(22), to_t(4), to_t(12)]),
                                  'edges': [(0, 1), (2, 3)],
                                  'edge_points_mm': [np.stack([to_t(x) for x in np.arange(5, 23)]), np.stack([to_t(x) for x in np.arange(4, 13)])]}}
        self.params = {'min_radius_mm': 1.2, 'path_factor': 3.0, 'min_limit_mm': 2.0, 'snap_mm': 1.0,
                       'grid_affine': np.diag([1., 1., 1., 1.]) @ np.eye(4), 'grid_shape': np.array([40, 40, 40])}
        self.params['grid_affine'][:3, 3] = -20.

    def tearDown(self):
        self.temp.cleanup()

    def run_subject(self, **kwargs):
        row = write_subject(self.temp.name, 's', **kwargs)
        return subject_attributes((row, np.eye(3), np.zeros(3), self.population, self.params))

    def test_graph_connected_across_mask_gap(self):
        out = self.run_subject(graph_gap_edge=False, mask_gap=True)['populations']['r2']
        self.assertTrue(out['present'].all())
        np.testing.assert_array_equal(out['both'], [True, True])
        np.testing.assert_array_equal(out['graph'], [True, True])
        np.testing.assert_array_equal(out['mask'], [False, True])     # mask interrupted between x 13 and 16 (no bridging)

    def test_mask_connected_but_graph_fragmented(self):
        out = self.run_subject(graph_gap_edge=True, mask_gap=False)['populations']['r2']
        np.testing.assert_array_equal(out['mask'], [True, True])
        np.testing.assert_array_equal(out['graph'], [False, False])   # edge 7-8 (x 9-10) missing, no repair

    def test_radius_is_minus_sdf_and_graph_secondary(self):
        result = self.run_subject(graph_gap_edge=False, mask_gap=False)
        node_r = result['populations']['r2']['node_radius']
        np.testing.assert_allclose(node_r[:, 0], 2.0, atol=0.05)      # -SDF on the tube axis
        np.testing.assert_allclose(node_r[:, 1], 1.5, atol=1e-6)      # graph radius
        self.assertEqual(result['selection']['graph_selected'], result['selection']['points'])
        self.assertAlmostEqual(result['scale'], 1.0)

    def test_stats_and_scores(self):
        both = np.array([[True, True], [True, False], [False, False]])
        g = np.array([[True, False], [False, False], [False, False]]); m = np.array([[True, True], [True, False], [False, False]])
        s = connectivity_stats(both, g, m)
        np.testing.assert_allclose(s['p_both'], [2 / 3, 1 / 3])
        np.testing.assert_allclose(s['graph_p_conn_given_both'], [.5, 0.])
        np.testing.assert_allclose(s['mask_not_graph_given_both'], [.5, 1.])
        counts = {'both': both.sum(0), 'graph': g.sum(0), 'mask': m.sum(0)}
        score = score_connectivity({'both': np.array([True, True]), 'graph': np.array([True, False]), 'mask': np.array([True, True])}, counts, 3)
        self.assertEqual(score['mask']['edges_connected'], 2)
        errors = radius_errors(np.array([2.0]), np.array([[1.0], [2.0], [3.0], [np.nan]]))
        self.assertEqual(errors['median_abs_error_mm'], 0.); self.assertAlmostEqual(errors['median_percentile'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
