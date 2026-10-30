"""Phase 0 diagnostic helpers: supersampled grid geometry and subject selection."""
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

from analysis.phase0_graph.diagnose_checkpoint import normalized, select_subjects, subject_grid
from mra_sdf_only.inference import native_grid

AFFINE = np.array([[0.9, 0.01, 0, -100], [0, 0.93, 0.1, -90], [0, 0, 0.8, 5], [0, 0, 0, 1.]])
SHAPE = (6, 5, 4)
CONFIG = {'data': {'world_bbox_mm': [240, 240, 120]}}


class Phase0DiagnosticsTest(unittest.TestCase):
    def test_factor_one_matches_native_grid(self):
        _, world, shape, affine = subject_grid(AFFINE, SHAPE, 1)
        image = nib.Nifti1Image(np.zeros(SHAPE, np.float32), AFFINE)
        self.assertEqual(shape, SHAPE)
        np.testing.assert_allclose(affine, AFFINE)
        np.testing.assert_allclose(normalized(world, AFFINE, SHAPE, CONFIG), native_grid(image, CONFIG)[0], atol=1e-6)

    def test_supersampled_grid_shares_fov_and_affine(self):
        _, world, shape, affine = subject_grid(AFFINE, SHAPE, 2)
        self.assertEqual(shape, (12, 10, 8))
        np.testing.assert_allclose(nib.affines.apply_affine(affine, np.indices(shape).reshape(3, -1).T), world, atol=1e-9)
        centre = nib.affines.apply_affine(AFFINE, (np.asarray(SHAPE) - 1) / 2)
        np.testing.assert_allclose(world.mean(0), centre, atol=1e-9)

    def test_subject_selection_spans_age_and_sex(self):
        frame = pd.DataFrame({'subject_id': list('abcdefgh'), 'sex': list('MMMMFFFF'),
                              'scan_age': [20, 30, 50, 80, 25, 45, 60, 85]})
        chosen = select_subjects(frame)
        self.assertEqual(chosen, ['e', 'f', 'h', 'a', 'b', 'd'])
        self.assertEqual(set(frame.set_index('subject_id').loc[chosen].sex), {'M', 'F'})


if __name__ == '__main__':
    unittest.main()
