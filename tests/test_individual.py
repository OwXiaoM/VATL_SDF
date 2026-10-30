"""Small CPU tests for cohort isolation, adaptation isolation, coordinates and scoring."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

from mra_sdf_only.config import load_config, condition_vector
from mra_sdf_only.individual_metrics import component_metrics, evaluate_sdf, summarize
from mra_sdf_only.model import MRASDFModel, rigid_transform
from mra_sdf_only.test_individual import (select_subjects, observation_partition, fit_subject,
    forward_sdf, parameter_digest, load_reference, infer_native, save_volume, run)


class IndividualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.config = load_config('config.yaml')
        self.config['loss']['mra_weight'] = 0.
        self.config['data']['world_bbox_mm'] = [12, 12, 12]
        self.config['model'].update(latent_shape=[4, 2, 2, 2], hidden_size=12, hidden_layers=3,
                                    modulated_layers=[1, 3], use_moe=True)
        self.settings = yaml.safe_load(Path('configs/evaluation/sdf_only.yaml').read_text())
        self.settings.update(steps=2, batch_size=16, chunk_size=256, num_observations=80,
                             block_size_voxels=4)
        self.training = pd.DataFrame([{'subject_id':'train', 'scan_age':30., 'sex':'M', 'SDF':'/not/read/train.nii.gz'}])
        # Deliberately unsorted metadata; split-list order differs too.
        self.ids = ['test4', 'test1', 'test5', 'test2', 'test3']
        sdf = np.linalg.norm(np.indices((12, 12, 12)).transpose(1, 2, 3, 0)-[6, 6, 6], axis=-1).astype(np.float32)-2
        rows = []
        for i, ident in enumerate(self.ids[::-1]):
            image = nib.Nifti1Image(sdf, np.eye(4)); image.header.set_xyzt_units('mm')
            path = self.root/(ident+'.nii.gz'); nib.save(image, path)
            rows.append({'subject_id':ident, 'scan_age':30.+i, 'sex':'M', 'SDF':str(path)})
        self.metadata = self.root/'metadata.tsv'; pd.DataFrame(rows).to_csv(self.metadata, sep='\t', index=False)
        self.split = self.root/'split.yaml'; self.split.write_text(yaml.safe_dump({'cohort':{'train':['train'],'val':self.ids,'test':[]}}))
        torch.manual_seed(9); self.model = MRASDFModel(self.config, 1).eval()
        for p in self.model.parameters():p.requires_grad_(False)

    def tearDown(self):
        self.temp.cleanup()

    def test_cohort_prefix_contamination_and_missing_split(self):
        a = select_subjects(self.metadata, self.training, 3, self.split, 'cohort.val')
        b = select_subjects(self.metadata, self.training, 5, self.split, 'cohort.val')
        self.assertEqual(a.subject_id.tolist(), self.ids[:3]); self.assertEqual(b.subject_id.tolist(), self.ids)
        self.split.write_text(yaml.safe_dump({'cohort':{'train':['train'],'val':[],'test':[]}}))
        with self.assertRaisesRegex(ValueError, 'empty'):select_subjects(self.metadata, self.training, 3, self.split, 'cohort.val')
        self.split.write_text(yaml.safe_dump({'cohort':{'train':['train'],'val':self.ids+['train'],'test':[]}}))
        with self.assertRaisesRegex(ValueError, 'overlap'):select_subjects(self.metadata, self.training, 3, self.split, 'cohort.val')
        with self.assertRaisesRegex(ValueError, 'explicit test split'):select_subjects(self.metadata, self.training, 3)

    def test_native_coordinates_partition_and_external_rigid(self):
        row = pd.read_csv(self.metadata, sep='\t').iloc[0]
        image, sdf, coords, valid = load_reference(row, self.config)
        np.testing.assert_array_equal(coords[0], [-1, -1, -1])
        obs, adapt, ev = observation_partition(image.shape, valid, self.settings, 123)
        obs2, adapt2, ev2 = observation_partition(image.shape, valid, self.settings, 123)
        np.testing.assert_array_equal(obs, obs2); np.testing.assert_array_equal(ev, ev2)
        self.assertFalse(ev.ravel()[obs].any()); self.assertFalse((adapt & ev).any())
        points = torch.tensor([[.1, .2, .3], [-.3, .1, -.4]])
        tf = torch.tensor([[.01, -.02, .03, .2, -.3, .1]])
        cond = condition_vector(self.config, 30, 'M'); latent = self.model.latents[:1].detach()
        result = forward_sdf(self.model, points, cond, latent, tf)
        mapped = rigid_transform(points, tf.expand(2,-1), self.model.half_bbox)
        expected = self.model(mapped, torch.zeros(2,dtype=torch.long), torch.tensor(cond).expand(2,-1),
                              latent_override=latent, apply_transform=False)[0][:,1]
        torch.testing.assert_close(result, expected)

    def test_only_external_parameters_change(self):
        before = parameter_digest(self.model)
        initial = self.model.latents[:1].detach().clone()
        cond = condition_vector(self.config, 30, 'M')
        coords = np.random.default_rng(1).uniform(-.7,.7,(64,3)).astype(np.float32)
        target = np.full(64, -1., dtype=np.float32)
        fitted, tf, trace = fit_subject(self.model, initial, cond, coords, target, self.config, self.settings, 123)
        self.assertEqual(before, parameter_digest(self.model)); self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertGreater((fitted-initial).abs().sum().item(), 0)
        self.assertGreater(tf.abs().sum().item(), 0)
        again, tf2, _ = fit_subject(self.model, initial, cond, coords, target, self.config, self.settings, 123)
        torch.testing.assert_close(fitted, again, rtol=0, atol=0); torch.testing.assert_close(tf, tf2, rtol=0, atol=0)

    def test_metrics_empty_cases_adjacency_surfaces_and_physical_distance(self):
        valid = np.ones((12,12,12), bool); sdf = np.ones(valid.shape, np.float32)
        empty = evaluate_sdf(sdf, sdf, np.eye(4), valid, valid)
        self.assertEqual(empty['dice'], 1); self.assertIsNone(empty['assd_mm']); self.assertEqual(empty['empty_case'], 'both_empty')
        gt = sdf.copy(); gt[4,4,4] = -1
        pred = sdf.copy(); pred[5,4,4] = -1
        affine = np.diag([2.,1.,1.,1.])
        m = evaluate_sdf(pred, gt, affine, valid, valid, tolerance_mm=1)
        self.assertEqual(m['assd_mm'], 2); self.assertEqual(m['hd95_mm'], 2); self.assertEqual(m['surface_dice_tolerance'], 0)
        identical = evaluate_sdf(gt, gt, affine, valid, valid)
        self.assertEqual(identical['dice'], 1); self.assertEqual(identical['assd_mm'], 0)
        diagonal = np.zeros(valid.shape, bool); diagonal[4,4,4] = diagonal[5,5,5] = True
        cm = component_metrics(diagonal); self.assertEqual(cm['components_6'], 2); self.assertEqual(cm['components_26'], 1)
        # Excluded observation values cannot affect held-out voxel or surface metrics beyond the guard.
        evaluation = np.zeros_like(valid); evaluation[2:10,2:10,2:10] = True
        changed = pred.copy(); changed[0] = -1
        self.assertEqual(evaluate_sdf(pred, gt, affine, valid, evaluation), evaluate_sdf(changed, gt, affine, valid, evaluation))
        summary = summarize([{'x':1., 'undefined':None}, {'x':3., 'undefined':None}])
        self.assertEqual(summary['x']['mean'], 2); self.assertAlmostEqual(summary['x']['std'], np.sqrt(2)); self.assertEqual(summary['undefined']['n_defined'], 0)

    def test_partial_fov_native_exports(self):
        config = copy.deepcopy(self.config); config['data']['world_bbox_mm'] = [6,6,6]
        row = pd.read_csv(self.metadata, sep='\t').iloc[0]
        image, sdf, coords, valid = load_reference(row, config)
        self.assertTrue(valid.any()); self.assertFalse(valid.all())
        model = MRASDFModel(config, 1).eval()
        for p in model.parameters():p.requires_grad_(False)
        result = infer_native(model, coords, valid, condition_vector(config, 30, 'M'),
                              model.latents[:1].detach(), torch.zeros(1,6), config, self.settings)
        self.assertTrue(np.isnan(result[~valid]).all()); self.assertTrue(np.isfinite(result[valid]).all())
        binary = (result<0).astype(np.uint8)
        self.assertFalse(binary[~valid].any())
        save_volume(self.root/'roundtrip.nii.gz', result, image)
        saved = nib.load(self.root/'roundtrip.nii.gz')
        np.testing.assert_array_equal(saved.affine, image.affine)
        np.testing.assert_array_equal(saved.get_fdata(), result)

    def test_end_to_end_three_five_same_prefix(self):
        checkpoint = self.root/'checkpoint.pt'
        torch.save({'config':self.config, 'model':self.model.state_dict(), 'metadata':self.training.to_dict(orient='records')}, checkpoint)
        self.settings.update(checkpoint=str(checkpoint), metadata=str(self.metadata), split_yaml=str(self.split), split_key='cohort.val')
        settings = self.root/'settings.yaml'; settings.write_text(yaml.safe_dump(self.settings))
        for n in [5,3]:
            args = argparse.Namespace(settings=str(settings), checkpoint=None, metadata=None, split_yaml=None, split_key=None,
                split_column=None, num_test_subjects=n, output=str(self.root/f'run{n}'), device='cpu', determinism='strict',
                protocols=['zero_shot_atlas','test_time_sdf_fit'], cohort_manifest=str(self.root/'run5/selected_subjects.json') if n==3 else None, prepare_only=False)
            run(args)
        for ident in self.ids[:3]:
            for protocol in ['zero_shot_atlas','test_time_sdf_fit']:
                a = self.root/'run3'/ident/protocol; b = self.root/'run5'/ident/protocol
                self.assertEqual(json.loads((a/'metrics.json').read_text()),json.loads((b/'metrics.json').read_text()))
                np.testing.assert_array_equal(nib.load(a/'predicted_sdf_mm.nii.gz').get_fdata(), nib.load(b/'predicted_sdf_mm.nii.gz').get_fdata())
        r = json.loads((self.root/'run5/run.json').read_text()); self.assertTrue(r['checkpoint_unchanged'])
        self.assertEqual(r['frozen_state_sha256_before'], r['frozen_state_sha256_after'])
        self.assertEqual(json.loads((self.root/'run3/selected_subjects.json').read_text())['subject_ids'], self.ids[:3])


if __name__ == '__main__':
    unittest.main()
