"""Split integrity tests: actual authoritative cohort plus synthetic failure cases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from vatl_sdf.config import load_config
from vatl_sdf.dataset import read_metadata, SDFGraphDataset
from vatl_sdf.splits import read_splits, ordered_rows, check_resume_subjects
from vatl_sdf.test_individual import select_subjects
from vatl_sdf.train import train
from tests.test_pipeline import tiny_config

SPLIT = Path('/medias/medical/data/Atlas_training/v1/subject_ids_label_t1_fast.yaml')
KEY = 'mra_atlas_synthseg_label_t1_fast.subject_ids'
BASE = Path('configs/sdf/00_sdf_baseline.yaml')
LEGACY = Path('configs/ablations/legacy_mra/02_sdf_only.yaml')


class SplitTests(unittest.TestCase):
    def test_actual_497_55_membership_and_yaml_order(self):
        self.assertTrue(SPLIT.is_file(), 'Authoritative split must be accessible for this project integration check')
        config = load_config(BASE)
        splits = read_splits(SPLIT, KEY, {'train':497,'val':55,'test':0})
        frame = read_metadata(config)
        self.assertEqual(len(frame),497)
        self.assertEqual(frame.subject_id.tolist(),splits['train'])
        self.assertFalse(set(frame.subject_id) & set(splits['val']))
        self.assertFalse(set(splits['train']) & set(splits['val']))
        self.assertFalse(any('mra' in key for section in config.values() if isinstance(section,dict) for key in section))
        self.assertIsNone(config['resume']);self.assertTrue(config['require_fresh_start'])
        selected3 = select_subjects(config['data']['tsv'],frame,3,SPLIT,KEY+'.val')
        selected5 = select_subjects(config['data']['tsv'],frame,5,SPLIT,KEY+'.val')
        self.assertEqual(selected3.subject_id.tolist(),splits['val'][:3])
        self.assertEqual(selected5.subject_id.tolist(),splits['val'][:5])
        self.assertEqual(selected5.subject_id.tolist()[:3],selected3.subject_id.tolist())
        contaminated = pd.read_csv(config['data']['tsv'],sep='\t')
        with self.assertRaisesRegex(ValueError,'match declared population train'):
            select_subjects(config['data']['tsv'],contaminated,3,SPLIT,KEY+'.val')
        with self.assertRaisesRegex(ValueError,'contaminated or incompatible'):
            check_resume_subjects({'metadata':contaminated.to_dict(orient='records')},frame)
        with self.assertRaisesRegex(ValueError,'outside the declared'):
            SDFGraphDataset(config,frame=selected3)

    def test_overlap_duplicate_wrong_count_and_missing_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'splits.yaml'
            path.write_text(yaml.safe_dump({'train':['a'],'val':['a'],'test':[]}))
            with self.assertRaisesRegex(ValueError,'overlap'):read_splits(path,None)
            path.write_text(yaml.safe_dump({'train':['a','a'],'val':[],'test':[]}))
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_splits(path,None)
            path.write_text(yaml.safe_dump({'train':['a'],'val':['b'],'test':[]}))
            with self.assertRaisesRegex(ValueError,'expected counts'):read_splits(path,None,{'train':497,'val':55,'test':0})
            with self.assertRaisesRegex(ValueError,'missing from metadata'):
                ordered_rows(pd.DataFrame({'subject_id':['b']}),['a'])

    def test_null_and_wrong_training_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=tiny_config(tmp)
            config['data']['subject_ids']=None
            with self.assertRaisesRegex(ValueError,'Explicit subject split'):read_metadata(config)
            config=tiny_config(tmp);config['data']['split']='val'
            with self.assertRaisesRegex(ValueError,'split=train'):read_metadata(config)
            config=tiny_config(tmp);config['data']['expected_split_counts']={'train':552,'val':0,'test':0}
            with self.assertRaisesRegex(ValueError,'expected counts'):read_metadata(config)

    def test_fresh_start_blocks_resume_before_loading_weights(self):
        config=load_config(BASE);config['resume']='/any/contaminated/checkpoint.pt'
        with patch('vatl_sdf.train.torch.load') as load:
            with self.assertRaisesRegex(ValueError,'fresh start'):train(config)
            load.assert_not_called()

    def test_baseline_differs_from_legacy_sdf_only_only_as_documented(self):
        old=yaml.safe_load(LEGACY.read_text());new=load_config(BASE)
        for section in ('model','sdf','optimizer','atlas','seed','device','require_fresh_start'):
            old_section=old[section]
            if section=='model':old_section={k:v for k,v in old_section.items() if k!='mra_activation'}
            self.assertEqual(old_section,new[section],section)
        self.assertEqual(old['sampling']['sdf'],new['sampling']['sdf']);self.assertEqual(set(new['sampling']),{'sdf'})
        self.assertEqual({k:v for k,v in old['training'].items() if k!='gradient_diagnostics_every'},new['training'])
        self.assertEqual({k:v for k,v in old['data'].items() if not k.startswith('intensity')},new['data'])
        self.assertEqual((old['loss']['sdf_weight'],new['loss']['sdf_weight']),(0.1,1.0))  # documented change
        for key in ('moe_weight','tf_weight','sdf_metric'):self.assertEqual(old['loss'][key],new['loss'][key])


if __name__=='__main__':unittest.main()
