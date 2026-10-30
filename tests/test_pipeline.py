import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from mra_sdf_only.config import load_config,condition_vector
from mra_sdf_only.dataset import MRASDFDataset,read_metadata,load_pair,sample_tasks,collate_subjects,normalize_mra
from mra_sdf_only.model import MRASDFModel,rigid_transform
from mra_sdf_only.losses import normalized_sdf,task_losses
from mra_sdf_only.inference import world_grid,export_fields,predict_field,generate_atlas
from mra_sdf_only.evaluate import evaluate


def tiny_config(directory):
    c=load_config(Path(__file__).resolve().parents[1]/'config.yaml')
    c['device']='cpu';c['output_root']=str(Path(directory)/'runs');c['data']['tsv']=str(Path(directory)/'metadata.tsv')
    c['data']['world_bbox_mm']=[8,8,8];c['data']['num_workers']=0
    c['sampling']['mra']['points']=32;c['sampling']['sdf']['points']=24
    c['model'].update(latent_shape=[4,2,2,2],hidden_size=16,hidden_layers=3,modulated_layers=[1,3],num_experts=4,moe_k=2)
    c['training'].update(epochs=2,subjects_per_rank=1,task_points_per_step=16,checkpoint_every=1,atlas_every=0,max_steps=1)
    c['atlas'].update(spacing_mm=[2,2,2],chunk_size=16)
    rows=[]
    xyz=np.indices((8,8,8));sdf=np.sqrt(((xyz-np.array([4,4,4])[:,None,None,None])**2).sum(0)).astype(np.float32)-1.5
    for i in range(2):
        mra=np.zeros((8,8,8),dtype=np.float32);mra[1:7,1:7,1:7]=.2;mra[sdf<0]=.9
        for key,value in [('MRA',mra),('SDF',sdf)]:
            image=nib.Nifti1Image(value,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,Path(directory)/f'{key}{i}.nii.gz')
        rows.append({'subject_id':f's{i}','scan_age':20+i,'sex':'M' if i==0 else 'F',
                     'MRA':str(Path(directory)/f'MRA{i}.nii.gz'),'SDF':str(Path(directory)/f'SDF{i}.nii.gz')})
    pd.DataFrame(rows).to_csv(c['data']['tsv'],sep='\t',index=False)
    import yaml
    split_path=Path(directory)/'splits.yaml'
    split_path.write_text(yaml.safe_dump({'train':['s0','s1'],'val':[],'test':[]}))
    c['data'].update(subject_ids=str(split_path),split_key=None,split='train',
                     expected_split_counts={'train':2,'val':0,'test':0})
    return c


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.c=tiny_config(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_minimal_metadata_and_affine_validation(self):
        data=MRASDFDataset(self.c);self.assertEqual(len(data),2)
        self.assertEqual(set(data.frame.columns),{'subject_id','MRA','SDF','scan_age','sex'})
        row=data.frame.iloc[0];image=nib.load(row.SDF)
        bad=image.affine.copy();bad[0,3]=1
        nib.save(nib.Nifti1Image(image.get_fdata(),bad),row.SDF)
        with self.assertRaisesRegex(ValueError,'affines'):load_pair(row,self.c)

    def test_independent_samples(self):
        pair=load_pair(read_metadata(self.c).iloc[0],self.c)
        first=sample_tasks(pair,self.c,np.random.default_rng(5))
        self.c['sampling']['sdf']['fractions']={'surface':0.,'interior':1.,'outside':0.}
        second=sample_tasks(pair,self.c,np.random.default_rng(5))
        np.testing.assert_array_equal(first['mra_coords'],second['mra_coords'])
        np.testing.assert_array_equal(first['mra_target'],second['mra_target'])
        self.assertTrue((second['sdf_target_mm']<0).all())
        self.assertEqual(len(first['mra_coords']),32);self.assertEqual(len(first['sdf_coords']),24)
        self.assertLessEqual(second['sampling_report']['sdf']['unique_voxels'],24)

    def test_two_heads_and_loss_routing(self):
        model=MRASDFModel(self.c,2)
        output,moe=model(torch.randn(7,3)*.1,torch.tensor([0,0,0,1,1,1,1]),torch.zeros(7,2))
        self.assertEqual(output.shape,(7,2))
        output.retain_grad()
        parts=task_losses(output,3,torch.zeros(3,1),torch.ones(4,1),moe,model.transforms,self.c)
        parts['mra_raw'].backward(retain_graph=True)
        self.assertEqual(output.grad[3:].abs().sum().item(),0.)
        self.assertEqual(output.grad[:3,1].abs().sum().item(),0.)
        output.grad.zero_();parts['sdf_raw'].backward()
        self.assertEqual(output.grad[:3].abs().sum().item(),0.)
        self.assertEqual(output.grad[3:,0].abs().sum().item(),0.)
        self.assertTrue(model.latents.grad is not None)

    def test_sdf_sign_units_and_output_mapping(self):
        np.testing.assert_allclose(normalized_sdf(torch.tensor([-9.,-1.,0.,1.,9.]),5).numpy(),[-1,-.2,0,.2,1])
        output=np.array([[.7,-.2],[.6,0],[.5,.4],[.4,-2]],dtype=np.float32)
        directory=Path(self.temp.name)/'export'
        affine=np.diag([2,3,4,1])
        export_fields(output,(2,2,1),affine,directory,5)
        np.testing.assert_allclose(nib.load(directory/'MRA_atlas.nii.gz').get_fdata().ravel(),output[:,0])
        np.testing.assert_allclose(nib.load(directory/'VesselSDF_atlas.nii.gz').get_fdata().ravel(),output[:,1]*5)
        np.testing.assert_array_equal(nib.load(directory/'VesselMask_atlas.nii.gz').get_fdata().ravel(),[1,0,0,1])
        np.testing.assert_allclose(nib.load(directory/'VesselSDF_atlas.nii.gz').affine,affine)

    def test_sdf_only_has_no_intensity_supervision(self):
        self.c['loss']['mra_weight']=0.
        output=torch.zeros(7,2,requires_grad=True)
        parts=task_losses(output,3,torch.ones(3,1),-torch.ones(4,1),
                          output.sum()*0.,None,self.c)
        changed=task_losses(output,3,torch.ones(3,1)*100,-torch.ones(4,1),
                            output.sum()*0.,None,self.c)
        self.assertEqual(parts['total'].item(),changed['total'].item())
        parts['total'].backward()
        self.assertEqual(output.grad[:,0].abs().sum().item(),0.)
        self.assertEqual(output.grad[:3,1].abs().sum().item(),0.)
        self.assertGreater(output.grad[3:,1].abs().sum().item(),0.)

    def test_grid_and_physical_rigid(self):
        coords,shape,affine=world_grid(self.c)
        self.assertEqual(shape,(4,4,4));np.testing.assert_allclose(affine[:3,3],[-4,-4,-4])
        np.testing.assert_allclose(coords[0],[-1,-1,-1])
        x=torch.tensor([[.5,0.,0.]])
        tf=torch.tensor([[0.,0.,np.pi/2,0.,0.,0.]])
        result=rigid_transform(x,tf,torch.tensor([4.,2.,1.]))
        np.testing.assert_allclose(result.numpy(),[[0,1,0]],atol=1e-6)

    def test_numeric_optimization_and_inference(self):
        torch.manual_seed(1);model=MRASDFModel(self.c,2)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.zeros(16,3);ids=torch.zeros(16,dtype=torch.long);cv=torch.zeros(16,2)
        values=[]
        for _ in range(10):
            optimizer.zero_grad();out,moe=model(x,ids,cv)
            loss=task_losses(out,8,torch.ones(8,1)*.3,torch.ones(8,1)*-1,moe,model.transforms,self.c)['total']
            values.append(loss.item());loss.backward();optimizer.step()
        self.assertLess(values[-1],values[0])
        output=predict_field(model,np.zeros((1,3),np.float32),[0,0],self.c,subject_index=0)
        self.assertEqual(output.shape,(1,2))

    def test_atlas_and_average_levelset_export(self):
        self.c['atlas']['spacing_mm']=[1,1,1]
        model=MRASDFModel(self.c,2)
        frame=read_metadata(self.c)
        root=Path(self.temp.name)/'atlas'
        generate_atlas(model,frame,self.c,root,age=20,sex='M',average_fields=True)
        image=nib.load(root/'MRA_atlas.nii.gz')
        self.assertEqual(image.shape,(8,8,8))
        np.testing.assert_allclose(image.affine,np.array([[1,0,0,-4],[0,1,0,-4],[0,0,1,-4],[0,0,0,1]]))
        self.assertTrue((root/'AveragedLevelSet_mm_NOT_exact_SDF.nii.gz').exists())
        sdf=nib.load(root/'VesselSDF_atlas.nii.gz').get_fdata()
        mask=nib.load(root/'VesselMask_atlas.nii.gz').get_fdata()
        np.testing.assert_array_equal(mask,sdf<0)

    def test_intensity_normalization_without_anatomical_fields(self):
        image=np.array([0.,.1,1.],dtype=np.float32)
        normalized,scale=normalize_mra(image,self.c)
        np.testing.assert_array_equal(normalized,image)
        self.assertEqual(scale,1.)
        sparse=np.zeros(10000,dtype=np.float32);sparse[-1]=100
        normalized,scale=normalize_mra(sparse,self.c)
        self.assertEqual(scale,100.)
        self.assertEqual(normalized[-1],1.)

    def test_offline_vessel_distance_construction(self):
        from mra_sdf_only.preprocess_sdf import signed_distance
        mask=np.zeros((8,8,8),dtype=bool);mask[3:5,3:5,3:5]=True
        sdf=signed_distance(mask,np.diag([2.,2.,2.,1.]))
        np.testing.assert_array_equal(sdf<0,mask)
        self.assertEqual(sdf[3,3,3],-2.)
        self.assertEqual(sdf[2,3,3],2.)

    def test_evaluation(self):
        mra=np.zeros((8,8,8));sdf=np.ones_like(mra);sdf[3:5,3:5,3:5]=-1
        result=evaluate(mra,sdf,mra,sdf,np.eye(4),5,exported_mask=sdf<0)
        self.assertEqual(result['sdf']['zero_level_dice_vs_reference_sdf'],1.)
        self.assertEqual(result['sdf']['export_zero_level_mismatch_voxels'],0)
        self.assertEqual(result['sdf']['components']['vessel_volume_mm3'],8.)

if __name__=='__main__':unittest.main()
