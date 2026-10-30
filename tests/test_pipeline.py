import pickle
import tempfile
import unittest
from pathlib import Path
import networkx as nx
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import yaml
from vatl_sdf import CHANNELS
from vatl_sdf.config import load_config
from vatl_sdf.dataset import SDFGraphDataset,read_metadata,load_subject,sample_sdf,collate_subjects,voxel_coordinates
from vatl_sdf.model import SDFModel,rigid_transform
from vatl_sdf.losses import normalized_sdf,sdf_losses
from vatl_sdf.inference import world_grid,export_fields,predict_field,generate_atlas,native_grid
from vatl_sdf.evaluate import evaluate,reference_relative_connectivity
from vatl_sdf.graph_coords import voxel_to_world

ROOT = Path(__file__).resolve().parents[1]
# Oblique, anisotropic affine like the real data (0.9375 x 0.9375 x 0.8 mm, ~7 degrees).
ANGLE = np.deg2rad(7)
AFFINE = np.eye(4)
AFFINE[:3,:3] = np.array([[1,0,0],[0,np.cos(ANGLE),-np.sin(ANGLE)],[0,np.sin(ANGLE),np.cos(ANGLE)]])@np.diag([.9375,.9375,.8])
AFFINE[:3,3] = [-3.7,-3.2,-3.4]


def tiny_config(directory):
    c=load_config(ROOT/'config.yaml')
    c['device']='cpu';c['output_root']=str(Path(directory)/'runs');c['data']['tsv']=str(Path(directory)/'metadata.tsv')
    c['data']['world_bbox_mm']=[8,8,8];c['data']['num_workers']=0
    c['sampling']['sdf']['points']=24
    c['model'].update(latent_shape=[4,2,2,2],hidden_size=16,hidden_layers=3,modulated_layers=[1,3],num_experts=4,moe_k=2)
    c['training'].update(epochs=2,subjects_per_rank=1,task_points_per_step=16,checkpoint_every=1,atlas_every=0,max_steps=1,
                         stop_after_epoch=None)
    c['atlas'].update(spacing_mm=[2,2,2],chunk_size=16)
    rows=[]
    centre=np.array([4,4,4])
    world=voxel_to_world(AFFINE,np.indices((8,8,8)).reshape(3,-1).T)
    sdf=(np.linalg.norm(world-voxel_to_world(AFFINE,centre[None]),axis=1)-1.5).reshape(8,8,8).astype(np.float32)
    for i in range(2):
        image=nib.Nifti1Image(sdf,AFFINE);image.header.set_xyzt_units('mm');nib.save(image,Path(directory)/f'SDF{i}.nii.gz')
        graph=nx.Graph()
        for n,v in enumerate(([4,4,4],[5,4,4],[4,5,4])):
            graph.add_node(n,pos=voxel_to_world(AFFINE,np.array(v,float)),radius=np.float32(1.5))
        graph.add_edges_from([(0,1),(0,2)],weight=1.)
        with open(Path(directory)/f'Graph{i}.pkl','wb') as f:pickle.dump({'graph':graph},f)
        rows.append({'subject_id':f's{i}','scan_age':20+i,'sex':'M' if i==0 else 'F','MRA':'unused_never_read.nii.gz',
                     'SDF':str(Path(directory)/f'SDF{i}.nii.gz'),'Graph':str(Path(directory)/f'Graph{i}.pkl')})
    pd.DataFrame(rows).to_csv(c['data']['tsv'],sep='\t',index=False)
    split_path=Path(directory)/'splits.yaml'
    split_path.write_text(yaml.safe_dump({'train':['s0','s1'],'val':[],'test':[]}))
    c['data'].update(subject_ids=str(split_path),split_key=None,split='train',
                     expected_split_counts={'train':2,'val':0,'test':0})
    return c


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.c=tiny_config(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_metadata_uses_only_sdf_graph_age_sex(self):
        data=SDFGraphDataset(self.c);self.assertEqual(len(data),2)
        self.assertEqual(set(data.frame.columns),{'subject_id','SDF','Graph','scan_age','sex'})
        item=data[0]
        self.assertEqual(set(item),{'sdf_coords','sdf_target_mm','sampling_report','condition','subject_index'})

    def test_graph_and_sdf_share_one_coordinate_system(self):
        subject=load_subject(read_metadata(self.c).iloc[0],self.c)
        # Graph nodes were placed at voxel centres (4,4,4),(5,4,4),(4,5,4): their normalized
        # coordinates must equal those of the same voxels in SDF sampling and the native grid.
        flat=np.ravel_multi_index(np.array([[4,5,4],[4,4,5],[4,4,4]]),(8,8,8))
        np.testing.assert_allclose(subject['graph']['coords'],voxel_coordinates(flat,(8,8,8),AFFINE,self.c),atol=1e-6)
        coords,_,_=native_grid(nib.load(read_metadata(self.c).iloc[0].SDF),self.c)
        np.testing.assert_allclose(subject['graph']['coords'],coords[flat],atol=1e-6)
        self.assertLess(subject['sdf'].ravel()[flat].max(),0)  # nodes lie inside the vessel

    def test_sampling_pools(self):
        subject=load_subject(read_metadata(self.c).iloc[0],self.c)
        self.c['sampling']['sdf']['fractions']={'surface':0.,'interior':1.,'outside':0.}
        sample=sample_sdf(subject,self.c,np.random.default_rng(5))
        self.assertTrue((sample['sdf_target_mm']<0).all())
        self.assertEqual(sample['sdf_coords'].shape,(24,3))
        batch=collate_subjects([dict(sample,condition=np.zeros(2,np.float32),subject_index=0)])
        self.assertEqual(tuple(batch['sdf_coords'].shape),(1,24,3))

    def test_single_channel_output_and_loss_routing(self):
        model=SDFModel(self.c,2)
        self.assertFalse(any('mra' in name for name,_ in model.named_parameters()))
        output,moe=model(torch.randn(7,3)*.1,torch.tensor([0,0,0,1,1,1,1]),torch.zeros(7,2))
        self.assertEqual(output.shape,(7,1));self.assertEqual(CHANNELS,{'vessel_sdf_normalized':0})
        parts=sdf_losses(output,torch.ones(7,1),moe,model.transforms,self.c)
        self.assertEqual(set(parts),{'sdf_raw','moe_raw','transform_raw','sdf_weighted','moe_weighted','transform_weighted','total'})
        torch.testing.assert_close(parts['total'],self.c['loss']['sdf_weight']*parts['sdf_raw']+self.c['loss']['moe_weight']*moe)
        parts['total'].backward()
        self.assertGreater(model.latents.grad[:2].abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,'disagree'):sdf_losses(output,torch.ones(6,1),moe,None,self.c)

    def test_sdf_sign_units_and_export(self):
        np.testing.assert_allclose(normalized_sdf(torch.tensor([-9.,-1.,0.,1.,9.]),5).numpy(),[-1,-.2,0,.2,1])
        output=np.array([[-.2],[0],[.4],[-2]],dtype=np.float32)
        directory=Path(self.temp.name)/'export'
        affine=np.diag([2,3,4,1])
        export_fields(output,(2,2,1),affine,directory,5)
        self.assertEqual(sorted(p.name for p in directory.iterdir()),['VesselMask_atlas.nii.gz','VesselSDF_atlas.nii.gz'])
        np.testing.assert_allclose(nib.load(directory/'VesselSDF_atlas.nii.gz').get_fdata().ravel(),output[:,0]*5)
        np.testing.assert_array_equal(nib.load(directory/'VesselMask_atlas.nii.gz').get_fdata().ravel(),[1,0,0,1])
        with self.assertRaisesRegex(ValueError,'one SDF output channel'):
            export_fields(np.zeros((4,2),np.float32),(2,2,1),affine,Path(self.temp.name)/'bad',5)

    def test_grid_and_physical_rigid(self):
        coords,shape,affine=world_grid(self.c)
        self.assertEqual(shape,(4,4,4));np.testing.assert_allclose(affine[:3,3],[-4,-4,-4])
        np.testing.assert_allclose(coords[0],[-1,-1,-1])
        result=rigid_transform(torch.tensor([[.5,0.,0.]]),torch.tensor([[0.,0.,np.pi/2,0.,0.,0.]]),torch.tensor([4.,2.,1.]))
        np.testing.assert_allclose(result.numpy(),[[0,1,0]],atol=1e-6)

    def test_optimization_decreases_loss(self):
        torch.manual_seed(1);model=SDFModel(self.c,2)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.rand(16,3)-.5;ids=torch.zeros(16,dtype=torch.long);cv=torch.zeros(16,2)
        values=[]
        for _ in range(10):
            optimizer.zero_grad();out,moe=model(x,ids,cv)
            loss=sdf_losses(out,-torch.ones(16,1),moe,model.transforms,self.c)['total']
            values.append(loss.item());loss.backward();optimizer.step()
        self.assertLess(values[-1],values[0])
        self.assertEqual(predict_field(model,np.zeros((1,3),np.float32),[0,0],self.c,subject_index=0).shape,(1,1))

    def test_training_step_checkpoint_and_exact_resume(self):
        from vatl_sdf.train import train
        first=train(self.c)
        checkpoint=next(Path(first).glob('checkpoint_step_*.pt'))
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.assertEqual(state['step'],1);self.assertNotIn('mra',' '.join(state['model']))
        losses=[__import__('json').loads(l) for l in (Path(first)/'losses.jsonl').read_text().splitlines()]
        self.assertTrue(np.isfinite(losses[0]['total']))
        resumed=dict(self.c,resume=str(checkpoint),require_fresh_start=False,training=dict(self.c['training'],max_steps=3))
        resumed_dir=train(resumed)
        straight=train(dict(self.c,training=dict(self.c['training'],max_steps=3)))
        a=torch.load(Path(resumed_dir)/'checkpoint_step_00000003.pt',map_location='cpu',weights_only=False)['model']
        b=torch.load(Path(straight)/'checkpoint_step_00000003.pt',map_location='cpu',weights_only=False)['model']
        self.assertLess(max((a[k]-b[k]).abs().max().item() for k in a),1e-6)

    def test_fresh_start_guard(self):
        from vatl_sdf.train import train
        with self.assertRaisesRegex(ValueError,'fresh start'):
            train(dict(self.c,require_fresh_start=True,resume='/any/checkpoint.pt'))

    def test_atlas_and_average_levelset_export(self):
        self.c['atlas']['spacing_mm']=[1,1,1]
        model=SDFModel(self.c,2)
        frame=read_metadata(self.c)
        root=Path(self.temp.name)/'atlas'
        generate_atlas(model,frame,self.c,root,age=20,sex='M',average_fields=True)
        image=nib.load(root/'VesselSDF_atlas.nii.gz')
        self.assertEqual(image.shape,(8,8,8))
        np.testing.assert_allclose(image.affine,np.array([[1,0,0,-4],[0,1,0,-4],[0,0,1,-4],[0,0,0,1]]))
        self.assertTrue((root/'AveragedLevelSet_mm_NOT_exact_SDF.nii.gz').exists())
        np.testing.assert_array_equal(nib.load(root/'VesselMask_atlas.nii.gz').get_fdata(),image.get_fdata()<0)

    def test_offline_vessel_distance_construction(self):
        from vatl_sdf.preprocess_sdf import signed_distance
        mask=np.zeros((8,8,8),dtype=bool);mask[3:5,3:5,3:5]=True
        sdf=signed_distance(mask,np.diag([2.,2.,2.,1.]))
        np.testing.assert_array_equal(sdf<0,mask)
        self.assertEqual(sdf[3,3,3],-2.);self.assertEqual(sdf[2,3,3],2.)

    def test_evaluation_and_reference_relative_connectivity(self):
        sdf=np.ones((8,8,8));sdf[3:5,3:5,3:5]=-1
        result=evaluate(sdf,sdf,np.eye(4),5,exported_mask=sdf<0)
        self.assertEqual(result['sdf']['zero_level_dice_vs_reference_sdf'],1.)
        self.assertEqual(result['sdf']['export_zero_level_mismatch_voxels'],0)
        self.assertEqual(result['sdf']['components']['vessel_volume_mm3'],8.)
        self.assertEqual(result['sdf']['reference_relative_connectivity']['component_ratio'],1.)
        # Reference: two pieces (largest 2/3). Prediction: same two plus a cut in the large one.
        reference=np.zeros((12,3,3),bool);reference[0:8,1,1]=True;reference[10:12,1,1]=True
        pred=reference.copy();pred[4,1,1]=False
        rel=reference_relative_connectivity(pred,reference,min_voxels=1)
        self.assertEqual((rel['reference_components'],rel['pred_components'],rel['component_excess']),(2,3,1))
        self.assertAlmostEqual(rel['reference_largest_fraction'],0.8)
        self.assertAlmostEqual(rel['largest_fraction_difference'],4/9-0.8)
        # Diagonal contact is connected under 26-connectivity.
        diag=np.zeros((3,3,3),bool);diag[0,0,0]=diag[1,1,1]=True
        self.assertEqual(reference_relative_connectivity(diag,diag,1)['pred_components'],1)

if __name__=='__main__':unittest.main()
