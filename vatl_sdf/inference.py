"""SDF inference and export (SDF + its zero-level mask); no anatomical masks or component filtering."""
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from .config import condition_vector
from . import CHANNELS
from .graph_coords import normalize_world, voxel_to_world


def world_grid(config, spacing=None):
    bbox = np.asarray(config['data']['world_bbox_mm'],dtype=float)
    spacing = np.asarray(spacing if spacing is not None else config['atlas']['spacing_mm'],dtype=float)
    axes = [np.arange(0,b,s,dtype=np.float32) for b,s in zip(bbox,spacing)]
    mesh = np.meshgrid(*axes,indexing='ij')
    coords = np.stack(mesh,axis=-1).reshape(-1,3)/(bbox/2.)-1.
    affine = np.diag([*spacing,1.]); affine[:3,3] = -bbox/2.
    return coords.astype(np.float32),tuple(len(a) for a in axes),affine


def native_grid(image, config):
    shape = image.shape[:3]
    voxels = np.indices(shape).reshape(3,-1).T
    coords = normalize_world(voxel_to_world(image.affine,voxels),image.affine,shape,config['data']['world_bbox_mm'])
    return coords.astype(np.float32),shape,image.affine.copy()


@torch.no_grad()
def predict_field(model, coords, condition, config, subject_index=None, latent_override=None):
    model.eval(); device = next(model.parameters()).device
    outputs=[]; chunk=config['atlas']['chunk_size']
    for start in range(0,len(coords),chunk):
        points=torch.as_tensor(coords[start:start+chunk],device=device,dtype=torch.float32)
        ids=torch.full((len(points),),subject_index if subject_index is not None else 0,device=device,dtype=torch.long)
        conditions=torch.as_tensor(condition,device=device,dtype=torch.float32).reshape(1,2).expand(len(points),-1)
        output,_=model(points,ids,conditions,latent_override=latent_override,apply_transform=subject_index is not None)
        outputs.append(output.cpu().numpy())
    return np.concatenate(outputs,axis=0)


def export_fields(output, shape, affine, directory, tau, subject=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    if output.shape!=(int(np.prod(shape)),1):raise ValueError('Expected exactly one SDF output channel on the requested grid')
    sdf=output[:,CHANNELS['vessel_sdf_normalized']].reshape(shape)*tau
    if not np.isfinite(output).all():raise ValueError('Nonfinite inference outputs')
    names=('Subject_VesselSDF','Subject_VesselMask') if subject else ('VesselSDF_atlas','VesselMask_atlas')
    for name,data in zip(names,(sdf.astype(np.float32),(sdf<0).astype(np.uint8))):
        image=nib.Nifti1Image(data,affine);image.header.set_xyzt_units('mm');image.set_sform(affine,code=1)
        nib.save(image,directory/f'{name}.nii.gz')
    return directory


def weighted_latent(model, frame, config, age):
    # Same age-only Gaussian/top-N scheme as VATL_MM; sex conditions the decoder.
    ages=torch.as_tensor(frame['scan_age'].values,dtype=torch.float32,device=model.latents.device)
    sigma=config['atlas']['gaussian_span_years']/2.
    if sigma<=0:raise ValueError('Gaussian span must be positive')
    scores=-((ages-float(age))**2)/(2*sigma*sigma)
    keep=torch.argsort(scores,descending=True)[:config['atlas']['max_subjects']]
    weights=torch.zeros_like(scores);weights[keep]=torch.softmax(scores[keep],dim=0)
    mean=(model.latents*weights[:,None,None,None,None]).sum(0,keepdim=True)
    return mean,weights


def generate_atlas(model, frame, config, directory, age=None, sex=None, average_fields=False):
    age=config['atlas']['age'] if age is None else age;sex=config['atlas']['sex'] if sex is None else sex
    coords,shape,affine=world_grid(config)
    mean,weights=weighted_latent(model,frame,config,age)
    condition=condition_vector(config,age,sex)
    values=predict_field(model,coords,condition,config,latent_override=mean)
    directory=export_fields(values,shape,affine,directory,config['sdf']['truncation_mm'])
    import json
    (directory/'provenance.json').write_text(json.dumps({'age':age,'sex':sex,'method':'decode_age_weighted_mean_latent',
        'weights':weights.detach().cpu().tolist(),'subject_ids':frame['subject_id'].tolist(),'shape':shape,
        'spacing_mm':config['atlas']['spacing_mm'],'channel_mapping':CHANNELS},indent=2))
    if average_fields:
        field=np.zeros(len(coords),dtype=np.float64)
        for i in torch.where(weights>0)[0].tolist():
            prediction=predict_field(model,coords,condition,config,latent_override=model.latents[i:i+1])
            field+=weights[i].item()*prediction[:,CHANNELS['vessel_sdf_normalized']]*config['sdf']['truncation_mm']
        image=nib.Nifti1Image(field.reshape(shape).astype(np.float32),affine);image.header.set_xyzt_units('mm')
        nib.save(image,directory/'AveragedLevelSet_mm_NOT_exact_SDF.nii.gz')
    return directory
