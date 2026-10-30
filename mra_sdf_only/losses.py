import torch
from torch.nn import functional as F
from . import CHANNELS


def normalized_sdf(sdf_mm, truncation_mm):
    if truncation_mm <= 0 or not torch.isfinite(sdf_mm).all():
        raise ValueError('SDF requires finite distances and positive truncation')
    return sdf_mm.clamp(-truncation_mm,truncation_mm)/truncation_mm


def task_losses(output, mra_count, mra_target, sdf_target_mm, moe, transform_parameters, config):
    if output.shape != (mra_target.numel()+sdf_target_mm.numel(),2) or mra_count != mra_target.numel():
        raise ValueError('Output/task sample counts disagree')
    c = config['loss']
    def metric(name,x,y):
        if name=='l1':return F.l1_loss(x,y)
        if name=='smooth_l1':return F.smooth_l1_loss(x,y)
        raise ValueError('Metric must be l1 or smooth_l1')
    mra = metric(c['mra_metric'],output[:mra_count,CHANNELS['mra_intensity']],mra_target.reshape(-1))
    sdf = metric(c['sdf_metric'],output[mra_count:,CHANNELS['vessel_sdf_normalized']],normalized_sdf(sdf_target_mm,config['sdf']['truncation_mm']).reshape(-1))
    # Rotation in radians, translation divided by bounding-box half lengths.
    if transform_parameters is not None:
        half = output.new_tensor(config['data']['world_bbox_mm'])/2.
        tf = transform_parameters[:,:3].square().mean()+(transform_parameters[:,3:]/half).square().mean()
    else:
        tf = output.sum()*0.
    parts = {'mra_raw':mra,'sdf_raw':sdf,'moe_raw':moe,'transform_raw':tf,
             'mra_weighted':c['mra_weight']*mra,'sdf_weighted':c['sdf_weight']*sdf,
             'moe_weighted':c['moe_weight']*moe,'transform_weighted':c['tf_weight']*tf}
    parts['total'] = sum(parts[k] for k in ('mra_weighted','sdf_weighted','moe_weighted','transform_weighted'))
    return parts


def gradient_diagnostics(parts, parameter):
    gradients = []
    for key in ('mra_weighted','sdf_weighted'):
        gradient, = torch.autograd.grad(parts[key],parameter,retain_graph=True,allow_unused=True)
        gradients.append(gradient.detach().flatten() if gradient is not None else torch.zeros_like(parameter).flatten())
    return {'mra_gradient_norm':gradients[0].norm().item(),'sdf_gradient_norm':gradients[1].norm().item(),
            'gradient_cosine':F.cosine_similarity(gradients[0],gradients[1],dim=0).item()}
