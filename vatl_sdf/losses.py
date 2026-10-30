import torch
from torch.nn import functional as F
from . import CHANNELS


def normalized_sdf(sdf_mm, truncation_mm):
    if truncation_mm <= 0 or not torch.isfinite(sdf_mm).all():
        raise ValueError('SDF requires finite distances and positive truncation')
    return sdf_mm.clamp(-truncation_mm,truncation_mm)/truncation_mm


def metric(name, x, y):
    if name == 'l1':
        return F.l1_loss(x, y)
    if name == 'smooth_l1':
        return F.smooth_l1_loss(x, y)
    raise ValueError('Metric must be l1 or smooth_l1')


def sdf_losses(output, sdf_target_mm, moe, transform_parameters, config):
    """total = sdf_weight * L1(normalized SDF) + moe_weight * MoE balance + tf_weight * transform."""
    if output.shape != (sdf_target_mm.numel(), 1):
        raise ValueError('Output/target sample counts disagree')
    c = config['loss']
    sdf = metric(c['sdf_metric'], output[:, CHANNELS['vessel_sdf_normalized']],
                 normalized_sdf(sdf_target_mm, config['sdf']['truncation_mm']).reshape(-1))
    # Rotation in radians, translation divided by bounding-box half lengths.
    if transform_parameters is not None:
        half = output.new_tensor(config['data']['world_bbox_mm'])/2.
        tf = transform_parameters[:,:3].square().mean()+(transform_parameters[:,3:]/half).square().mean()
    else:
        tf = output.sum()*0.
    parts = {'sdf_raw':sdf,'moe_raw':moe,'transform_raw':tf,
             'sdf_weighted':c['sdf_weight']*sdf,'moe_weighted':c['moe_weight']*moe,'transform_weighted':c['tf_weight']*tf}
    parts['total'] = parts['sdf_weighted']+parts['moe_weighted']+parts['transform_weighted']
    return parts
