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


def weighted_mean(values, weight):
    return (values * weight).sum() / weight.sum().clamp_min(1.)


def graph_term_loss(term, prediction, target_mm, weight, truncation_mm):
    """prediction: normalized SDF (n, 1). Targets in mm, normalized by truncation like the SDF."""
    target = target_mm.clamp(-truncation_mm, truncation_mm) / truncation_mm
    if term in ('centerline_sdf', 'radius'):
        return weighted_mean((prediction - target).abs(), weight)
    if term in ('centerline_hinge', 'edge_hinge'):
        return weighted_mean(F.relu(prediction - target), weight)
    raise ValueError(f'Unknown graph term {term}')


def add_graph_losses(parts, predictions, batch_terms, config):
    """Adds {term}_raw/{term}_weighted for each graph term and updates the total."""
    tau = config['sdf']['truncation_mm']
    for term, (prediction, target_mm, weight) in zip(batch_terms, predictions):
        raw = graph_term_loss(term, prediction, target_mm, weight, tau)
        parts[f'{term}_raw'] = raw
        parts[f'{term}_weighted'] = config['graph'][term]['weight'] * raw
        parts['total'] = parts['total'] + parts[f'{term}_weighted']
    return parts


def eikonal_loss(sdf_normalized, coords, half_bbox_mm, truncation_mm, band=None):
    """mean (|grad SDF|_mm - 1)^2 over points in band. Inputs are normalized by the anisotropic bbox
    half-lengths and outputs by the truncation, so d SDF_mm / d x_mm = truncation * d s / d u / half_bbox
    (component-wise). Each output depends only on its own input point, so grad of the sum is per point."""
    gradient, = torch.autograd.grad(sdf_normalized.sum(), coords, create_graph=True)
    physical = gradient * truncation_mm / half_bbox_mm
    residual = (physical.norm(dim=-1) - 1).square()
    if band is None:
        return residual.mean()
    return (residual * band).sum() / band.sum().clamp_min(1)
