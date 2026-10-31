"""Test-time fitting of new subjects (generalization test).

The trained network (backbone, modulator, heads) is frozen; only each new subject's latent grid and transform
are optimized, following the training procedure (vatl_sdf/train.py):
  alignment stage (align_epochs): latents gated to zero, transform only, alignment loss (wide SDF truncation +
      large-vessel centreline distance) + MoE balance;
  refinement: latents unfrozen; SDF + graph terms + field heads + L2 latent prior + affine regularization.
One epoch = steps_per_epoch optimizer steps per subject on a fresh 250k-point sample (training: 50 steps per
subject per epoch), so the default budget matches the per-subject update count of training. Learning rates are
the training ones with a per-epoch cosine of horizon_epochs (the training run length by default).

Template frame: the trained template. With mean_identity the population gauge was fixed during training; the
new subjects' effective transform is their own parameter (no mean over the new subjects is removed).
"""
import copy
import math

import numpy as np
import torch

from .config import condition_vector, field_names, graph_terms
from .dataset import alignment_enabled, load_subject, sample_graph, sample_sdf
from .losses import add_graph_losses, affine_regularization, alignment_losses, field_losses, sdf_losses
from .model import SDFModel
from .train import parameter_groups


def fitting_config(config):
    """Copy of the training config for the new subjects: the mean-identity gauge is not applied to them."""
    config = copy.deepcopy(config)
    transform = config['model'].get('transform')
    if transform and transform.get('mode', 'legacy') != 'legacy':
        transform['mean_identity'] = False
    return config


def build_fitting_model(checkpoint, num_subjects, device='cpu'):
    """Frozen trained network with fresh per-subject tables (latents zero, identity transforms)."""
    config = fitting_config(checkpoint['config'])
    model = SDFModel(config, num_subjects)
    tables = set(model.subject_tensors())
    state = {k: v for k, v in checkpoint['model'].items() if k not in tables}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != tables or unexpected:
        raise ValueError(f'Checkpoint/model mismatch: missing {missing}, unexpected {unexpected}')
    with torch.no_grad():
        model.latents.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in tables)
    return model.to(device), config


class SubjectSampler:
    """Training sampling (sample_sdf / sample_graph) for explicit subject rows; subjects are kept in memory."""
    def __init__(self, frame, config, seed_offset=0):
        self.frame, self.config, self.seed_offset = frame.reset_index(drop=True), config, seed_offset
        self.subjects = [load_subject(row, config) for _, row in self.frame.iterrows()]

    def __len__(self):
        return len(self.frame)

    def sample(self, index, epoch, stage):
        rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'], self.seed_offset, epoch, index]))
        result = sample_sdf(self.subjects[index], self.config, rng, alignment_stage=stage)
        if graph_terms(self.config) and not stage:
            graph, _ = sample_graph(self.subjects[index], self.config, rng)
            result.update(graph)
        row = self.frame.iloc[index]
        result['condition'] = condition_vector(self.config, row.scan_age, row.sex)
        return result


def epoch_chunks(samples, config, steps, stage, seed):
    """Per subject: shuffle its sample and split it into `steps` chunks. Returns a list over steps of
    (sdf arrays, {term: graph arrays}) concatenated over subjects."""
    terms = [] if stage else graph_terms(config)
    extra = [k for k in ('field_target', 'field_weight', 'align_target', 'align_weight') if k in samples[0]]
    rng = np.random.default_rng(seed)
    per_step = [{'sdf': [], 'graph': {t: [] for t in terms}} for _ in range(steps)]
    for index, s in enumerate(samples):
        n = len(s['sdf_coords']); perm = rng.permutation(n)
        cond = np.repeat(s['condition'][None], n, 0)
        arrays = [s['sdf_coords'], s['sdf_target_mm'], np.full(n, index), cond] + [s[k] for k in extra]
        arrays = [a[perm] for a in arrays]
        for k in range(steps):
            per_step[k]['sdf'].append([a[k * n // steps:(k + 1) * n // steps] for a in arrays])
        for t in terms:
            g = len(s[f'{t}_coords']); gperm = rng.permutation(g)
            garrays = [s[f'{t}_coords'], s[f'{t}_target_mm'], s[f'{t}_weight'], np.full(g, index), np.repeat(s['condition'][None], g, 0)]
            garrays = [a[gperm] for a in garrays]
            for k in range(steps):
                per_step[k]['graph'][t].append([a[k * g // steps:(k + 1) * g // steps] for a in garrays])
    join = lambda parts: [np.concatenate(column) for column in zip(*parts)]
    return [(join(p['sdf']), {t: join(p['graph'][t]) for t in terms}) for p in per_step], extra


def step_loss(model, config, sdf_arrays, graph_arrays, extra, stage):
    """Training loss of one step (train.py) for the subjects in the batch."""
    device = model.half_bbox.device
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, device=device, dtype=dtype)
    sc, st, si, sv = tensor(sdf_arrays[0]), tensor(sdf_arrays[1]).reshape(-1, 1), tensor(sdf_arrays[2], torch.long), tensor(sdf_arrays[3])
    ex = {k: tensor(v) for k, v in zip(extra, sdf_arrays[4:])}
    active = list(graph_arrays)
    pieces = {t: (tensor(g[0]), tensor(g[1]), tensor(g[2]), tensor(g[3], torch.long), tensor(g[4])) for t, g in graph_arrays.items()}
    coords = torch.cat([sc] + [pieces[t][0] for t in active]); ids = torch.cat([si] + [pieces[t][3] for t in active])
    cond = torch.cat([sv] + [pieces[t][4] for t in active])
    output, moe, fields = model(coords, ids, cond, balance_count=len(sc) if active else None, return_fields=True)
    outputs = output.split([len(sc)] + [len(pieces[t][0]) for t in active])
    if stage:
        parts = alignment_losses(outputs[0], st, fields['align_distance'][:len(sc)], ex['align_target'], ex['align_weight'], moe, config)
    else:
        parts = sdf_losses(outputs[0], st, moe, None, config)
        if active:
            parts = add_graph_losses(parts, [(o, pieces[t][1], pieces[t][2]) for o, t in zip(outputs[1:], active)], active, config)
        if field_names(config):
            parts = field_losses(parts, {k: v[:len(sc)] for k, v in fields.items() if k != 'align_distance'},
                                 ex['field_target'], ex['field_weight'], config)
        if alignment_enabled(config):
            subjects = torch.unique(si)
            prior = model.latents[subjects].flatten(1).square().sum(1).mean()
            parts['latent_prior_raw'] = prior
            parts['latent_prior_weighted'] = config['alignment']['latent_prior_weight'] * prior
            parts['total'] = parts['total'] + parts['latent_prior_weighted']
    regularization = affine_regularization(model)
    if regularization is not None:
        parts['affine_reg_raw'] = regularization
        parts['affine_reg_weighted'] = config['model']['transform'].get('affine_reg_weight', 0.) * regularization
        parts['total'] = parts['total'] + parts['affine_reg_weighted']
    return parts


def fit(model, config, sampler, align_epochs, total_epochs, steps_per_epoch=50, horizon_epochs=None, on_epoch_end=None,
        log=None):
    """Optimize the subject tables of `model` (all subjects of `sampler` jointly; their parameters are independent).
    on_epoch_end(epoch_completed, stage) is called after every epoch; log(record) after every step."""
    if config['model'].get('transform', {}).get('mode', 'legacy') == 'legacy':
        raise ValueError('Test-time fitting supports normalized transforms only')
    if align_epochs and not alignment_enabled(config):
        raise ValueError('align_epochs > 0 requires a checkpoint trained with an alignment stage')
    groups = [g for g in parameter_groups(model, config)[1:]]
    optimizer = torch.optim.Adam(groups, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon_epochs or total_epochs,
                                                           eta_min=config['optimizer']['eta_min'])
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    step = 0
    for epoch in range(total_epochs):
        stage = epoch < align_epochs
        model.latent_gate = 0.0 if stage else 1.0
        samples = [sampler.sample(i, epoch, stage) for i in range(len(sampler))]
        chunks, extra = epoch_chunks(samples, config, steps_per_epoch, stage, seed=config['seed'] * 1000 + epoch)
        for sdf_arrays, graph_arrays in chunks:
            optimizer.zero_grad(set_to_none=True)
            parts = step_loss(model, config, sdf_arrays, graph_arrays, extra, stage)
            if not torch.isfinite(parts['total']):
                raise FloatingPointError('Nonfinite test-time loss')
            parts['total'].backward(); optimizer.step(); step += 1
            if log:
                log({'epoch': epoch, 'step': step, 'stage': 'align' if stage else 'main',
                     **{k: float(v.detach()) for k, v in parts.items()}})
        scheduler.step()
        if on_epoch_end:
            on_epoch_end(epoch + 1, stage)
    model.latent_gate = 1.0
    for name, parameter in model.named_parameters():
        if name in frozen and not torch.equal(parameter.detach(), frozen[name]):
            raise RuntimeError(f'Frozen parameter {name} changed during test-time fitting')
    return step


def subject_maps(model):
    """(A, t_mm) of the fitted subjects as float64 numpy arrays."""
    a, t = model.maps_mm()
    return a.double().cpu().numpy(), t.double().cpu().numpy()


def training_steps_per_epoch(checkpoint_config):
    """Per-subject optimizer updates per training epoch: every step of a batch contains all its subjects, so
    ceil(subjects_per_rank * sampling.sdf.points / task_points_per_step) (50 for the A4 recipe)."""
    t = checkpoint_config['training']
    return math.ceil(t['subjects_per_rank'] * checkpoint_config['sampling']['sdf']['points'] / t['task_points_per_step'])
