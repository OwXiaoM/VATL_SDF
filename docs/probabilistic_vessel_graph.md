# Label-free probabilistic vessel graph: feasibility and design

Status: design document (Phase 5). Nothing here is implemented beyond the prototypes named in
"Evidence so far". Constraints: binary vessel masks (segmentation label 1) and imperfect skeleton
graphs only; no anatomical labels; every method must tolerate missing connections and spurious loops.

## 1. Why a graph-valued atlas

* An SDF atlas decoded from a mean latent, and any per-point statistic (frequency p(x), SDF mean and
  variance), is a marginal. It carries no connectivity: a thin vessel present in all subjects but
  shifted 2-3 mm has p(x) of 0.2-0.3 along its course and vanishes at a 0.5 threshold.
* Connectivity is a property of an individual sample. It survives only through correspondence, a
  generative model whose samples are complete trees, or an explicit topological representation.
* A skeleton with radii renders analytically into an SDF (round cone per edge; min or smooth min),
  so a graph is a parametric, connected-by-construction representation of the same geometry
  (`vatl_sdf/render_graph.py`).

## 2. Data facts that shape the design (Phase 0, 552 subjects)

* Graph node positions are world mm in the segmentation frame; radius is mm, an EDT on a 0.8 mm
  isotropic regrid (values 0.8 sqrt(k)); 84% of nodes sit at the 0.8 mm floor (one graph voxel),
  which overstates one-voxel vessels about twofold.
* 99.8% of nodes lie within 0.4 mm (half a graph voxel) of the voxelized mask; thin-vessel nodes
  fall just outside one-voxel-thick vessels by sub-voxel amounts.
* Graphs are more fragmented than masks: median 51 graph pieces (largest 68% of nodes) vs 93.5
  26-connected mask components (largest 84% of voxels, many tiny); 17 graph pieces lie inside the
  largest mask component. No graph edge joins two mask components.
* Median 35 independent cycles per graph: ~3 short (< 5 mm, mostly floor radius, half their nodes
  outside the mask: likely touching-vessel / discretization artefacts), ~31 long (median 20 mm).
  Long loops can be true anastomoses or merges of touching vessels; this is not established.

## 3. Consensus graph from population density

1. Map every subject's mask-supported graph to template space with its learned rigid transform
   (direction verified: `tests/test_template_mapping.py`).
2. Radius-aware centreline density D(x) = expected centreline length per voxel per subject
   (`analysis/phase5/centerline_density.py`), overall and by calibre.
3. Consensus centrelines = ridges of D (or of a density sharpened by radius), traced into a graph
   (same tracking as Phase 4 extraction, with the density gradient/Hessian giving tangents).
4. Per-edge existence probability: fraction of subjects with a matching segment, where a match is a
   subject centreline within distance d of the consensus edge over >= f of its length and with
   compatible direction (|cos| >= c). Report sensitivity to d (1, 2, 3 mm), f and c; d below the
   residual misregistration under-counts thin vessels, d above the inter-vessel spacing merges
   neighbours.
5. Offset and radius distributions per edge from the matched segments (offset vector in the edge's
   normal plane; radius from the reference SDF along the match, not from the quantized graph radius).

## 4. Robustness to annotation noise without labels

* Real structures, including true anastomoses, recur at consistent template positions; annotation
  artefacts (spurious short loops, stray fragments, missing links) do not. Population recurrence is
  the label-free signal. Phase 5 measures recurrence vs calibre to see where it is informative.
* Bootstrap over subjects: recompute consensus graph and edge probabilities on resamples; keep edges
  whose probability is stable (e.g. CI width < 0.2) and that exceed a recurrence threshold.
* Failure modes: (a) rigid alignment leaves large residual offsets for peripheral vessels, so true
  vessels look non-recurrent; (b) systematic artefacts (the same touching-vessel merge in many
  subjects) recur and would be accepted; (c) variable anatomy with several common configurations
  produces mixed consensus; (d) graph fragmentation lowers matched length fractions uniformly.
  Mitigations: deformable correspondence (section 6), mixture modelling of configurations,
  per-calibre thresholds, and the mask-support rule for every subject segment.

## 5. Hierarchy from statistics, not names

* Core: high-probability, large-radius edges (stable under bootstrap) modelled edge by edge
  (existence probability, offset/radius distributions, age dependence).
* Periphery: low-probability or highly variable small vessels modelled statistically (density of
  branching per region, length/radius/tortuosity distributions, attachment points on the core),
  i.e. a generative model whose samples are trees attached to the core rather than averaged
  geometry.
* The split is a threshold on probability x calibre, chosen from the recurrence analysis.

## 6. Correspondence options

| Option | Cost | Pros | Cons |
| --- | --- | --- | --- |
| Rigid (current, learned in the INR) | none | already available; exact inverse | residual offsets grow toward the periphery |
| Deformable registration of masks/SDFs (e.g. diffeomorphic, SDF-driven) | hours CPU/GPU per cohort | standard, invertible | thin vessels drive little of the similarity; may misalign them |
| Learned deformation field inside the INR (template SDF + per-subject displacement) | retraining | joint with the atlas; graph points can be pushed through it | identifiability vs latent capacity; needs regularization |
| Unsupervised graph / point-set matching (e.g. CPD, optimal transport on centreline samples with tangents) | minutes per pair | uses topology directly | pairwise; needs groupwise extension; sensitive to fragmentation |

Recommended order: rigid baseline -> point-set matching of centreline samples (with tangent and
radius features) to refine correspondences -> learned deformation inside the INR if residuals remain.

## 7. Coupling with the INR (both directions)

* Graph -> INR: rendered graph SDF as a prior or as additional supervision restricted to the mask
  (Phase 2 terms already restrict graph supervision to mask-supported parts); edge-path hinges keep
  interior paths within each piece without bridging pieces.
* INR -> graph: the INR supplies residual geometric detail (true radii, lumen shape) and fills
  under-annotated vessels; implicit graph fields (Phase 4: centreline distance, radius, tangent,
  junction heads on the shared backbone) are the shared embedding from which graphs are extracted.
* Atlas sampling: sample a graph from the probabilistic graph model, render it, and refine with the
  INR conditioned on the sampled graph; each sample is connected by construction.

## 8. Evaluation

* Sample connectivity relative to the reference: component counts and largest-component fraction of
  rendered samples vs subject masks (never vs one connected component).
* Rendered-vs-reference agreement for individual graphs (Dice, band MAE, connectivity), to separate
  model error from graph imperfection.
* Bootstrap stability of edge probabilities; sensitivity to d, f, c.
* Held-out subjects: likelihood of their graphs under the model (matched-length coverage).

## 9. Interface for anatomical labels (future)

Labels are optional metadata, never required: `edge_attributes: {label: str | None, label_source:
str | None, confidence: float | None}` per consensus edge and per subject segment. Matching and
probabilities ignore labels; if annotations (or an external automatic labeller) become available,
they can (i) name consensus edges by majority vote, (ii) constrain matching (same-label matches
only), and (iii) stratify statistics. No method in this repository may depend on labels.

## Evidence so far

Filled from Phase 5 runs; see CONTEXT.md for job IDs.

* Rendering prototype (job 46354; 20 subjects spread over the TSV; native grids; all edges as given;
  medians). Graph rendered with its stored radii (hard min): Dice 0.71 vs the reference mask, recall
  0.65, precision 0.78, volume ratio 0.83, band MAE 0.53 mm; 26-components >= 10 voxels 1.57x the
  mask's; largest-component fraction 0.84 vs 0.91. The graph therefore captures about two thirds of
  the mask volume and is more fragmented than the mask; it is an incomplete, not a wrong, description.
  Radius from -reference SDF at nodes (hard min) is worse (Dice 0.66, 4.3x components): thin-vessel
  nodes sit just outside one-voxel vessels, so their SDF radius is tiny and cones stop overlapping
  voxel centres. Exponential smooth min (k = 0.3 mm) inflates volume (ratio 1.42 with graph radii)
  because ~10 cones overlap at each point (-k log n); a pairwise polynomial smooth min or a
  per-point top-2 smooth min is needed for a fair smooth union. Outputs:
  `training_outputs/sdf_only/analysis/20261030_phase5_render/`.
* Population centreline density / recurrence vs calibre: pending the full 00_sdf_baseline checkpoint.
