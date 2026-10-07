# Method details and implementation correspondence

The compact manuscript describes the implemented model. These notes preserve exact details without adding an exhaustive appendix to the paper.

## Sources

Architecture: ../model.py. Objectives: ../loss.py and ../config.py. Training: ../train.py.
Augmentation/loader: ../dataset.py. Conversion: ../egopoint_preprocessing/prepare_egopoint.py.
Frozen reference: ../runs/main_run/outputs_spatialnet/best_model.pth.
Run-specific settings and selection: each run's training/run_info.json and evaluation/metrics.json.

## Loss weights, in implementation order

hand CIoU: 0.10
fingertip Wing: 0.18
direction cosine: 0.03
target IoU plus distance/shape penalties: 0.45
auxiliary target 1-IoU: 0.10
hand gate: 0.08
pointing gate: 0.10
target gate: 0.06
ray alignment: 0.08
hand-axis cone: 0.02
hand / fingertip / target heatmaps: 0.10 each

Each term is masked to its valid annotation population. Wing uses w=0.5 and epsilon=0.1.
Heatmap Gaussian sigma in feature cells: 1.5 / 0.8 / 1.2.
Heatmap focal parameters alpha=2, beta=4; gate focal gamma=2.
Positive false-negative multipliers are 3 for hand and 2 for the downstream gates.

## Exact target box objective

Let IoU be intersection/union of the normalized, unclipped boxes.
Let dx,dy be target-minus-prediction centre displacement; cw,ch the enclosing rectangle width/height.
Set rho_x=(dx/(cw+eps))^2 and rho_y=(dy/(ch+eps))^2.
Let z=min(abs(dx),abs(dy))/sqrt(dx^2+dy^2+eps), clipped before arcsin.
angle_cost = 1 - 2 sin(arcsin(z) - pi/4)^2; gamma = 2-angle_cost.
distance_cost = rho_x^gamma + rho_y^gamma.
omega_w=abs(w-w_gt)/(max(w,w_gt)+eps), with omega_h defined analogously.
shape_cost=(1-exp(-omega_w))^4 + (1-exp(-omega_h))^4.
The principal target loss is 1-IoU + (distance_cost+shape_cost)/2.
The auxiliary loss is 1-IoU. eps=1e-8 (arcsin clipping uses 1e-6).

This is SIoU-inspired. The code's power distance term differs from canonical SIoU's exponential distance term; the paper does not silently equate them.

## Architectural details

Backbone: ShuffleNetV2 0.5x feature stages, output channels 48/96/192 at strides 8/16/32.
Neck: two 192-channel BiFPN layers. Parameter-free SimAM follows depthwise smoothing.
Shared map: coordinate channels, depthwise-separable projection to 256 channels, two 5x5 residual mixer blocks.
Centre decoder: selected-peak window, (inverse temperature,radius) 8/2 hand, 10/1 fingertip, 8/2 target.
Local samples: 3x3 at one-cell spacing, 128D projection.
Hand descriptor: 5x5 bilinear box lattice enlarged 1.15x, projected to 128D and fused with box coordinates.
Fingertip prior: detached predicted hand box, 1.3x dilation, penalty strength 4 after temperature scaling.
Target conditioning: six ray maps concatenated with appearance, 48-channel bottleneck, zero-initialized final layer.
Direction is stored as (sin,cos) in the implementation; manuscript uses geometric (dx,dy) order.
The target size head reads a local patch in both the full model and A4.

## Ownership augmentation

Attempt probability 0.5 on hand-positive and 0.3 on hand-negative images.
Up to two distractors; relative scale 0.45-1.0; centre in the upper 65% of the image.
Reorientation probability 0.7; self-paste probability 0.3; background-decoy probability 0.15.
Maximum overlap fraction with original hand/target: 0.15.
Recorded distractor boxes produce a Gaussian negative weighting 1+4*D at the wearer-hand heatmap.
The pasted hands are not positive pointing instances; they are not literally unannotated.

## Ablation interventions

A1 removes both copy-paste and extra negative weighting, not either in isolation.
A2 removes both ray-alignment and hand-axis losses.
A3 removes the target heatmap conditioning branch; it keeps direction supervision and consistency losses.
A4 removes fingertip hand-offset conditioning and box prior, uses pooled direction/gate inputs, and removes the hand-descriptor pathway from those heads. It retains local box-size heads, local soft-argmax, and target ray conditioning. It is not a wholly independent-head baseline.

## Training and cost

AdamW; batch 32; weight decay 1e-4; peak learning rates 0.003 heads / 0.0003 backbone; one-cycle 10% warm-up.
Maximum 180 epochs; patience 25 on validation loss; gradient clipping 1; mixed precision.
Ground-truth anchor forcing: p=0.5 decaying linearly to zero over 20 epochs.
Same-state image mixing retains the dominant image's spatial labels rather than interpolating spatial targets.

Full model: 722,104 parameters. A3: 708,999. A4: 709,191.
Counted GFLOPs: 160=0.231199232, 224=0.452650496, 320=0.923234432.
A multiply-accumulate counts as two operations; unsupported operators, including bilinear sampling, are omitted.
Serialized FP16 weights are not total runtime memory. No training or aggregate evaluation was rerun for this paper revision.
