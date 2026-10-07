"""
Multi-task loss for SpatialNet: box IoU losses, Wing fingertip, direction,
ray alignment, hand-axis cone, CenterNet heatmaps and focal BCE gates.
Imported by train.py; `python loss.py` runs a quick self-test.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


def _corners(b):
    return (b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2)


def _iou_parts(pred, target):
    px1, py1, px2, py2 = _corners(pred)
    tx1, ty1, tx2, ty2 = _corners(target)
    ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
    ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    pa = torch.clamp(px2 - px1, min=0) * torch.clamp(py2 - py1, min=0)
    ta = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)
    union = pa + ta - inter + 1e-8
    iou = inter / union
    ex1, ey1 = torch.min(px1, tx1), torch.min(py1, ty1)
    ex2, ey2 = torch.max(px2, tx2), torch.max(py2, ty2)
    return iou, (ex1, ey1, ex2, ey2)


def compute_ciou(pred, target):
    iou, (ex1, ey1, ex2, ey2) = _iou_parts(pred, target)
    c_diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-8
    rho_sq = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2
    v = (4.0 / (math.pi ** 2)) * (
        torch.atan(target[:, 2] / (target[:, 3] + 1e-8)) -
        torch.atan(pred[:, 2] / (pred[:, 3] + 1e-8))) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-8)
    return iou - rho_sq / c_diag_sq - alpha * v, iou


def compute_siou(pred, target):
    iou, (ex1, ey1, ex2, ey2) = _iou_parts(pred, target)
    cw, ch = ex2 - ex1, ey2 - ey1
    dcx = target[:, 0] - pred[:, 0]
    dcy = target[:, 1] - pred[:, 1]
    dist = torch.sqrt(dcx ** 2 + dcy ** 2 + 1e-8)
    sin_alpha = torch.abs(dcy) / dist
    sin_beta = torch.abs(dcx) / dist
    threshold = (2 ** 0.5) / 2
    sin_c = torch.where(sin_alpha <= threshold, sin_alpha, sin_beta)
    angle_cost = 1 - 2 * torch.sin(torch.arcsin(sin_c.clamp(-1 + 1e-6, 1 - 1e-6)) - math.pi / 4) ** 2
    rho_x = (dcx / (cw + 1e-8)) ** 2
    rho_y = (dcy / (ch + 1e-8)) ** 2
    gamma_coeff = 2 - angle_cost
    dist_cost = rho_x.pow(gamma_coeff) + rho_y.pow(gamma_coeff)
    omega_w = torch.abs(pred[:, 2] - target[:, 2]) / (torch.max(pred[:, 2], target[:, 2]) + 1e-8)
    omega_h = torch.abs(pred[:, 3] - target[:, 3]) / (torch.max(pred[:, 3], target[:, 3]) + 1e-8)
    shape_cost = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
    return iou - (dist_cost + shape_cost) / 2, iou


def wing_loss(pred, target, w=0.5, epsilon=0.1):
    delta = (pred - target).abs()
    C = w - w * math.log(1.0 + w / epsilon)
    return torch.where(delta < w, w * torch.log(1.0 + delta / epsilon), delta - C)


def focal_bce(logits, target, gamma=2.0):
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    return (1.0 - p_t) ** gamma * bce


def hand_axis_cone_loss(fingertip, hand_bbox, angle, mask, margin_cos=None):
    """Hinge keeping the predicted direction within a cone (margin_cos) around
    the hand axis (hand centre -> fingertip); zero inside the cone."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=mask.device)
    margin_cos = cfg.HAND_AXIS_MARGIN_COS if margin_cos is None else margin_cos
    axis = F.normalize(fingertip - hand_bbox[:, :2] + 1e-8, p=2, dim=1)
    ray = F.normalize(torch.stack([angle[:, 1], angle[:, 0]], dim=1) + 1e-8, p=2, dim=1)
    cos_sim = (axis * ray).sum(dim=1)
    return (F.relu(margin_cos - cos_sim) * mask).sum() / (mask.sum() + 1e-8)


def ray_alignment_loss(fingertip, angle, roi_bbox, mask):
    """Perpendicular distance of the target centre from the fingertip ray, plus
    a penalty when the target lies behind the fingertip. angle is (sin, cos).
    The penalty grows with target distance for a fixed angular error."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=mask.device)
    vec = roi_bbox[:, :2] - fingertip
    ray = F.normalize(torch.stack([angle[:, 1], angle[:, 0]], dim=1) + 1e-8, p=2, dim=1)
    proj_len = (vec * ray).sum(dim=1, keepdim=True)
    perp = torch.norm(vec - proj_len * ray, p=2, dim=1)
    behind = F.relu(-proj_len.squeeze(1))
    loss = perp + 0.5 * behind
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def gaussian_target(points, H, W, sigma_cells, valid):
    """Render one Gaussian per sample at `points` ([B,2] normalised xy).
    Returns [B,1,H,W]; samples with valid==0 get an all-zero map."""
    B = points.size(0)
    dev, dt = points.device, points.dtype
    ys = torch.arange(H, device=dev, dtype=dt).view(1, H, 1)
    xs = torch.arange(W, device=dev, dtype=dt).view(1, 1, W)
    px = (points[:, 0] * (W - 1)).view(B, 1, 1)
    py = (points[:, 1] * (H - 1)).view(B, 1, 1)
    d2 = (xs - px) ** 2 + (ys - py) ** 2
    g = torch.exp(-d2 / (2 * sigma_cells ** 2))
    # nearest cell set to exactly 1 so the focal loss sees one positive per instance
    ix = px.round().clamp(0, W - 1).long().view(B, 1)
    iy = py.round().clamp(0, H - 1).long().view(B, 1)
    flat = g.reshape(B, H * W).clone()
    flat.scatter_(1, iy * W + ix, 1.0)
    g = flat.view(B, H, W) * valid.view(B, 1, 1)
    return g.unsqueeze(1)


def distractor_weight_map(boxes, num, H, W, sigma_cells, weight):
    """Extra negative weight (1 + weight * Gaussian) at each distractor-hand
    centre. boxes [B,K,4], num [B]. Returns [B,1,H,W] multiplier."""
    B, K, _ = boxes.shape
    if weight <= 0 or K == 0:
        return torch.ones(B, 1, H, W, device=boxes.device, dtype=boxes.dtype)
    dev, dt = boxes.device, boxes.dtype
    ys = torch.arange(H, device=dev, dtype=dt).view(1, 1, H, 1)
    xs = torch.arange(W, device=dev, dtype=dt).view(1, 1, 1, W)
    px = (boxes[:, :, 0] * (W - 1)).view(B, K, 1, 1)
    py = (boxes[:, :, 1] * (H - 1)).view(B, K, 1, 1)
    valid = (torch.arange(K, device=dev).view(1, K) < num.view(B, 1)).to(dt).view(B, K, 1, 1)
    g = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma_cells ** 2)) * valid
    g = g.max(dim=1, keepdim=True).values
    return 1.0 + weight * g


def heatmap_focal(logits, target, neg_weight=None, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss, normalised by the number of
    positive peaks (>= 1). `neg_weight` multiplies the negative term."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    target = target.float()
    pos = (target > 0.999).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    if neg_weight is not None:
        neg_loss = neg_loss * neg_weight
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None, focal_gamma=None, fn_penalty_mult=None,
                 hand_fn_penalty_mult=None, ownership_neg_weight=None):
        super().__init__()
        self.w = dict(cfg.LOSS_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.focal_gamma = cfg.FOCAL_GAMMA if focal_gamma is None else focal_gamma
        self.fn_mult = cfg.FN_PENALTY_MULT if fn_penalty_mult is None else fn_penalty_mult
        self.hand_fn_mult = cfg.HAND_FN_PENALTY_MULT if hand_fn_penalty_mult is None else hand_fn_penalty_mult
        self.own_w = cfg.OWNERSHIP_NEG_WEIGHT if ownership_neg_weight is None else ownership_neg_weight
        self.cosine = nn.CosineSimilarity(dim=1)

    @staticmethod
    def _masked_mean(x, mask):
        return (x * mask).sum() / (mask.sum() + 1e-8)

    def _gate(self, logits, target, pop_mask, fn_mult):
        """Focal BCE on a conditional population with asymmetric FN weight."""
        if pop_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        fl = focal_bce(logits, target, self.focal_gamma).squeeze(1)
        w = torch.where(target.squeeze(1) > 0.5,
                        torch.full_like(fl, fn_mult), torch.ones_like(fl))
        return self._masked_mean(fl * w, pop_mask)

    def forward(self, preds, batch, img_size=None):
        img_size = img_size or cfg.IMG_SIZE
        device = preds['hand_conf'].device
        hc_t, pc_t, rc_t = batch['hand_conf'], batch['point_conf'], batch['roi_conf']
        hc_m, pc_m = hc_t.squeeze(1), pc_t.squeeze(1)
        valid = batch['valid_mask']
        all_m = torch.ones_like(hc_m)

        hand_conf_loss = self._gate(preds['hand_conf'], hc_t, all_m, self.hand_fn_mult)
        point_conf_loss = self._gate(preds['point_conf'], pc_t, hc_m, self.fn_mult)
        roi_conf_loss = self._gate(preds['roi_conf'], rc_t, pc_m, self.fn_mult)

        hand_m = valid[:, 0] * hc_m
        ft_m = valid[:, 1] * pc_m
        roi_m = valid[:, 2] * pc_m

        if hand_m.sum() > 0:
            ciou, hiou = compute_ciou(preds['hand'], batch['hand_bbox'])
            hand_loss = self._masked_mean(1 - ciou, hand_m)
            hand_iou_m = self._masked_mean(hiou, hand_m)
        else:
            hand_loss = hand_iou_m = torch.tensor(0., device=device)

        if roi_m.sum() > 0:
            siou, riou = compute_siou(preds['roi'], batch['roi_bbox'])
            roi_loss = self._masked_mean(1 - siou, roi_m)
            roi_iou_aux = self._masked_mean(1 - riou, roi_m)
            roi_iou_m = self._masked_mean(riou, roi_m)
        else:
            roi_loss = roi_iou_aux = roi_iou_m = torch.tensor(0., device=device)

        if ft_m.sum() > 0:
            ft_loss = self._masked_mean(wing_loss(preds['fingertip'], batch['fingertip_kp']).mean(1), ft_m)
            ft_px = torch.sqrt((((preds['fingertip'] - batch['fingertip_kp']) * img_size) ** 2).sum(1))
            ft_px_m = self._masked_mean(ft_px, ft_m)
            ang_loss = self._masked_mean(1 - self.cosine(preds['angle'], batch['sin_cos_angle']), ft_m)
        else:
            ft_loss = ft_px_m = ang_loss = torch.tensor(0., device=device)

        ray_loss = ray_alignment_loss(preds['fingertip'], preds['angle'], preds['roi'], ft_m * roi_m)

        axis_loss = hand_axis_cone_loss(preds['fingertip'], preds['hand'],
                                        preds['angle'], ft_m * hand_m)

        H, W = preds['hand_hm'].shape[-2:]
        hand_t = gaussian_target(batch['hand_bbox'][:, :2], H, W, cfg.HEATMAP_SIGMA['hand'], hand_m)
        neg_w = None
        if 'distractor_boxes' in batch and self.own_w > 0:
            neg_w = distractor_weight_map(batch['distractor_boxes'], batch['num_distractors'].view(-1),
                                          H, W, cfg.HEATMAP_SIGMA['hand'], self.own_w)
        hand_hm_loss = heatmap_focal(preds['hand_hm'], hand_t, neg_w, cfg.HM_ALPHA, cfg.HM_BETA)

        ft_t = gaussian_target(batch['fingertip_kp'], H, W, cfg.HEATMAP_SIGMA['fingertip'], ft_m)
        ft_hm_loss = heatmap_focal(preds['ft_hm'], ft_t, None, cfg.HM_ALPHA, cfg.HM_BETA) \
            if ft_m.sum() > 0 else torch.tensor(0., device=device)

        roi_t = gaussian_target(batch['roi_bbox'][:, :2], H, W, cfg.HEATMAP_SIGMA['roi'], roi_m)
        roi_hm_loss = heatmap_focal(preds['roi_hm'], roi_t, None, cfg.HM_ALPHA, cfg.HM_BETA) \
            if roi_m.sum() > 0 else torch.tensor(0., device=device)

        w = self.w
        total = (w['hand'] * hand_loss + w['fingertip'] * ft_loss + w['angle'] * ang_loss +
                 w['roi'] * roi_loss + w['roi_iou_aux'] * roi_iou_aux +
                 w['hand_conf'] * hand_conf_loss + w['point_conf'] * point_conf_loss +
                 w['roi_conf'] * roi_conf_loss + w['ray'] * ray_loss +
                 w.get('hand_axis', 0.0) * axis_loss +
                 w['hand_hm'] * hand_hm_loss + w['ft_hm'] * ft_hm_loss + w['roi_hm'] * roi_hm_loss)

        with torch.no_grad():
            def acc_fn(logits, t, pop):
                if pop.sum() == 0:
                    return 0.0, 0.0
                pb = (torch.sigmoid(logits).squeeze(1) > 0.5).float()
                tb = t.squeeze(1)
                acc = self._masked_mean((pb == tb).float(), pop).item()
                pos = tb * pop
                fn = ((pb < 0.5).float() * pos).sum() / (pos.sum() + 1e-8) if pos.sum() > 0 else torch.tensor(0.)
                return acc, float(fn)
            if ft_m.sum() > 0:
                cos_a = (preds['angle'] * batch['sin_cos_angle']).sum(dim=1)
                cos_a = cos_a.clamp(-1 + 1e-7, 1 - 1e-7)
                ang_err_deg = float(self._masked_mean(torch.rad2deg(torch.acos(cos_a)), ft_m))
            else:
                ang_err_deg = 0.0
            hc_acc, hand_fn = acc_fn(preds['hand_conf'], hc_t, all_m)
            pc_acc, point_fn = acc_fn(preds['point_conf'], pc_t, hc_m)
            rc_acc, roi_fn = acc_fn(preds['roi_conf'], rc_t, pc_m)

            # ownership: predicted hand centre nearer the wearer hand than any distractor
            own_acc = 0.0
            if 'distractor_boxes' in batch:
                nd = batch['num_distractors'].view(-1)
                sel = (nd > 0) & (hand_m > 0)
                if sel.sum() > 0:
                    c = preds['hand'][:, :2]
                    d_w = torch.norm(c - batch['hand_bbox'][:, :2], dim=1)
                    K = batch['distractor_boxes'].shape[1]
                    dd = torch.norm(c.unsqueeze(1) - batch['distractor_boxes'][:, :, :2], dim=2)
                    kmask = torch.arange(K, device=device).view(1, K) < nd.view(-1, 1)
                    dd = torch.where(kmask, dd, torch.full_like(dd, 1e3)).min(dim=1).values
                    own_acc = ((d_w < dd).float()[sel]).mean().item()

        def _f(x):
            return float(x.detach()) if torch.is_tensor(x) else float(x)

        metrics = {
            'roi_iou': _f(roi_iou_m), 'hand_iou': _f(hand_iou_m), 'ft_px_err': _f(ft_px_m),
            'ray_loss': _f(ray_loss), 'axis_loss': _f(axis_loss),
            'angle_err_deg': ang_err_deg, 'hand_hm': _f(hand_hm_loss),
            'hand_conf_acc': hc_acc, 'point_conf_acc': pc_acc, 'roi_conf_acc': rc_acc,
            'hand_fn_rate': hand_fn, 'point_fn_rate': point_fn, 'roi_fn_rate': roi_fn,
            'ownership_acc': own_acc,
        }
        return total, metrics


if __name__ == "__main__":
    torch.manual_seed(0)
    crit = MultiTaskLoss()
    B, G = 4, 20
    preds = {
        'hand': torch.rand(B, 4), 'fingertip': torch.rand(B, 2),
        'angle': F.normalize(torch.rand(B, 2), dim=1), 'roi': torch.rand(B, 4),
        'hand_conf': torch.randn(B, 1), 'point_conf': torch.randn(B, 1), 'roi_conf': torch.randn(B, 1),
        'hand_hm': torch.randn(B, 1, G, G, requires_grad=True),
        'ft_hm': torch.randn(B, 1, G, G), 'roi_hm': torch.randn(B, 1, G, G),
    }
    batch = {
        'hand_bbox': torch.rand(B, 4), 'fingertip_kp': torch.rand(B, 2),
        'sin_cos_angle': F.normalize(torch.rand(B, 2), dim=1), 'roi_bbox': torch.rand(B, 4),
        'valid_mask': torch.ones(B, 3), 'hand_conf': torch.ones(B, 1),
        'point_conf': torch.ones(B, 1), 'roi_conf': torch.ones(B, 1),
        'distractor_boxes': torch.rand(B, 2, 4), 'num_distractors': torch.tensor([[2], [1], [0], [2]]),
    }
    loss, m = crit(preds, batch)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print({k: round(v, 4) for k, v in m.items()})
    print("[OK] loss module")
