"""
SpatialNet: ShuffleNetV2 + BiFPN trunk with chained instance-anchored heads
(wearer hand -> fingertip -> pointing direction -> target box). ONNX opset 17 exportable.

Imported by the training/eval/export scripts; `python model.py` runs a shape and gradient self-test.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ShuffleV2Block(nn.Module):
    """ShuffleNet V2 building block with channel shuffle."""

    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2Backbone(nn.Module):
    """ShuffleNetV2 (0.5x) backbone with pretrained weights support."""

    def __init__(self, load_pretrained=True, pretrained_path=None):
        super(ShuffleNetV2Backbone, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                   mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

        # Default Conv2d init lets the signal decay to ~1e-6 by stage 4; pretrained weights overwrite this.
        self._initialize_weights()
        if load_pretrained:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path=None):
        if pretrained_path is None:
            possible_paths = [
                cfg.PRETRAINED_WEIGHTS,
                './shufflenetv2.pth',
                './module/shufflenetv2.pth',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shufflenetv2.pth'),
            ]
            for path in possible_paths:
                if path and os.path.exists(path):
                    pretrained_path = path
                    break

        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            self.load_state_dict(state_dict, strict=True)
            self.pretrained_path = pretrained_path
        else:
            self.pretrained_path = None
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        P1 = self.stage2(x)   # 48 channels, H/8
        P2 = self.stage3(P1)  # 96 channels, H/16
        P3 = self.stage4(P2)  # 192 channels, H/32
        return P1, P2, P3


class SimAM(nn.Module):
    """Parameter-free SimAM attention."""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class FastNormalizedFusion(nn.Module):
    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.eps = eps

    def forward(self, inputs):
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        out = 0.0
        for wi, xi in zip(w, inputs):
            out = out + wi * xi
        return out


class DWSmooth(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.simam = SimAM()

    def forward(self, x):
        return self.simam(self.act(self.bn(self.dw(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fuse_p2_td = FastNormalizedFusion(2)
        self.smooth_p2_td = DWSmooth(channels)
        self.fuse_p1_td = FastNormalizedFusion(2)
        self.smooth_p1_td = DWSmooth(channels)
        self.fuse_p2_out = FastNormalizedFusion(3)
        self.smooth_p2_out = DWSmooth(channels)
        self.fuse_p3_out = FastNormalizedFusion(2)
        self.smooth_p3_out = DWSmooth(channels)

    def forward(self, P1, P2, P3):
        P3_td = P3
        P2_td = self.smooth_p2_td(self.fuse_p2_td([P2, self.upsample(P3_td)]))
        P1_td = self.smooth_p1_td(self.fuse_p1_td([P1, self.upsample(P2_td)]))
        P1_out = P1_td
        P2_out = self.smooth_p2_out(self.fuse_p2_out([P2_td, P2, self.downsample(P1_out)]))
        P3_out = self.smooth_p3_out(self.fuse_p3_out([P3, self.downsample(P2_out)]))
        return P1_out, P2_out, P3_out


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_layers=2):
        super().__init__()
        c1, c2, c3 = in_channels_list
        self.lateral_p1 = nn.Conv2d(c1, out_channels, 1, bias=False)
        self.lateral_p2 = nn.Conv2d(c2, out_channels, 1, bias=False)
        self.lateral_p3 = nn.Conv2d(c3, out_channels, 1, bias=False)
        self.layers = nn.ModuleList([BiFPNLayer(out_channels) for _ in range(num_layers)])

    def forward(self, P1, P2, P3):
        P1 = self.lateral_p1(P1)
        P2 = self.lateral_p2(P2)
        P3 = self.lateral_p3(P3)
        for layer in self.layers:
            P1, P2, P3 = layer(P1, P2, P3)
        return P1


class SingleLateral(nn.Module):
    """BiFPN replacement when use_bifpn=False: one 1x1 projection of P1."""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels_list[0], out_channels, 1, bias=False)

    def forward(self, P1, P2, P3):
        return self.proj(P1)


class ConvMixerBlock(nn.Module):
    """Residual depthwise-separable block; zero-init BN => identity at init."""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, 1, pad, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        nn.init.zeros_(self.dw_bn.weight)
        nn.init.zeros_(self.pw_bn.weight)

    def forward(self, x):
        x = x + self.act(self.dw_bn(self.dw(x)))
        x = x + self.act(self.pw_bn(self.pw(x)))
        return x


def make_coord_grids(H, W, device, dtype):
    """Normalised [0,1] cell-centre coordinates, shapes [1,1,H,1] / [1,1,1,W]."""
    ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1) / max(H - 1, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W) / max(W - 1, 1)
    return xs, ys


def local_softargmax(hm, temperature, radius):
    """Soft-argmax over a (2r+1)^2 window around the arg-max cell, so a second peak cannot drag it.

    hm is [B,1,H,W] logits; radius >= H makes it global.
    Returns coords [B,2] (x, y) in [0,1], prob [B,1,H,W], peak_logit [B,1].
    """
    B, _, H, W = hm.shape
    flat = hm.reshape(B, H * W)
    idx = flat.argmax(dim=1)                       # [B]
    peak_logit = flat.gather(1, idx.view(B, 1))    # [B,1]
    py = torch.div(idx, W, rounding_mode='floor').to(hm.dtype).view(B, 1, 1)
    px = (idx % W).to(hm.dtype).view(B, 1, 1)

    ys = torch.arange(H, device=hm.device, dtype=hm.dtype).view(1, H, 1)
    xs = torch.arange(W, device=hm.device, dtype=hm.dtype).view(1, 1, W)
    inside = ((ys - py).abs() <= radius) & ((xs - px).abs() <= radius)   # [B,H,W]

    logits = hm.reshape(B, H, W) * temperature
    neg = torch.full_like(logits, -1e4)
    logits = torch.where(inside, logits, neg)
    prob = F.softmax(logits.reshape(B, H * W), dim=1).reshape(B, 1, H, W)

    xg, yg = make_coord_grids(H, W, hm.device, hm.dtype)
    cx = (prob * xg).sum(dim=[2, 3])
    cy = (prob * yg).sum(dim=[2, 3])
    return torch.cat([cx, cy], dim=1), prob, peak_logit


def sample_local_patch(feat, center, grid, step):
    """grid_sample a (grid x grid) neighbourhood around `center` (normalised
    [0,1] xy), spaced `step` feature cells apart. Returns [B,C,grid,grid]."""
    B = feat.size(0)
    _, _, H, W = feat.shape
    half = grid // 2
    offs = torch.arange(-half, half + 1, device=feat.device, dtype=feat.dtype) * step
    ox = offs.view(1, 1, -1).expand(1, grid, grid)
    oy = offs.view(1, -1, 1).expand(1, grid, grid)
    # align_corners=True maps cell k to k/(N-1), matching make_coord_grids
    c = center * 2.0 - 1.0
    gx = c[:, 0].view(B, 1, 1) + ox * (2.0 / max(W - 1, 1))
    gy = c[:, 1].view(B, 1, 1) + oy * (2.0 / max(H - 1, 1))
    g = torch.stack([gx, gy], dim=-1)   # [B,grid,grid,2]
    return F.grid_sample(feat, g, mode='bilinear', padding_mode='border', align_corners=True)


def sample_box_grid(feat, box, grid, expand=1.0):
    """RoIAlign-style bilinear sample of a (grid x grid) lattice spanning the
    box (cx,cy,w,h in [0,1]), expanded by `expand`. Returns [B,C,grid,grid]."""
    B = feat.size(0)
    t = torch.linspace(-0.5, 0.5, grid, device=feat.device, dtype=feat.dtype)
    tx = t.view(1, 1, grid).expand(1, grid, grid)
    ty = t.view(1, grid, 1).expand(1, grid, grid)
    cx, cy, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    gx = cx.view(B, 1, 1) + tx * (w.view(B, 1, 1) * expand)
    gy = cy.view(B, 1, 1) + ty * (h.view(B, 1, 1) * expand)
    g = torch.stack([gx, gy], dim=-1) * 2.0 - 1.0
    return F.grid_sample(feat, g, mode='bilinear', padding_mode='border', align_corners=True)


class HeatmapHead(nn.Module):
    """Visual heatmap logits plus an optional additive geometry term (log-space product of evidence and prior)."""
    def __init__(self, in_channels, cond_channels=0, bottleneck=48):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1, bias=True),
        )
        # Not exactly zero: a flat map would put the arg-max on the same corner cell for every image.
        nn.init.normal_(self.visual[-1].weight, std=0.01)
        nn.init.zeros_(self.visual[-1].bias)

        self.cond_channels = cond_channels
        if cond_channels > 0:
            self.cond = nn.Sequential(
                nn.Conv2d(in_channels + cond_channels, bottleneck, 1, bias=False),
                nn.BatchNorm2d(bottleneck),
                nn.SiLU(inplace=True),
                nn.Conv2d(bottleneck, 1, 3, 1, 1, bias=True),
            )
            nn.init.zeros_(self.cond[-1].weight)
            nn.init.zeros_(self.cond[-1].bias)
        else:
            self.cond = None

    def forward(self, feat, cond=None):
        hm = self.visual(feat)
        geo = None
        if self.cond is not None and cond is not None:
            geo = self.cond(torch.cat([feat, cond], dim=1))
            hm = hm + geo
        return hm, geo


class LocalCollapse(nn.Module):
    """Collapse a [B,C,k,k] neighbourhood into a vector (valid depthwise kxk
    -> 1x1, then pointwise projection)."""
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.collapse = nn.Conv2d(in_channels, in_channels, k, groups=in_channels, bias=False)
        self.collapse_bn = nn.BatchNorm2d(in_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.collapse_bn(self.collapse(x)))
        x = self.act(self.proj_bn(self.proj(x)))
        return x.flatten(1)


def mlp(in_dim, hidden, out_dim, out_act=None):
    layers = [nn.Linear(in_dim, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, out_dim)]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class SpatialNet(nn.Module):
    """SpatialNet pointing model.

    Outputs (dict):
      hand        [B,4]   wearer hand box (cx,cy,w,h) in [0,1]
      fingertip   [B,2]   (x,y) in [0,1]
      angle       [B,2]   (sin, cos) unit vector
      roi         [B,4]   target box (cx,cy,w,h)
      hand_conf   [B,1]   raw logit  P(wearer hand present)
      point_conf  [B,1]   raw logit  P(pointing | hand)
      roi_conf    [B,1]   raw logit  P(target | pointing)
      hand_hm / ft_hm / roi_hm  [B,1,G,G]  heatmap logits (for losses / viz)
      roi_geo     [B,1,G,G]  additive geometric gating term (viz)

    Constructor flags disable individual mechanisms:
      use_bifpn, use_coordconv, mixer_depth, anchor_chain, local_softargmax,
      roi_geometry ('additive' | 'multiplicative' | 'none'), ft_box_prior
    Experiment flags (defaults reproduce the paper's model):
      roi_forward_only  zero the additive ray prior behind the fingertip (t < 0)
      roi_size_step     spacing, in feature cells, of the 3x3 sample used for the target size
    """

    def __init__(self, input_size=None, load_pretrained=True, pretrained_path=None,
                 use_bifpn=True, use_coordconv=True, mixer_depth=None,
                 anchor_chain=True, local_softargmax=True,
                 roi_geometry='additive', ft_box_prior=True,
                 roi_forward_only=False, roi_size_step=1.0):
        super().__init__()
        self.input_size = input_size or cfg.IMG_SIZE
        self.grid = self.input_size // 8
        fpn_out = cfg.FPN_OUT_CHANNELS
        D = cfg.HEAD_DIM
        self.D = D
        self.use_coordconv = use_coordconv
        self.anchor_chain = anchor_chain
        self.local_sa = local_softargmax
        assert roi_geometry in ('additive', 'multiplicative', 'none')
        self.roi_geometry = roi_geometry
        self.roi_forward_only = bool(roi_forward_only)
        self.roi_size_step = float(roi_size_step)
        self.ft_box_prior = ft_box_prior and anchor_chain
        mixer_depth = cfg.CONV_MIXER_DEPTH if mixer_depth is None else mixer_depth

        self.backbone = ShuffleNetV2Backbone(load_pretrained, pretrained_path)
        self.neck = BiFPN([48, 96, 192], fpn_out, num_layers=2) if use_bifpn \
            else SingleLateral([48, 96, 192], fpn_out)
        in_reduce = fpn_out + (2 if use_coordconv else 0)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_reduce, in_reduce, 3, 1, 1, groups=in_reduce, bias=False),
            nn.BatchNorm2d(in_reduce),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_reduce, D, 1, bias=False),
            nn.BatchNorm2d(D),
            nn.SiLU(inplace=True),
        )
        self.mixer = nn.Sequential(*[ConvMixerBlock(D, cfg.CONV_MIXER_KERNEL)
                                     for _ in range(mixer_depth)])

        k = cfg.LOCAL_GRID
        Lp = cfg.LOCAL_PROJ_DIM
        Hd = cfg.HAND_DESC_DIM
        self.hand_hm = HeatmapHead(D, cond_channels=0)
        self.hand_size_mix = LocalCollapse(D, Lp, k)
        self.hand_size = mlp(Lp, Lp // 2, 2, nn.Sigmoid())
        self.hand_desc_mix = LocalCollapse(D, Hd, cfg.HAND_ROI_GRID)
        self.hand_desc = nn.Sequential(nn.Linear(Hd + 4, Hd), nn.SiLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.hand_conf = mlp(D + 1 + (Hd if anchor_chain else 0), 64, 1)

        self.ft_hm = HeatmapHead(D, cond_channels=2 if anchor_chain else 0,
                                 bottleneck=cfg.COND_BOTTLENECK)
        self.ft_scale = nn.Parameter(torch.tensor(5.0))
        self.ft_local_mix = LocalCollapse(D, Lp, k)

        if anchor_chain:
            self.angle = mlp(Hd + Lp, 128, 2, nn.Tanh())
            self.point_conf = mlp(Hd, 64, 1)
            self.roi_conf = mlp(Hd + Lp + 1, 64, 1)
        else:
            # without the anchor chain, heads read the globally pooled vector
            self.angle = mlp(D, 128, 2, nn.Tanh())
            self.point_conf = mlp(D, 64, 1)
            self.roi_conf = mlp(D + 1, 64, 1)

        # 6 cond channels: fingertip offsets (dx,dy), along-ray t, perpendicular |s|, direction (cos,sin)
        self.roi_hm = HeatmapHead(D, cond_channels=6 if roi_geometry == 'additive' else 0,
                                  bottleneck=cfg.COND_BOTTLENECK)
        self.roi_scale = nn.Parameter(torch.tensor(5.0))
        self.roi_size_mix = LocalCollapse(D, Lp, k)
        self.roi_size = mlp(Lp, Lp // 2, 2, nn.Sigmoid())
        if roi_geometry == 'multiplicative':
            self.roi_mask = nn.Sequential(
                nn.Conv2d(D + 4, cfg.COND_BOTTLENECK, 1, bias=False),
                nn.BatchNorm2d(cfg.COND_BOTTLENECK), nn.SiLU(inplace=True),
                nn.Conv2d(cfg.COND_BOTTLENECK, 1, 3, 1, 1, bias=True), nn.Sigmoid())

        self._init_linear()

    def _init_linear(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _force_anchor(pred, anchors, key):
        """Scheduled anchor forcing: swap in the GT anchor where mix=1. anchors[key] = (target [B,2], mix [B,1])."""
        if not anchors or key not in anchors:
            return pred
        tgt, mix = anchors[key]
        return mix * tgt.to(pred.dtype) + (1.0 - mix) * pred

    def _softargmax(self, hm, key):
        T = cfg.SOFTARGMAX_TEMP[key]
        r = cfg.SOFTARGMAX_RADIUS[key] if self.local_sa else 10_000
        return local_softargmax(hm, T, r)

    @staticmethod
    def _append_coords(x):
        B, C, H, W = x.shape
        yg = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xg = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, xg, yg], dim=1)

    @staticmethod
    def _offset_channels(B, H, W, point, scale, device, dtype):
        """Point-relative offset channels (Δx, Δy) for every cell."""
        xg, yg = make_coord_grids(H, W, device, dtype)
        dx = (xg.expand(B, 1, H, W) - point[:, 0].view(B, 1, 1, 1)) * scale
        dy = (yg.expand(B, 1, H, W) - point[:, 1].view(B, 1, 1, 1)) * scale
        return dx, dy

    @staticmethod
    def _box_prior(B, H, W, box, dilation, strength, device, dtype):
        """Additive logit prior: 0 inside the dilated box, -strength per half-box-width beyond the edge."""
        xg, yg = make_coord_grids(H, W, device, dtype)
        cx, cy, w, h = [box[:, i].view(B, 1, 1, 1) for i in range(4)]
        hw = (w * dilation * 0.5).clamp(min=1.0 / max(W - 1, 1))
        hh = (h * dilation * 0.5).clamp(min=1.0 / max(H - 1, 1))
        ex = F.relu((xg.expand(B, 1, H, W) - cx).abs() / hw - 1.0)
        ey = F.relu((yg.expand(B, 1, H, W) - cy).abs() / hh - 1.0)
        return -strength * (ex + ey)

    def forward(self, x, return_all=True, anchors=None):
        B = x.size(0)
        P1, P2, P3 = self.backbone(x)
        S = self.neck(P1, P2, P3)
        if self.use_coordconv:
            S = self._append_coords(S)
        S = self.reduce(S)
        S = self.mixer(S)                       # [B,D,G,G]
        _, _, H, W = S.shape
        dev, dt = S.device, S.dtype
        step = 1.0  # one feature cell between local taps

        hand_hm, _ = self.hand_hm(S)
        hand_c, hand_prob, hand_peak = self._softargmax(hand_hm, 'hand')
        hand_c = self._force_anchor(hand_c, anchors, 'hand')
        hand_wh = self.hand_size(self.hand_size_mix(sample_local_patch(S, hand_c, cfg.LOCAL_GRID, step)))
        hand_box = torch.cat([hand_c, hand_wh], dim=1)

        pooled = self.gap(S).flatten(1)          # [B,D]
        if self.anchor_chain:
            desc = self.hand_desc_mix(sample_box_grid(S, hand_box, cfg.HAND_ROI_GRID, cfg.HAND_ROI_EXPAND))
            v_h = self.hand_desc(torch.cat([desc, hand_box], dim=1))   # [B,Hd]
            # instance features let a frame with only other people's hands answer "no"
            hand_conf = self.hand_conf(torch.cat([pooled, hand_peak, v_h], dim=1))
            # instance only, so distractor hands cannot drive this gate
            point_conf = self.point_conf(v_h)
        else:
            v_h = None
            hand_conf = self.hand_conf(torch.cat([pooled, hand_peak], dim=1))
            point_conf = self.point_conf(pooled)

        if self.anchor_chain:
            dxh, dyh = self._offset_channels(B, H, W, hand_c, self.ft_scale, dev, dt)
            ft_hm, _ = self.ft_hm(S, torch.cat([dxh, dyh], dim=1))
            if self.ft_box_prior:
                ft_hm = ft_hm + self._box_prior(B, H, W, hand_box.detach(),
                                                cfg.FT_BOX_PRIOR_DILATION,
                                                cfg.FT_BOX_PRIOR_STRENGTH / cfg.SOFTARGMAX_TEMP['fingertip'],
                                                dev, dt)
        else:
            ft_hm, _ = self.ft_hm(S)
        ft, ft_prob, _ = self._softargmax(ft_hm, 'fingertip')
        ft = self._force_anchor(ft, anchors, 'fingertip')
        v_f = self.ft_local_mix(sample_local_patch(S, ft, cfg.LOCAL_GRID, step))  # [B,Lp]

        if self.anchor_chain:
            angle = self.angle(torch.cat([v_h, v_f], dim=1))
        else:
            angle = self.angle(pooled)
        angle = F.normalize(angle, p=2, dim=1)      # (sin, cos)

        roi_geo = None
        if self.roi_geometry != 'none':
            dxf, dyf = self._offset_channels(B, H, W, ft, self.roi_scale, dev, dt)
            dir_x = angle[:, 1].view(B, 1, 1, 1).expand(B, 1, H, W)
            dir_y = angle[:, 0].view(B, 1, 1, 1).expand(B, 1, H, W)
            t_along = dxf * dir_x + dyf * dir_y            # signed distance along the ray
            s_perp = (dxf * dir_y - dyf * dir_x).abs()     # distance from the ray line
            cond = torch.cat([dxf, dyf, t_along, s_perp, dir_x, dir_y], dim=1)
            if self.roi_geometry == 'additive':
                roi_hm, roi_geo = self.roi_hm(S, cond)
                if self.roi_forward_only:
                    # the learned prior may only act in front of the fingertip
                    fwd = (t_along >= 0).to(dt)
                    roi_hm = roi_hm - roi_geo * (1.0 - fwd)
                    roi_geo = roi_geo * fwd
            else:  # multiplicative sigmoid mask on signed logits
                roi_hm, _ = self.roi_hm(S)
                roi_geo = self.roi_mask(torch.cat([S, cond[:, :2], cond[:, 4:]], dim=1))
                roi_hm = roi_hm * roi_geo
        else:
            roi_hm, _ = self.roi_hm(S)
        roi_c, roi_prob, roi_peak = self._softargmax(roi_hm, 'roi')
        roi_c = self._force_anchor(roi_c, anchors, 'roi')
        roi_wh = self.roi_size(self.roi_size_mix(sample_local_patch(S, roi_c, cfg.LOCAL_GRID, self.roi_size_step)))
        roi_box = torch.cat([roi_c, roi_wh], dim=1)

        if self.anchor_chain:
            roi_conf = self.roi_conf(torch.cat([v_h, v_f, roi_peak], dim=1))
        else:
            roi_conf = self.roi_conf(torch.cat([pooled, roi_peak], dim=1))

        out = {
            'hand': hand_box,
            'fingertip': ft,
            'angle': angle,
            'roi': roi_box,
            'hand_conf': hand_conf,
            'point_conf': point_conf,
            'roi_conf': roi_conf,
        }
        if return_all:
            out.update({
                'hand_hm': hand_hm, 'ft_hm': ft_hm, 'roi_hm': roi_hm,
                'hand_prob': hand_prob, 'ft_prob': ft_prob, 'roi_prob': roi_prob,
                'roi_geo': roi_geo,
            })
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        for p in self.parameters():
            if id(p) not in backbone_ids:
                yield p


class ExportWrapper(nn.Module):
    """Flat-tuple output for ONNX export (no dicts / None)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        o = self.model(x, return_all=False)
        return (o['hand'], o['fingertip'], o['angle'], o['roi'],
                o['hand_conf'], o['point_conf'], o['roi_conf'])


if __name__ == "__main__":
    print("=" * 60)
    print("SpatialNet Model Test")
    print("=" * 60)
    model = SpatialNet(load_pretrained=False)
    total = model.count_parameters()
    bb = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Total params:    {total:,}")
    print(f"  Backbone params: {bb:,}")
    print(f"  Head params:     {total - bb:,}")
    print(f"  FP32 size:       {total * 4 / 1024 / 1024:.2f} MB")
    bids = {id(p) for p in model.get_backbone_params()}
    hids = {id(p) for p in model.get_head_params()}
    assert bids.isdisjoint(hids) and (bids | hids) == {id(p) for p in model.parameters()}
    x = torch.randn(2, 3, model.input_size, model.input_size)
    model.eval()
    with torch.no_grad():
        out = model(x)
    for k in ['hand', 'fingertip', 'angle', 'roi', 'hand_conf', 'point_conf', 'roi_conf', 'hand_hm']:
        print(f"  {k:<11} {tuple(out[k].shape)}")
    # gradient check through the chain
    model.train()
    out = model(x)
    loss = out['roi'].sum() + out['angle'].sum() + out['fingertip'].sum() + out['hand'].sum()
    loss.backward()
    g = model.hand_hm.visual[-1].weight.grad
    print(f"  grad reaches hand heatmap from ROI/angle/ft/hand losses: {g is not None and g.abs().sum() > 0}")
    g2 = model.ft_hm.visual[-1].weight.grad
    print(f"  grad reaches fingertip heatmap from ROI/angle losses:    {g2 is not None and g2.abs().sum() > 0}")
    print("[OK] All tests passed!")
