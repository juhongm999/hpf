"""Implementation of Hyperpixel Flow: Semantic Correspondence with Multi-layer Nueral Features"""

from functools import reduce
from operator import add

from torchvision.models import resnet
import torch.nn.functional as F
import torch
import gluoncvth as gcv

from . import geometry
from . import util
from . import rhm


class HyperpixelFlow:
    r"""Hyperpixel Flow framework"""
    def __init__(self, backbone, hyperpixel_ids, benchmark, device):
        r"""Constructor for Hyperpixel Flow framework"""

        # Feature extraction network initialization.
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            nbottlenecks = [3, 4, 23, 3]
        elif backbone == 'fcn101':
            self.backbone = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()

        # Hyperpixel id and pre-computed jump and receptive field size initialization
        # Reference: https://fomoro.com/research/article/receptive-field-calculator
        # (the jump and receptive field sizes for 'fcn101' are heuristic values)
        self.hyperpixel_ids = util.parse_hyperpixel(hyperpixel_ids)
        self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16]).to(device)
        self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)

        # Miscellaneous
        self.hsfilter = geometry.gaussian2d(7).to(device)
        self.device = device
        self.benchmark = benchmark

    def __call__(self, *args, **kwargs):
        r"""Forward pass"""
        src_hyperpixels = self.extract_hyperpixel(args[0])
        trg_hyperpixels = self.extract_hyperpixel(args[1])
        confidence_ts = rhm.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter)
        return confidence_ts, src_hyperpixels[0], trg_hyperpixels[0]

    def extract_hyperpixel(self, img):
        r"""Given image, extract desired list of hyperpixels"""
        hyperfeats, rfsz, jsz = self.extract_intermediate_feat(img.unsqueeze(0))
        hpgeometry = geometry.receptive_fields(rfsz, jsz, hyperfeats.size()).to(self.device)
        hyperfeats = hyperfeats.view(hyperfeats.size()[0], -1).t()

        # Prune boxes on margins (causes error on Caltech benchmark)
        if self.benchmark != 'caltech':
            hpgeometry, valid_ids = geometry.prune_margin(hpgeometry, img.size()[1:], jsz.float())
            hyperfeats = hyperfeats[valid_ids, :]

        return hpgeometry, hyperfeats, img.size()[1:][::-1]

    def extract_intermediate_feat(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                if hid + 1 == max(self.hyperpixel_ids):
                    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)

        return feats[0], rfsz, jsz
