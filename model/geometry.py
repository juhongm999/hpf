"""Provides functions that manipulate boxes and points"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util


def center(box):
    r"""Calculates centers, (x, y), of box (N, 4)"""
    x_center = box[:, 0] + (box[:, 2] - box[:, 0]) // 2
    y_center = box[:, 1] + (box[:, 3] - box[:, 1]) // 2
    return torch.stack((x_center, y_center)).t().to(box.device)


def receptive_fields(rfsz, jsz, feat_size):
    r"""Returns a set of receptive fields (N, 4)"""
    width = feat_size[2]
    height = feat_size[1]

    feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
    feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

    box = torch.zeros(feat_ids.size()[0], 4)
    box[:, 0] = feat_ids[:, 1] * jsz - rfsz // 2
    box[:, 1] = feat_ids[:, 0] * jsz - rfsz // 2
    box[:, 2] = feat_ids[:, 1] * jsz + rfsz // 2
    box[:, 3] = feat_ids[:, 0] * jsz + rfsz // 2

    return box


def prune_margin(receptive_box, imsize, threshold=0):
    r"""Remove receptive fields on the margin of the image"""
    im_width = imsize[1]
    im_height = imsize[0]

    xmin = util.where(receptive_box[:, 0] >= 0 - threshold)
    ymin = util.where(receptive_box[:, 1] >= 0 - threshold)
    xmax = util.where(receptive_box[:, 2] < im_width + threshold)
    ymax = util.where(receptive_box[:, 3] < im_height + threshold)

    val1 = util.intersect1d(xmin, ymin)
    val2 = util.intersect1d(xmax, ymax)
    valid_ids = torch.sort(torch.unique(util.intersect1d(val1, val2)))[0]

    pruned_receptive_box = receptive_box[valid_ids]

    return pruned_receptive_box, valid_ids


def predict_kps(src_box, trg_box, src_kps, confidence_ts):
    r"""Transfer keypoints by nearest-neighbour assignment"""

    # 1. Prepare geometries & argmax target indices
    _, trg_argmax_idx = torch.max(confidence_ts, dim=1)
    src_geomet = src_box[:, :2].unsqueeze(0).repeat(len(src_kps.t()), 1, 1)
    trg_geomet = trg_box[:, :2].unsqueeze(0).repeat(len(src_kps.t()), 1, 1)

    # 2. Retrieve neighbouring source boxes that cover source key-points
    src_nbr_onehot, n_neighbours = neighbours(src_box, src_kps)

    # 3. Get displacements from source neighbouring box centers to each key-point
    src_displacements = src_kps.t().unsqueeze(1).repeat(1, len(src_box), 1) - src_geomet
    src_displacements = src_displacements * src_nbr_onehot.unsqueeze(2).repeat(1, 1, 2).float()

    # 4. Transfer the neighbours based on given confidence tensor
    vector_summator = torch.zeros_like(src_geomet)
    src_idx = src_nbr_onehot.nonzero()
    trg_idx = trg_argmax_idx.index_select(dim=0, index=src_idx[:, 1])
    vector_summator[src_idx[:, 0], src_idx[:, 1]] = trg_geomet[src_idx[:, 0], trg_idx]
    vector_summator += src_displacements
    pred = (vector_summator.sum(dim=1) / n_neighbours.unsqueeze(1).repeat(1, 2).float())

    return pred.t()


def neighbours(box, kps):
    r"""Returns boxes in one-hot format that covers given keypoints"""
    box_duplicate = box.unsqueeze(2).repeat(1, 1, len(kps.t())).transpose(0, 1)
    kps_duplicate = kps.unsqueeze(1).repeat(1, len(box), 1)

    xmin = kps_duplicate[0].ge(box_duplicate[0])
    ymin = kps_duplicate[1].ge(box_duplicate[1])
    xmax = kps_duplicate[0].le(box_duplicate[2])
    ymax = kps_duplicate[1].le(box_duplicate[3])

    nbr_onehot = torch.mul(torch.mul(xmin, ymin), torch.mul(xmax, ymax)).t()
    n_neighbours = nbr_onehot.sum(dim=1)

    return nbr_onehot, n_neighbours


def gaussian2d(side=7):
    r"""Returns 2-dimensional gaussian filter"""
    dim = [side, side]

    siz = torch.LongTensor(dim)
    sig_sq = (siz.float()/2/2.354).pow(2)
    siz2 = (siz-1)/2

    x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
    y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

    gaussian = torch.exp(-(x_axis.pow(2)/2/sig_sq[0] + y_axis.pow(2)/2/sig_sq[1]))
    gaussian = gaussian / gaussian.sum()

    return gaussian


class PointTPS(nn.Module):
    def __init__(self, kps_a, kps_b, relaxation):
        super(PointTPS, self).__init__()
        l2 = PointTPS.__build_l2_mat(kps_a, kps_a)

        l2[l2 == 0] = 1
        K = l2 * torch.log(l2) * (1/2.0)
        K = K + relaxation * torch.diag(torch.ones(len(kps_a)))
        P = torch.cat([torch.ones(len(kps_a), 1), kps_a], dim=1)

        L_up = torch.cat([K, P], dim=1)
        L_down = torch.cat([P.t(), torch.zeros((3, 3))], dim=1)
        L = torch.cat([L_up, L_down], dim=0)

        v = kps_b - kps_a
        v = torch.cat([v, torch.zeros(3, 2)], dim=0)
        W = L.inverse() @ v

        self.register_buffer('kps_a', kps_a)
        self.register_buffer('W', W)

    def forward(self, pts):
        l2 = PointTPS.__build_l2_mat(self.kps_a, pts)

        l2[(l2 == 0).detach()] = 1
        v = l2 * torch.log(l2) * (1/2.0)

        o = pts[:, 0] - pts[:, 0] + 1

        x = torch.stack([o, pts[:, 0], pts[:, 1]], dim=0)
        x = torch.cat([v, x], dim=0)

        z_x = x.t() @ (self.W[:, 0])
        z_y = x.t() @ (self.W[:, 1])

        target_x = pts[:, 0] + z_x
        target_y = pts[:, 1] + z_y

        return torch.stack([target_x, target_y], dim=1)

    @staticmethod
    def __build_l2_mat(pts_a, pts_b):
        n_a = len(pts_a)
        n_b = len(pts_b)

        m1 = torch.stack([pts_a] * n_b)
        m2 = torch.stack([pts_b] * n_a)

        l2 = ((m1.transpose(0, 1) - m2) ** 2).sum(2)

        return l2


class ImageTPS(PointTPS):
    def __init__(self, kps_a, kps_b, in_size, out_size, relaxation=0):
        super().__init__(kps_b, kps_a, relaxation)
        ws = torch.linspace(0, out_size[0], steps=out_size[0])
        hs = torch.linspace(0, out_size[1], steps=out_size[1])

        self.register_buffer('hs', hs)
        self.register_buffer('ws', ws)
        self.register_buffer('in_size', torch.FloatTensor([in_size]))
        self.out_size = out_size

    def forward(self, img):
        img_h = self.out_size[1]
        img_w = self.out_size[0]
        hs = torch.stack([self.hs] * img_w, dim=1).view(-1)
        ws = torch.stack([self.ws] * img_h, dim=0).view(-1)

        pts = torch.stack([ws, hs], dim=1)
        target_pts = super().forward(pts)
        target_pts = (target_pts - (self.in_size / 2)) / (self.in_size / 2)

        warped_img = F.grid_sample(img.unsqueeze(0), target_pts.view(1, img_h, img_w, 2), mode='bilinear')
        warped_img = warped_img.squeeze(0)

        return warped_img