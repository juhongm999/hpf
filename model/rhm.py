"""Implementation of regularized Hough matching algorithm (RHM)"""

import math

import torch.nn.functional as F
import torch

from . import geometry


def appearance_similarity(src_feats, trg_feats, cosd=3):
    r"""Semantic appearance similarity (exponentiated cosine)"""
    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)
    sim = torch.matmul(src_feats, trg_feats.t()) / \
          torch.matmul(src_feat_norms, trg_feat_norms)
    sim = torch.pow(torch.clamp(sim, min=0), cosd)

    return sim


def hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
    r"""Compute Hough space bin id for the subsequent voting procedure"""
    src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
    src_trans = geometry.center(src_box)
    trg_trans = geometry.center(trg_box)
    xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                  repeat(1, 1, len(trg_box)) + \
              trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

    bin_ids = (xy_vote / hs_cellsize).long()

    return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x


def build_hspace(src_imsize, trg_imsize, ncells):
    r"""Build Hough space where voting is done"""
    hs_width = src_imsize[0] + trg_imsize[0]
    hs_height = src_imsize[1] + trg_imsize[1]
    hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
    nbins_x = int(hs_width / hs_cellsize) + 1
    nbins_y = int(hs_height / hs_cellsize) + 1

    return nbins_x, nbins_y, hs_cellsize


def rhm(src_hyperpixels, trg_hyperpixels, hsfilter, ncells=8192):
    r"""Regularized Hough matching"""
    # Unpack hyperpixels
    src_hpgeomt, src_hpfeats, src_imsize = src_hyperpixels
    trg_hpgeomt, trg_hpfeats, trg_imsize = trg_hyperpixels

    # Prepare for the voting procedure
    votes = appearance_similarity(src_hpfeats, trg_hpfeats)
    nbins_x, nbins_y, hs_cellsize = build_hspace(src_imsize, trg_imsize, ncells)
    bin_ids = hspace_bin_ids(src_imsize, src_hpgeomt, trg_hpgeomt, hs_cellsize, nbins_x)
    hspace = src_hpgeomt.new_zeros((len(votes), nbins_y * nbins_x))

    # Proceed voting
    hbin_ids = bin_ids.add(torch.arange(0, len(votes)).to(src_hpgeomt.device).
                           mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
    hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), votes.view(-1)).view_as(hspace)
    hspace = torch.sum(hspace, dim=0)

    # Aggregate the voting results
    hspace = F.conv2d(hspace.view(1, 1, nbins_y, nbins_x),
                      hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

    return votes * torch.index_select(hspace, dim=0, index=bin_ids.view(-1)).view_as(votes)
