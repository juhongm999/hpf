"""Provides memory buffer and logger for evaluation"""

import logging

from skimage import draw
import numpy as np
import torch


class Evaluator:
    r"""To evaluate and log evaluation metrics: PCK, LT-ACC, IoU"""
    def __init__(self, benchmark, device):
        r"""Constructor for Evaluator"""
        self.eval_buf = {
            'pfwillow': {'pck': [], 'cls_pck': dict()},
            'pfpascal': {'pck': [], 'cls_pck': dict()},
            'spair':    {'pck': [], 'cls_pck': dict()},
            'caltech':  {'ltacc': [], 'iou': []}
        }

        self.eval_funct = {
            'pfwillow': self.eval_pck,
            'pfpascal': self.eval_pck,
            'spair': self.eval_pck,
            'caltech': self.eval_caltech
        }

        self.log_funct = {
            'pfwillow': self.log_pck,
            'pfpascal': self.log_pck,
            'spair': self.log_pck,
            'caltech': self.log_caltech
        }

        self.eval_buf = self.eval_buf[benchmark]
        self.eval_funct = self.eval_funct[benchmark]
        self.log_funct = self.log_funct[benchmark]
        self.benchmark = benchmark
        self.device = device

    def evaluate(self, prd_kps, data):
        r"""Compute desired evaluation metric"""
        return self.eval_funct(prd_kps, data)

    def log_result(self, idx, data, average=False):
        r"""Print results: PCK, or LT-ACC & IoU """
        return self.log_funct(idx, data, average)

    def eval_pck(self, prd_kps, data):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""
        pckthres = data['pckthres'][0] * data['trg_intratio']
        ncorrt = correct_kps(data['trg_kps'].cuda(), prd_kps, pckthres, data['alpha'])
        pair_pck = int(ncorrt) / int(data['trg_kps'].size(1))

        self.eval_buf['pck'].append(pair_pck)

        if self.eval_buf['cls_pck'].get(data['pair_class'][0]) is None:
            self.eval_buf['cls_pck'][data['pair_class'][0]] = []
        self.eval_buf['cls_pck'][data['pair_class'][0]].append(pair_pck)

    def log_pck(self, idx, data, average):
        r"""Log percentage of correct key-points (PCK)"""
        if average:
            pck = sum(self.eval_buf['pck']) / len(self.eval_buf['pck'])
            for cls in self.eval_buf['cls_pck']:
                cls_avg = sum(self.eval_buf['cls_pck'][cls]) / len(self.eval_buf['cls_pck'][cls])
                logging.info('%15s: %3.3f' % (cls, cls_avg))
            logging.info(' * Average: %3.3f' % pck)

            return pck

        logging.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                     (idx + 1,
                      data['datalen'],
                      self.eval_buf['pck'][idx],
                      sum(self.eval_buf['pck']) / len(self.eval_buf['pck']),
                      data['pair_class'][0]))
        return None

    def eval_caltech(self, prd_kps, data):
        r"""Compute LT-ACC and IoU based on transferred points"""
        imsize = list(data['trg_img'].size())[1:]
        trg_xstr, trg_ystr = pts2ptstr(data['trg_kps'])
        trg_mask = ptstr2mask(trg_xstr, trg_ystr, imsize[0], imsize[1])
        prd_xstr, pred_ystr = pts2ptstr(prd_kps)
        prd_mask = ptstr2mask(prd_xstr, pred_ystr, imsize[0], imsize[1])

        lt_acc = label_transfer_accuracy(prd_mask, trg_mask)
        iou = intersection_over_union(prd_mask, trg_mask)

        self.eval_buf['ltacc'].append(lt_acc)
        self.eval_buf['iou'].append(iou)

    def log_caltech(self, idx, data, average):
        r"""Log Caltech-101 dataset evaluation metrics: LT-ACC and IoU"""
        if average:
            lt_acc = sum(self.eval_buf['ltacc']) / len(self.eval_buf['ltacc'])
            segiou = sum(self.eval_buf['iou']) / len(self.eval_buf['iou'])
            logging.info(' * Average LT-ACC: %3.2f' % lt_acc)
            logging.info(' * Average IoU: %3.2f' % segiou)

            return lt_acc, segiou

        logging.info('[%5d/%5d]: \t [LT-ACC/IoU: %5.2f/%.2f]\t[Average: %5.2f/%.2f]' %
                     (idx + 1,
                      data['datalen'],
                      self.eval_buf['ltacc'][idx],
                      self.eval_buf['iou'][idx],
                      sum(self.eval_buf['ltacc']) / len(self.eval_buf['ltacc']),
                      sum(self.eval_buf['iou']) / len(self.eval_buf['iou'])))
        return None


def correct_kps(trg_kps, prd_kps, pckthres, alpha=0.1):
    r"""Compute the number of correctly transferred key-points"""
    l2dist = torch.pow(torch.sum(torch.pow(trg_kps - prd_kps, 2), 0), 0.5)
    thres = pckthres.expand_as(l2dist).float()
    correct_pts = torch.le(l2dist, thres * alpha)

    return torch.sum(correct_pts)


def pts2ptstr(pts):
    r"""Convert tensor of points to string"""
    x_str = str(list(pts[0].cpu().numpy()))
    x_str = x_str[1:len(x_str)-1]
    y_str = str(list(pts[1].cpu().numpy()))
    y_str = y_str[1:len(y_str)-1]

    return x_str, y_str


def pts2mask(x_pts, y_pts, shape):
    r"""Build a binary mask tensor base on given xy-points"""
    x_idx, y_idx = draw.polygon(x_pts, y_pts, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[x_idx, y_idx] = True

    return mask


def ptstr2mask(x_str, y_str, out_h, out_w):
    r"""Convert xy-point mask (string) to tensor mask"""
    x_pts = np.fromstring(x_str, sep=',')
    y_pts = np.fromstring(y_str, sep=',')
    mask_np = pts2mask(y_pts, x_pts, [out_h, out_w])
    mask = torch.tensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).float()

    return mask


def intersection_over_union(mask1, mask2):
    r"""Computes IoU between two masks"""
    rel_part_weight = torch.sum(torch.sum(mask2.gt(0.5).float(), 2, True), 3, True) / \
                      torch.sum(mask2.gt(0.5).float())
    part_iou = torch.sum(torch.sum((mask1.gt(0.5) & mask2.gt(0.5)).float(), 2, True), 3, True) / \
               torch.sum(torch.sum((mask1.gt(0.5) | mask2.gt(0.5)).float(), 2, True), 3, True)
    weighted_iou = torch.sum(torch.mul(rel_part_weight, part_iou)).item()

    return weighted_iou


def label_transfer_accuracy(mask1, mask2):
    r"""LT-ACC measures the overlap with emphasis on the background class"""
    return torch.mean((mask1.gt(0.5) == mask2.gt(0.5)).double()).item()
