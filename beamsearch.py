r"""Beam search for hyperpixel layers"""

import datetime
import argparse
import logging
import time
import os

from torch.utils.data import DataLoader
import torch

from data import dataset, download
from model import hpflow
from model import util
import evaluate


def parse_layers(layer_ids):
    r"""Parse list of layer ids (int) into string format"""
    layer_str = ''.join(list(map(lambda x: '%d,' % x, layer_ids)))[:-1]
    layer_str = '(' + layer_str + ')'
    return layer_str


def find_topk(membuf, kval):
    r"""Return top-k performance along with layer combinations"""
    membuf.sort(key=lambda x: x[0], reverse=True)
    return membuf[:kval]


def log_evaluation(layers, score, elapsed):
    r"""Log a single evaluation result"""
    logging.info('%20s: %4.2f %% %5.1f sec' % (layers, score, elapsed))


def log_selected(depth, membuf_topk):
    r"""Log selected layers at each depth"""
    logging.info(' ===================== Depth %d =====================' % depth)
    for score, layers in membuf_topk:
        logging.info('%20s: %4.2f %%' % (layers, score))
    logging.info(' ====================================================')


def beamsearch_hp(datapath, benchmark, backbone, thres, alpha, logpath,
                  candidate_base, candidate_layers, beamsize, maxdepth):
    r"""Implementation of beam search for hyperpixel layers"""

    # 1. Model, and dataset initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = hpflow.HyperpixelFlow(backbone, '0', benchmark, device)
    download.download_dataset(os.path.abspath(datapath), benchmark)
    dset = download.load_dataset(benchmark, datapath, thres, device, 'val')
    dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 2. Search for the k-best base layers
    membuf_cand = []
    for base in candidate_base:
        start = time.time()
        hyperpixel = parse_layers(base)
        score = evaluate.run(datapath, benchmark, backbone, thres, alpha,
                             hyperpixel, logpath, True, model, dataloader)
        log_evaluation(base, score, time.time() - start)
        membuf_cand.append((score, base))
    membuf_topk = find_topk(membuf_cand, beamsize)
    score_sel, layer_sel = find_topk(membuf_cand, 1)[0]
    log_selected(0, membuf_topk)

    # 3. Proceed iterative search
    for depth in range(1, maxdepth):
        membuf_cand = []
        for _, test_layer in membuf_topk:
            for cand_layer in candidate_layers:
                if cand_layer not in test_layer and cand_layer > min(test_layer):
                    start = time.time()
                    test_layers = sorted(test_layer + [cand_layer])
                    if test_layers in list(map(lambda x: x[1], membuf_cand)):
                        break
                    hyperpixel = parse_layers(test_layers)
                    score = evaluate.run(datapath, benchmark, backbone, thres, alpha,
                                         hyperpixel, logpath, True, model, dataloader)

                    log_evaluation(test_layers, score, time.time() - start)
                    membuf_cand.append((score, test_layers))

        membuf_topk = find_topk(membuf_cand, beamsize)
        score_tmp, layer_tmp = find_topk(membuf_cand, 1)[0]

        if score_tmp > score_sel:
            layer_sel = layer_tmp
            score_sel = score_tmp
        log_selected(depth, membuf_topk)

    # 4. Log best layer combination and validation performance
    logging.info('\nBest layers, score: %s %5.3f' % (layer_sel, score_sel))

    return layer_sel


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Beam search for hyperpixel layers')
    parser.add_argument('--datapath', type=str, default='../Datasets_HPF')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--thres', type=str, default='bbox', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--beamsize', type=int, default=4)
    parser.add_argument('--maxdepth', type=int, default=8)
    args = parser.parse_args()

    # 1. Candidate layers for hyperpixel initialization
    n_layers = {'resnet50': 17, 'resnet101': 34, 'fcn101': 34}
    candidate_base = [[i] for i in range(args.beamsize)]
    candidate_layers = list(range(n_layers[args.backbone]))

    # 2. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    cur_datetime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
    logfile = os.path.join('logs', args.logpath + cur_datetime + '.log')
    util.init_logger(logfile)
    util.log_args(args)

    # 3. Run beam search
    logging.info('Beam search on \'%s validation split\' with \'%s\' backbone...\n' %
                 (args.dataset, args.backbone))
    layer_sel = beamsearch_hp(args.datapath, args.dataset, args.backbone, args.thres, args.alpha,
                              args.logpath, candidate_base, candidate_layers, args.beamsize, args.maxdepth)
