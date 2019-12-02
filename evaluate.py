r"""Runs Hyperpixel Flow framework"""

import argparse
import datetime
import os

from torch.utils.data import DataLoader
import torch

from model import hpflow, geometry, evaluation, util
from data import download


def run(datapath, benchmark, backbone, thres, alpha, hyperpixel,
        logpath, beamsearch, model=None, dataloader=None, visualize=False):
    r"""Runs Hyperpixel Flow framework"""

    # 1. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not beamsearch:
        cur_datetime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logfile = os.path.join('logs', logpath + cur_datetime + '.log')
        util.init_logger(logfile)
        util.log_args(args)
        if visualize: os.mkdir(logfile + 'vis')


    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataloader is None:
        download.download_dataset(os.path.abspath(datapath), benchmark)
        split = 'val' if beamsearch else 'test'
        dset = download.load_dataset(benchmark, datapath, thres, device, split)
        dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # 3. Model initialization
    if model is None:
        model = hpflow.HyperpixelFlow(backbone, hyperpixel, benchmark, device)
    else:
        model.hyperpixel_ids = util.parse_hyperpixel(hyperpixel)

    # 4. Evaluator initialization
    evaluator = evaluation.Evaluator(benchmark, device)

    for idx, data in enumerate(dataloader):

        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
        data['alpha'] = alpha

        # b) Feed a pair of images to Hyperpixel Flow model
        with torch.no_grad():
            confidence_ts, src_box, trg_box = model(data['src_img'], data['trg_img'])

        # c) Predict key-points & evaluate performance
        prd_kps = geometry.predict_kps(src_box, trg_box, data['src_kps'], confidence_ts)
        evaluator.evaluate(prd_kps, data)

        # d) Log results
        if not beamsearch:
            evaluator.log_result(idx, data=data)
        if visualize:
            vispath = os.path.join(logfile + 'vis', '%03d_%s_%s' % (idx, data['src_imname'][0], data['trg_imname'][0]))
            util.visualize_prediction(data['src_kps'].t().cpu(), prd_kps.t().cpu(),
                                      data['src_img'], data['trg_img'], vispath)
    if beamsearch:
        return (sum(evaluator.eval_buf['pck']) / len(evaluator.eval_buf['pck'])) * 100.
    else:
        evaluator.log_result(len(dset), data=None, average=True)


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Hyperpixel Flow in pytorch')
    parser.add_argument('--datapath', type=str, default='../Datasets_HPF')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hyperpixel', type=str, default='')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    run(datapath=args.datapath, benchmark=args.dataset, backbone=args.backbone, thres=args.thres, alpha=args.alpha,
        hyperpixel=args.hyperpixel, logpath=args.logpath, beamsearch=False, visualize=args.visualize)
