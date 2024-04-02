#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import json
from sklearn.metrics.cluster import (contingency_matrix,
                                     normalized_mutual_info_score)
from sklearn.metrics import (precision_score, recall_score)

__all__ = ['pairwise', 'bcubed', 'nmi', 'precision', 'recall', 'accuracy']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def _check(gt_labels, pred_labels):
    if gt_labels.ndim != 1:
        raise ValueError("gt_labels must be 1D: shape is %r" %
                         (gt_labels.shape,))
    if pred_labels.ndim != 1:
        raise ValueError("pred_labels must be 1D: shape is %r" %
                         (pred_labels.shape,))
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            "gt_labels and pred_labels must have same size, got %d and %d" %
            (gt_labels.shape[0], pred_labels.shape[0]))
    return gt_labels, pred_labels


def _get_lb2idxs(labels):
    lb2idxs = {}
    for idx, lb in enumerate(labels):
        if lb not in lb2idxs:
            lb2idxs[lb] = []
        lb2idxs[lb].append(idx)
    return lb2idxs


def _compute_fscore(pre, rec):
    return 2. * pre * rec / (pre + rec)


def fowlkes_mallows_score(gt_labels, pred_labels, sparse=True):
    ''' The original function is from `sklearn.metrics.fowlkes_mallows_score`.
        We output the pairwise precision, pairwise recall and F-measure,
        instead of calculating the geometry mean of precision and recall.
    '''
    n_samples, = gt_labels.shape

    c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples

    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore


def pairwise(gt_labels, pred_labels, sparse=True):
    _check(gt_labels, pred_labels)
    return fowlkes_mallows_score(gt_labels, pred_labels, sparse)


def bcubed0(gt_labels, pred_labels):
    """
    计算bcubed的precision, recall, f-score及expanding
    :param gt_labels:
    :param pred_labels:
    :return:
    """
    gt_lb2idxs = _get_lb2idxs(gt_labels)
    pred_lb2idxs = _get_lb2idxs(pred_labels)

    num_lbs = len(gt_lb2idxs)
    pre = np.zeros(num_lbs)
    rec = np.zeros(num_lbs)
    gt_num = np.zeros(num_lbs)

    expand = np.zeros(num_lbs)
    for i, gt_idxs in enumerate(gt_lb2idxs.values()):
        all_pred_lbs = np.unique(pred_labels[gt_idxs])
        gt_num[i] = len(gt_idxs)
        expand[i] = all_pred_lbs.shape[0]
        for pred_lb in all_pred_lbs:
            pred_idxs = pred_lb2idxs[pred_lb]
            n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
            pre[i] += n ** 2 / len(pred_idxs)
            rec[i] += n ** 2 / gt_num[i]

    gt_num = gt_num.sum()
    avg_pre = pre.sum() / gt_num
    avg_rec = rec.sum() / gt_num
    fscore = _compute_fscore(avg_pre, avg_rec)

    return avg_pre, avg_rec, fscore, expand.mean()


def bcubed(gt_labels, pred_labels):
    """
    输出becubed函数中各项指标，以及丢弃n个档案后的指标
    和剩余的图片数量和label数量
    :param gt_labels:
    :param pred_labels:
    :param n:
    :return:
    """
    pred_lb2idxs = _get_lb2idxs(pred_labels)
    n = 1
    ind = []
    for i in pred_lb2idxs.values():
        if len(i) > n:
            for m in i:
                ind.append(m)

    avg_pre, avg_rec, fscore, expand = bcubed0(gt_labels, pred_labels)
    # print('avg_pre:{}, avg_rec:{}, fscore:{}, expanding:{}, rest images:{}, rest_gt_labels:{} '.
    #       format(avg_pre, avg_rec, fscore, expand, len(gt_labels), len(list(set(gt_labels)))))
    #
    # avg_pre1, avg_rec1, fscore1, expand1 = bcubed0(gt_labels[ind], pred_labels[ind])
    # print('avg_pre:{}, avg_rec:{}, fscore:{}, expanding:{}, rest images:{}, rest_gt_labels:{} '.
    #       format(avg_pre1, avg_rec1, fscore1, expand1, len(ind), len(list(set(gt_labels[ind])))))

    return avg_pre, avg_rec, fscore


def nmi(gt_labels, pred_labels):
    return normalized_mutual_info_score(pred_labels, gt_labels)


def precision(gt_labels, pred_labels):
    return precision_score(gt_labels, pred_labels)


def recall(gt_labels, pred_labels):
    return recall_score(gt_labels, pred_labels)


def accuracy(gt_labels, pred_labels):
    return np.mean(gt_labels == pred_labels)
