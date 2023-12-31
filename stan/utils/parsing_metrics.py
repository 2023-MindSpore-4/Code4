import numpy as np
import os


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class, bg_thre=200):
    """Returns accuracy score evaluation result.
	  - overall accuracy
	  - mean accuracy
	  - mean IU
	  - fwavacc
	"""
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        # hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += _fast_hist(lt[lt < bg_thre].flatten(), lp[lt < bg_thre].flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def label_confusion_matrix(label_trues, label_preds, n_class, bg_thre=200):
    # eps=1e-20
    hist = np.zeros((n_class, n_class), dtype=float)
    """ (8,256,256), (256,256) """
    for lt, lp in zip(label_trues, label_preds):
        # hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += _fast_hist(lt[lt < bg_thre].flatten(), lp[lt < bg_thre].flatten(), n_class)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # for i in range(n_class):
    # 	hist[i,:]=(hist[i,:]+eps)/sum(hist[i,:]+eps)
    return hist, iu


def hist_based_accu_cal(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu
