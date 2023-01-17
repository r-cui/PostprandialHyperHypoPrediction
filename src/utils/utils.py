import numpy as np
import yaml
from datetime import datetime


def n_params(model):
    """ Calculate total number of parameters in a model.
    Args:
        model: nn.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(yaml_dir):
    with open(yaml_dir) as f:
        res = yaml.safe_load(f)
    return res


# for evaluation
def prepare_confmat(conf_mat):
    """ Convert 1d confusion matrix into 2d by treating all correct prediction as TN.
    """
    if conf_mat.shape == (1, 1):
        conf_mat = np.array([[conf_mat[0, 0], 0], [0, 0]])
    assert conf_mat.shape == (2, 2)
    return conf_mat


def ravel(conf_mat):
    """ Export the metrics from a 2d confusion matrix.
    Args:
        conf_mat: np.array(
            [[int, int],
             [int, int]]
        )
    """
    tn, fp, fn, tp = conf_mat.ravel()
    se = round(tp / (tp + fn + 1e-6), 2)
    sp = round(tn / (tn + fp + 1e-6), 2)
    fa = round(fp / (tp + fp + 1e-6), 2)
    mcc = round((tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1e-6), 2)

    return tn, fp, fn, tp, se, sp, fa, mcc


def de_standardize(glucose, mean_std):
    mean = mean_std["mean"]["glucose"]
    std = mean_std["std"]["glucose"]
    return glucose * std + mean


