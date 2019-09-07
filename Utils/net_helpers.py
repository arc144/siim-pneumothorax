import time
from contextlib import contextmanager

import numpy as np


###########################################################################
########################## AUXILIARY FUNCTIONS ############################
###########################################################################
@contextmanager
def timer(title, min=True):
    t0 = time.time()
    yield
    t1 = (time.time() - t0)
    if min:
        t1 /= 60
        unit = 'min'
    else:
        unit = 's'
    print("{} - done in {:.3f}{:}".format(title, t1, unit))


def update_metrics_dict(metrics_dict, new_metrics):
    for key, values in new_metrics.items():
        try:
            if isinstance(values, list) or isinstance(values, np.ndarray):
                metrics_dict[key].extend(values)
            else:
                metrics_dict[key].append(values)
        except KeyError:
            if isinstance(values, list) or isinstance(values, np.ndarray):
                metrics_dict[key] = list(values)
            else:
                metrics_dict[key] = [values]
    return metrics_dict


def update_log(log, key, value):
    '''Update log dict by appending new values to key
    If key does not exits, creates it automatically'''
    try:
        log[key].append(value)
    except KeyError:
        log[key] = [value]


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
        # p.requires_grad = False

def train_gn_only(m):
    classname = m.__class__.__name__
    if classname.find('GroupNorm') == -1:
        m.requires_grad = False
