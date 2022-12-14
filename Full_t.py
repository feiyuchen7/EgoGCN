from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np

def full_t(epoch, model, prefix, logger=None):   
    logger.write(prefix+' start...')
    model.eval()

    with no_grad():
        precision, recall, ndcg_score = model.accuracy()
        logger.write('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, precision, recall, ndcg_score))
        return precision, recall, ndcg_score



