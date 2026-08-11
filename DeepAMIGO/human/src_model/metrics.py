import numpy as np
import scipy.sparse as ssp
import warnings
import torch
from sklearn.metrics import roc_auc_score  

def compute_cafa_metrics(targets, scores):
 
  
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        

    if scores.min() < 0 or scores.max() > 1.0:
        scores = np.clip(scores, -20, 20)
        scores = 1.0 / (1.0 + np.exp(-scores))
        
   
    row_sums = targets.sum(axis=1)
    valid_mask = row_sums > 0
    
    if not np.any(valid_mask):
        return {'Fmax': 0.0, 'AUPR_Macro': 0.0, 'AUPR': 0.0, 'AUC': 0.0} 
        
    targets = targets[valid_mask]
    scores = scores[valid_mask]
    
   
    auc_list = []
    for col in range(targets.shape[1]):
        if len(np.unique(targets[:, col])) == 2:
            auc_list.append(roc_auc_score(targets[:, col], scores[:, col]))
    auc_score = np.mean(auc_list) if len(auc_list) > 0 else 0.0
    
    targets = ssp.csr_matrix(targets)
    
    fmax_ = 0.0, 0.0
    precisions = []
    recalls = []
    
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = correct / cut_sc.sum(axis=1)
            r = correct / targets.sum(axis=1)
            p = np.average(p[np.invert(np.isnan(p))])
            r = np.average(r[np.invert(np.isnan(r))])
            
        if np.isnan(p) or np.isnan(r):
            continue
            
        precisions.append(p)
        recalls.append(r)
            
        try:
            if p + r > 0.0:
                f1 = 2 * p * r / (p + r)
                fmax_ = max(fmax_, (f1, cut))
        except ZeroDivisionError:
            pass
            
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    if len(recalls) > 0:
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)
    else:
        aupr = 0.0
    
    return {'Fmax': float(fmax_[0]), 'AUPR_Macro': float(aupr), 'AUPR': float(aupr), 'AUC': float(auc_score)}