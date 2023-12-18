import numpy as np
from sklearn.metrics import auc, roc_curve

def calculate_auc_ci(label_list, predict_list):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_auc = []
    np.random.seed(rng_seed)
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(predict_list), len(predict_list))
        fpr, tpr, thresholds = roc_curve(label_list[indices], predict_list[indices])
        bootstrapped_auc.append(auc(fpr, tpr))
    bootstrapped_auc = np.array(bootstrapped_auc)
    lower_bound = np.percentile(bootstrapped_auc, 2.5)
    upper_bound = np.percentile(bootstrapped_auc, 97.5)
    return bootstrapped_auc, lower_bound, upper_bound