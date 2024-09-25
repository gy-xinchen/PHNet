# -*- coding: utf-8 -*-
# @Time    : 2024/9/25 10:37
# @Author  : yuan
# @File    : logistic.py
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,f1_score, recall_score
from sklearn.metrics import roc_curve, auc

txt_path = r"F:\袁心辰研究生资料\投稿资料PHNet\20240830大修投稿资料\修改意见与结果\修改意见分类\实验补充部分\（待办）R1.4.临床实验\二元逻辑回归结果\data.txt"

dataset = np.loadtxt(txt_path)
kf = KFold(n_splits=5,  shuffle=True)

lower_bound_list = []
upper_bound_list = []
test_accuracy = []
test_recall = []
test_specificity = []
test_positive_predictive_value = []
test_negative_predictive_value = []
test_f1 = []
test_roc_auc = []

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
    lower_bound = np.nanpercentile(bootstrapped_auc, 2.5)
    upper_bound = np.nanpercentile(bootstrapped_auc, 97.5)
    return bootstrapped_auc, lower_bound, upper_bound

for train_index, test_index in kf.split(dataset):
    dataset_train, dataset_test = dataset[train_index], dataset[test_index]
    X_train, y_train = dataset_train[:, 1:], dataset_train[:, 0]
    X_test, y_test = dataset_test[:, 1:], dataset_test[:, 0]

    model = LogisticRegression(solver="liblinear")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    bootstrapped_auc, lower_bound, upper_bound = calculate_auc_ci(y_test, y_pred_prob)
    lower_bound_list.append(lower_bound)
    upper_bound_list.append(upper_bound)
    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
    test_roc_auc_for_val = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    test_accuracy.append(accuracy)
    test_recall.append(recall)
    # 计算特异性
    specificity = recall_score(y_test, y_pred, pos_label=0)
    test_specificity.append(specificity)
    test_positive_predictive_value.append(precision)
    # 计算负预测值
    negative_predictive_value = precision_score(y_test, y_pred, pos_label=0)
    test_negative_predictive_value.append(negative_predictive_value)
    test_f1.append(f1)
    test_roc_auc.append(roc_auc)


# 打印各个评估指标的结果
print("lower_bound_list",lower_bound_list)
print("upper_bound_list",upper_bound_list)
print("Accuracy:", test_accuracy)
print("Recall:", test_recall)
print("Specificity:", test_specificity)
print("Positive Predictive Value:", test_positive_predictive_value)
print("Negative Predictive Value:", test_negative_predictive_value)
print("F1 Score:", test_f1)
print("ROC AUC:", test_roc_auc)
