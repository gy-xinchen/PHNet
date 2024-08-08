""" --cfg configs/tutorial_lr.yaml """
import os
import numpy as np
import pandas as pd
from compared_ML_model.MPCA.MPCA_raw_code.config import get_cfg_defaults
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer,roc_auc_score
from sklearn.model_selection import cross_val_predict
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import SimpleITK as sitk
import argparse
import csv
from sklearn.metrics import roc_curve, auc
import argparse
import os
from compared_ML_model.MPCA.MPCA_raw_code.config import get_cfg_defaults
from kale.loaddata.image_access import dicom2arraylist, read_dicom_dir
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url
import cv2

def get_predict_proba(estimator, X, y):
    return estimator.predict_proba(X)

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

def calcu_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def calcu_positive_predictive_value(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp)

def calcu_negative_predictive_value(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)

# define args
def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Machine learning pipeline for PAH diagnosis")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    args = parser.parse_args()
    return args

# output_excle_path = r"I:\CMR-res\MPCACodes1.3\20230816\output_path"
dir_path = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_224x224"
# dir_path = r"I:\CMR-res\muti_center_data0927\xiefeierde"
pah_path = os.listdir(os.path.join(dir_path,"PAH"))
nopah_path = os.listdir(os.path.join(dir_path,"noPAH"))
png_list = []
label_list = []
for i in range(len(pah_path)):
    sitk_img = sitk.ReadImage(os.path.join(dir_path,"PAH",pah_path[i]))
    sitk_array = sitk.GetArrayFromImage(sitk_img)
    label = 1
    png_list.append(sitk_array)
    label_list.append(label)
    print("{} done".format(i))
for i in range(len(nopah_path)):
    sitk_img = sitk.ReadImage(os.path.join(dir_path,"noPAH",nopah_path[i]))
    sitk_array = sitk.GetArrayFromImage(sitk_img)
    label = 0
    png_list.append(sitk_array)
    label_list.append(label)
    print("{} done".format(i))
png_list = np.array(png_list)


# ---- setup configs ----
args = arg_parse()
cfg = get_cfg_defaults() #
cfg.merge_from_file(args.cfg)
cfg.freeze()
print(cfg)
# ----- resize -----
img_rescaled = rescale_img_stack(png_list.copy(), scale=1)  # resample data
x = np.concatenate([img_rescaled[i].reshape((1,) + img_rescaled[i].shape) for i in range(len(pah_path)+len(nopah_path))], axis=0)
trainer = MPCATrainer(classifier="lr", classifier_params="auto", n_features=66) # cfg.PIPELINE.CLASSIFIER == linear_svm


label_list = np.array(label_list)

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_accuracy = []
test_recall = []
test_specificity = []
test_positive_predictive_value = []
test_negative_predictive_value = []
test_f1 = []
test_roc_auc = []
circle_num = 0

test_accuracy_output = []
test_recall_output = []
test_specificity_output = []
test_positive_predictive_value_output = []
test_negative_predictive_value_output = []
test_f1_output = []
test_roc_auc_output = []
test_roc_auc_out =[]
test_roc_auc_out2 =[]

# # ---- setup dataset ----
# base_dir = cfg.DATASET.BASE_DIR
# file_format = cfg.DATASET.FILE_FORAMT
# download_file_by_url(cfg.DATASET.SOURCE, cfg.DATASET.ROOT, "%s.%s" % (base_dir, file_format),
#                      file_format)
# img_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.IMG_DIR)
# patient_dcm_list = read_dicom_dir(img_path, sort_instance=True, sort_patient=True)
# images, patient_ids = dicom2arraylist(patient_dcm_list, return_patient_id=True)
# patient_ids = np.array(patient_ids, dtype=int)
# n_samples = len(images)
#
# mask_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.MASK_DIR)
# mask_dcm = read_dicom_dir(mask_path, sort_instance=True)
# mask = dicom2arraylist(mask_dcm, return_patient_id=False)[0][0, ...]
# landmark_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.LANDMARK_FILE)
# landmark_df = pd.read_csv(landmark_path, index_col="Subject").loc[patient_ids]  # read .csv file as dataframe
# landmarks = landmark_df.iloc[:, :-1].values
# y_test_out = landmark_df["Group"].values
# y_test_out[np.where(y_test_out != 0)] = 1  # convert to binary classification problem, i.e. no PH vs PAH

###########################################################################################
# ceshi1
ceshi_path = r"G:\CMR-res\muti_center_data0927\guangrenming"
# ceshi_path = r"I:\CMR-res\muti_center_data0927\mix_train_data\slice05"
ceshi2_path = r"G:\CMR-res\muti_center_data0927\chaoyang"
dir_path = os.path.join(ceshi_path,"PAH_MaxMin_score_norm64_slice05")
dir_path1 = os.path.join(ceshi_path,"noPAH_MaxMin_score_norm64_slice05")
dir2_path = os.path.join(ceshi2_path,"PAH_MaxMin_score_norm64_slice05")
dir2_path1 = os.path.join(ceshi2_path,"noPAH_MaxMin_score_norm64_slice05")
files = os.listdir(dir_path)
files1 = os.listdir(dir_path1)
new_files = os.listdir(dir2_path)
new_files1 = os.listdir(dir2_path1)
x_test_out = []
y_test_out = []
x_test2_out = []
y_test2_out = []
for path in range(len(files)):
    raw_img = sitk.ReadImage(os.path.join(dir_path, files[path]))
    img = sitk.GetArrayFromImage(raw_img)
    x_test_out.append(img)
    y_test_out.append(1)
for path in range(len(files1)):
    raw_img1 = sitk.ReadImage(os.path.join(dir_path1, files1[path]))
    img1 = sitk.GetArrayFromImage(raw_img1)
    x_test_out.append(img1)
    y_test_out.append(0)
x_test_out = np.array(x_test_out)
for path in range(len(new_files)):
    raw_img = sitk.ReadImage(os.path.join(dir2_path, new_files[path]))
    img_new = sitk.GetArrayFromImage(raw_img)
    x_test2_out.append(img_new)
    y_test2_out.append(1)
for path in range(len(new_files1)):
    raw_img1 = sitk.ReadImage(os.path.join(dir2_path1, new_files1[path]))
    img_new1 = sitk.GetArrayFromImage(raw_img1)
    x_test2_out.append(img_new1)
    y_test2_out.append(0)
x_test2_out = np.array(x_test2_out)


lower_bound_list = []
upper_bound_list = []


for train_index, test_index in kf.split(png_list, label_list):
    x_train, x_test = png_list[train_index], png_list[test_index]
    y_train, y_test = label_list[train_index], label_list[test_index]


    trainer.fit(x_train, y_train)


    y_pred = trainer.predict(x_test)
    # out_value
    y_pred_out = trainer.predict(x_test_out)
    # ceshi2
    y_pred_out2 = trainer.predict(x_test2_out)

    # If the model supports predicted probabilities
    output_excle_path = r"G:\CMR-res\muti_center_data0927\Data_reword\MPCA"
    if hasattr(trainer, "predict_proba"):
        y_pred_prob = trainer.predict_proba(x_test)[:, 1]
        y_pred_prob_out = trainer.predict_proba(x_test_out)[:, 1]
        y_pred_prob_out2 = trainer.predict_proba(x_test2_out)[:, 1]
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_prob)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thersholds})
        excel_filename = "Fold{}.csv".format(circle_num)
        excel_path = os.path.join(output_excle_path, "lr", excel_filename)
        wb = Workbook()
        ws = wb.active
        ws.title = "ROC"
        for r in dataframe_to_rows(roc_df, index=False, header=True):
            ws.append(r)
        bootstrapped_auc, lower_bound, upper_bound = calculate_auc_ci(y_test, y_pred_prob)
        lower_bound_list.append(lower_bound)
        upper_bound_list.append(upper_bound)
        auc_interval_df = pd.DataFrame({"AUC": [auc], "lower Bound": [lower_bound], "Upper Bound": [upper_bound]},
                                       index=[0])
        auc_interval_df.to_csv(excel_path, index=False)
        wb.save(excel_path)
        circle_num += 1

    else:
        y_pred_prob = None

    # Calculate various evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    # out_roc
    out_roc_auc = roc_auc_score(y_test_out, y_pred_prob_out) if y_pred_prob_out is not None else None
    out_roc_auc2 = roc_auc_score(y_test2_out, y_pred_prob_out2) if y_pred_prob_out2 is not None else None

    test_accuracy.append(accuracy)
    test_recall.append(recall)

    specificity = recall_score(y_test, y_pred, pos_label=0)
    test_specificity.append(specificity)
    test_positive_predictive_value.append(precision)

    negative_predictive_value = precision_score(y_test, y_pred, pos_label=0)
    test_negative_predictive_value.append(negative_predictive_value)
    test_f1.append(f1)
    test_roc_auc.append(roc_auc)
    test_roc_auc_out.append(out_roc_auc)
    test_roc_auc_out2.append(out_roc_auc2)






print("lower_bound_list",lower_bound_list)
print("upper_bound_list",upper_bound_list)
print("Accuracy:", test_accuracy)
print("Recall:", test_recall)
print("Specificity:", test_specificity)
print("Positive Predictive Value:", test_positive_predictive_value)
print("Negative Predictive Value:", test_negative_predictive_value)
print("F1 Score:", test_f1)
print("ROC AUC:", test_roc_auc)
# out_test value
print("##########")
print("out_ROC AUC:",test_roc_auc_out)
print("out2_ROC AUC:",test_roc_auc_out2)
print()




