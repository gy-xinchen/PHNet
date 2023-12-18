import SimpleITK as sitk
from radiomics import  featureextractor
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy.stats
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# PAH_folder = r"G:\CMR-res\muti_center_data0927\xiefeierde\PAH"
# noPAH_folder = r"G:\CMR-res\muti_center_data0927\xiefeierde\noPAH"
# test_PAH_folder = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_64x64\PAH"
# test_noPAH_folder = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_64x64\noPAH"

PAH_folder = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_64x64\PAH"
noPAH_folder = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice05_64x64\noPAH"
test_PAH_folder = r"G:\CMR-res\muti_center_data0927\xiefeierde\PAH"
test_noPAH_folder = r"G:\CMR-res\muti_center_data0927\xiefeierde\noPAH"


class_label_noPAH = 0  # 类别标签 1
class_label_PAH = 1  # 类别标签 2

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
    lower_bound = np.nanpercentile(bootstrapped_auc, 2.5)
    upper_bound = np.nanpercentile(bootstrapped_auc, 97.5)
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

def create_binary_mask(image):
    # 获取图像数组
    image_array = sitk.GetArrayFromImage(image)

    # 使用图像的最小值作为阈值
    threshold_value = np.min(image_array)

    # 使用阈值将图像二值化
    binary_image = sitk.BinaryThreshold(image, lowerThreshold=threshold_value+0.000001)

    return binary_image

def load_and_extract_features(folder, class_label):
    feature_list = []
    labels = []

    for filename in os.listdir(folder):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(folder, filename)

            # 加载图像
            image = sitk.ReadImage(file_path)
            # 生成二值化掩码
            binary_mask = create_binary_mask(image)
            # 提取特征
            features = extract_features(image, binary_mask)

            # 添加到特征列表
            feature_list.append(features)

            # 添加标签
            labels.append(class_label)
            print("{} done".format(file_path))

    return feature_list, labels


def extract_features(image, binary_mask):
    params = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'sigma': [3, 5]
    }
    image_array = sitk.GetArrayFromImage(image)[0,:,:]

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    # 获取特征提取器
    extractor.enableImageTypeByName('Wavelet') # 小波
    extractor.enableImageTypeByName('Gradient') # 梯度
    extractor.enableImageTypeByName('LoG') # LoG
    extractor.enableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("gldm")


    # Radiomics Features
    radiomics_features = extractor.execute(image, binary_mask)
    # 提取 radiomics 特征的值
    radiomics_features_values = list(radiomics_features.values())[26:]

    return radiomics_features_values






# 4. 加载并提取特征
features_PAH, labels_PAH = load_and_extract_features(PAH_folder, class_label_PAH)
features_noPAH, labels_noPAH = load_and_extract_features(noPAH_folder, class_label_noPAH)

test_features_PAH, test_labels_PAH = load_and_extract_features(test_PAH_folder, class_label_PAH)
test_features_noPAH, test_labels_noPAH = load_and_extract_features(test_noPAH_folder, class_label_noPAH)

# 合并两类的特征和标签
X = features_PAH + features_noPAH
Y = labels_PAH + labels_noPAH

test_X = test_features_PAH + test_features_noPAH
test_Y = test_labels_PAH + test_labels_noPAH

# 6. 创建 MLP 分类器
mlp_classifier = MLPClassifier(learning_rate_init=0.0001, hidden_layer_sizes=(40,40,10), max_iter=500, alpha=0.01)

# 初始化交叉验证的折数
n_splits = 5
circle_num = 0
lower_bound_list = []
upper_bound_list = []
test_accuracy = []
test_recall = []
test_specificity = []
test_positive_predictive_value = []
test_negative_predictive_value = []
test_f1 = []
test_roc_auc = []
test_roc_auc2 = []
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  # 使用分层折叠交叉验证
# 手动进行交叉验证
for train_index, test_index in kf.split(X, Y):
    x_train, x_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(Y)[train_index], np.array(Y)[test_index]
    print(test_index)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    test_X_for_val = scaler.transform(test_X)
    # 7. 训练分类器
    mlp_classifier.fit(X_train_scaled, y_train)
    # 8. 预测
    y_pred = mlp_classifier.predict(X_test_scaled)
    y_pred_test = mlp_classifier.predict(test_X_for_val)
    # 如果模型支持预测概率
    output_excle_path = r"G:\CMR-res\muti_center_data0927\Data_reword\Texture_ML"
    if hasattr(mlp_classifier, "predict_proba"):
        y_pred_prob = mlp_classifier.predict_proba(X_test_scaled)[:, 1]
        print(y_pred_prob)
        fpr, tpr, thersholds = roc_curve(y_test, y_pred_prob)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thersholds})
        excel_filename = "Fold{}.csv".format(circle_num)
        excel_path = os.path.join(output_excle_path, "MLP", excel_filename)
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

    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
    test_roc_auc_for_val = roc_auc_score(test_Y, y_pred_test) if y_pred_prob is not None else None


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
    test_roc_auc2.append(test_roc_auc_for_val)
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
print("ROC AUC test:", test_roc_auc2)