"""
PAH Diagnosis from Cardiac MRI via a Multilinear PCA-based Pipeline

Reference:
Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2021). A machine learning
cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis.
European Heart Journal-Cardiovascular Imaging. https://academic.oup.com/ehjcimaging/article/22/2/236/5717931
"""
import argparse
import os

import numpy as np
import pandas as pd
from compared_ML_model.MPCA.MPCA_raw_code.config import get_cfg_defaults
from sklearn.model_selection import cross_validate
import SimpleITK as sitk
from kale.interpret import model_weights, visualize
from kale.loaddata.image_access import dicom2arraylist, read_dicom_dir
from kale.pipeline.mpca_trainer import MPCATrainer
from kale.prepdata.image_transform import mask_img_stack, normalize_img_stack, reg_img_stack, rescale_img_stack
from kale.utils.download import download_file_by_url
import cv2

output_path = r"G:\CMR-res\github_PAHNet\compared_ML_model\MPCA\MPCA_raw_code\outputs"

def crop_roi(image, first_roi=None):
    if first_roi == None:
        # Use thresholding to get a mask of non-zero parts of an image
        _, mask = cv2.threshold((image * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

        # Find the smallest bounding rectangle of non-zero pixels in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None  # has no non-zero part, returns None or appropriate value

        # 计算最小外接矩形的边界框
        x, y, w, h = cv2.boundingRect(contours[0])

        # 裁剪感兴趣的区域
        roi = image[y:y+h, x:x+w]

        first_roi = x, y, w, h

        return roi, first_roi
    else:
        x, y, w, h = first_roi
        roi = image[y:y + h, x:x + w]

        return roi, first_roi

# Define args hyperparameters
def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description="Machine learning pipeline for PAH diagnosis")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)

    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    # ---- setup configs ----
    cfg = get_cfg_defaults() # cfg is defined as the get_cfg_defaults function
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    save_figs = cfg.OUTPUT.SAVE_FIG
    fig_format = cfg.SAVE_FIG_KWARGS.format
    print(f"Save Figures: {save_figs}")

    # ---- initialize folder to store images ----
    save_figures_location = cfg.OUTPUT.ROOT
    print(f"Save Figures: {save_figures_location}")

    if not os.path.exists(save_figures_location):
        os.makedirs(save_figures_location)

    # ---- setup dataset ----
    base_dir = cfg.DATASET.BASE_DIR # 获取数据文件路径
    file_format = cfg.DATASET.FILE_FORAMT # 输入文件类型为zip
    download_file_by_url(cfg.DATASET.SOURCE, cfg.DATASET.ROOT, "%s.%s" % (base_dir, file_format), file_format)

    img_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.IMG_DIR)
    patient_dcm_list = read_dicom_dir(img_path, sort_instance=True, sort_patient=True)
    images, patient_ids = dicom2arraylist(patient_dcm_list, return_patient_id=True) # images is a list of size 179, each containing these 20 slice data
    patient_ids = np.array(patient_ids, dtype=int)
    n_samples = len(images) # 获取样本总数

    mask_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.MASK_DIR)
    mask_dcm = read_dicom_dir(mask_path, sort_instance=True)
    mask = dicom2arraylist(mask_dcm, return_patient_id=False)[0][0, ...]

    landmark_path = os.path.join(cfg.DATASET.ROOT, base_dir, cfg.DATASET.LANDMARK_FILE)
    landmark_df = pd.read_csv(landmark_path, index_col="Subject").loc[patient_ids]  # read .csv file as dataframe
    landmarks = landmark_df.iloc[:, :-1].values
    y = landmark_df["Group"].values # 获取二分类标签
    y[np.where(y != 0)] = 1  # convert to binary classification problem, i.e. no PH vs PAH

    # plot the first phase of images with landmarks
    marker_names = list(landmark_df.columns[1::2])
    markers = []
    for marker in marker_names:
        marker_name = marker.split(" ")
        marker_name.pop(-1)
        marker_name = " ".join(marker_name)
        markers.append(marker_name)

    if save_figs:
        n_img_per_fig = 45
        n_figures = int(n_samples / n_img_per_fig) + 1
        for k in range(n_figures):
            visualize.plot_multi_images(
                [images[i][0, ...] for i in range(k * n_img_per_fig, min((k + 1) * n_img_per_fig, n_samples))],
                marker_locs=landmarks[k * n_img_per_fig : min((k + 1) * n_img_per_fig, n_samples), :],
                im_kwargs=dict(cfg.PLT_KWS.IM),
                marker_cmap="Set1",
                marker_kwargs=dict(cfg.PLT_KWS.MARKER),
                marker_titles=markers,
                image_titles=list(patient_ids[k * n_img_per_fig : min((k + 1) * n_img_per_fig, n_samples)]),
                n_cols=5,
            ).savefig(
                str(save_figures_location) + "/0)landmark_visualization_%s_of_%s.%s" % (k + 1, n_figures, fig_format),
                **dict(cfg.SAVE_FIG_KWARGS),
            )

    # ---- data pre-processing ----
    # ----- image registration -----
    img_reg, max_dist = reg_img_stack(images.copy(), landmarks, landmarks[0])
    plt_kawargs = {**{"im_kwargs": dict(cfg.PLT_KWS.IM), "image_titles": list(patient_ids)}, **dict(cfg.PLT_KWS.PLT)}
    if save_figs:
        visualize.plot_multi_images([img_reg[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/1)image_registration.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

    # ----- masking -----
    img_masked = mask_img_stack(img_reg.copy(), mask)
    if save_figs:
        visualize.plot_multi_images([img_masked[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/2)masking.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

    # # ----- resize -----
    # img_rescaled = rescale_img_stack(img_masked.copy(), scale=4)
    # if save_figs:
    #     visualize.plot_multi_images([img_rescaled[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
    #         str(save_figures_location) + "/3)resize.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
    #     )

    # ----- normalization -----
    img_norm = normalize_img_stack(img_masked.copy())
    if save_figs:
        visualize.plot_multi_images([img_norm[i][0, ...] for i in range(n_samples)], **plt_kawargs).savefig(
            str(save_figures_location) + "/4)normalize.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS)
        )

    crop_img = []
    for slice in range(len(img_norm)):
        img_num = img_norm[slice]
        crop_list = []
        first_roi = None
        for time in range(img_num.shape[0]): # times==20
            roi, first_roi = crop_roi(img_num[time],first_roi)
            crop_list.append(roi)
        crop_list = np.array(crop_list)

        crop_img.append(crop_list)




    for name in range(len(crop_img)):
        label = y[name]
        img = crop_img[name]
        if label == 0:
            id = "nopah"
        else:
            id = "pah"
        out_name = r"I:\CMR-res\muti_center_data0927\xiefeierde\patient{}_{}.nii.gz".format(str(name).rjust(3,"0"),id)
        img = sitk.GetImageFromArray(img)
        sitk.WriteImage(img, out_name)





    # # ---- evaluating machine learning pipeline ----
    # x = np.concatenate([img_norm[i].reshape((1,) + img_norm[i].shape) for i in range(n_samples)], axis=0) # 将数据合并打包为ndarray,size = [179,20,32,32]
    # trainer = MPCATrainer(classifier=cfg.PIPELINE.CLASSIFIER, n_features=200) # cfg.PIPELINE.CLASSIFIER == linear_svm
    # cv_results = cross_validate(trainer, x, y, cv=10, scoring=["accuracy", "roc_auc"], n_jobs=1) # x是数据 y是标签
    #
    # print("Averaged training time: {:.4f} seconds".format(np.mean(cv_results["fit_time"])))
    # print("Averaged testing time: {:.4f} seconds".format(np.mean(cv_results["score_time"])))
    # print("Averaged Accuracy: {:.4f}".format(np.mean(cv_results["test_accuracy"])))
    # print("Averaged AUC: {:.4f}".format(np.mean(cv_results["test_roc_auc"])))
    #
    # # ---- model weights interpretation ----
    # trainer.fit(x, y)
    #
    # weights = trainer.mpca.inverse_transform(trainer.clf.coef_) - trainer.mpca.mean_
    # weights = rescale_img_stack(weights, cfg.PROC.SCALE)  # rescale weights to original shape
    # weights = mask_img_stack(weights, mask)  # masking weights
    # top_weights = model_weights.select_top_weight(weights, select_ratio=0.02)  # select top 2% weights
    # if save_figs:
    #     visualize.plot_weights(
    #         top_weights[0][0],
    #         background_img=images[0][0],
    #         im_kwargs=dict(cfg.PLT_KWS.IM),
    #         marker_kwargs=dict(cfg.PLT_KWS.WEIGHT),
    #     ).savefig(str(save_figures_location) + "/5)weights.%s" % fig_format, **dict(cfg.SAVE_FIG_KWARGS))


if __name__ == "__main__":
    main()