import glob
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from net_hub.PHNet import densenet121
import argparse
import SimpleITK as sitk
import glob
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

dir_path = r"D:\CMR-res\muti_center_data0927\test_qinyuan\crop_flow\layer05test"
files = glob.glob(dir_path+"/*nii.gz")
weights_path_dir = r"G:\CMR-res\muti_center_data0927\Data_reword\condensenetV2_adapt_triplet\densenet3D\weight\weight3x1_0.5_lr3_10-4_labelsmoothing0.1_drop0.3"
weights_path_file = os.listdir(weights_path_dir)
auc_list = []
acc_list = []

def main():
    for j in range(len(weights_path_file)):
        classifer_id = []
        predicted_probability_id = []
        batch_size = 8  # batchsize

        for i in range(0, len(files), batch_size):
            nii_imgs_batch = []
            for idx in range(i, min(i + batch_size, len(files))):
                nii_path = files[idx]
                nii_img = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(nii_path)))
                nii_img = torch.unsqueeze(torch.unsqueeze(nii_img, dim=0), dim=0)
                nii_imgs_batch.append(nii_img)

                id = nii_path.split("\\")[-1].split("_")[1][2]
                if id == "N":
                    classifer_id.append(0)
                else:
                    classifer_id.append(1)

            # stack batch
            nii_imgs_batch = torch.cat(nii_imgs_batch, dim=0)

            device = torch.device("cuda:0")
            json_path = r'G:\CMR-res\muti_center_data0927\class_indices.json'
            with open(json_path, "r") as f:
                class_indict = json.load(f)

            model = densenet121()
            weights_path = os.path.join(weights_path_dir, weights_path_file[j])
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model = model.cuda()
            model.eval()

            with torch.no_grad():
                # predict class
                output, embedding = model(nii_imgs_batch.float().to(device))
                predicts = torch.softmax(output, dim=1)
                predicted_probabilities = predicts[:, 1].tolist()
                predicted_probability_id.extend(predicted_probabilities)

            print("Batch {} to {} done".format(i, min(i + batch_size, len(files))))

        threshold = 0.50
        predicted_labels = [1 if prob > threshold else 0 for prob in predicted_probability_id]
        acc = accuracy_score(classifer_id, predicted_labels)
        # auc = roc_auc_score(classifer_id, predicted_probability_id)
        acc_list.append(acc)
        # auc_list.append(auc)

    print("acc:", acc_list)
    print("AUC:", auc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cdnv2_a', type=str, metavar='MODEL',
                        help='Name of model to train (default: "cdnv2_a"')  # 模型名称
    args = parser.parse_args()
    main()
