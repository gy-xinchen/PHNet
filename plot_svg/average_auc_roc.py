import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize mean fpr and tpr arrays for each model
mean_fpr_dict = {}
mean_tpr_dict = {}
mean_auc_dict = {}
mean_lower_bound_dict = {}
mean_upper_bound_dict = {}
# Iterate over the 5 folds
for i in range(5):
    fold = i + 1
    dir_path = r"G:\CMR-res\muti_center_data0927\Data_reword\ismsm會議統計"
    model_dict = {}
    model_dict['MPCA-LR'] = pd.read_excel(os.path.join(dir_path, "svc", "Fold{}.csv".format(fold)))
    model_dict['Texture-MLP'] = pd.read_excel(os.path.join(dir_path, "Texture-MLP", "Fold{}.csv".format(fold)))
    model_dict['AlexNet'] = pd.read_excel(os.path.join(dir_path, "AlexNet", "Fold{}.csv".format(fold)))
    model_dict['Vgg16'] = pd.read_excel(os.path.join(dir_path, "Vgg16", "Fold{}.csv".format(fold)))
    model_dict['ResNet101'] = pd.read_excel(os.path.join(dir_path, "resnet101", "Fold{}.csv".format(fold)))
    model_dict['DenseNet121'] = pd.read_excel(os.path.join(dir_path, "densenet121", "Fold{}.csv".format(fold)))
    model_dict['AlexNet-HSATBCL'] = pd.read_excel(os.path.join(dir_path, "AlexNet_loss", "Fold{}.csv".format(fold)))
    model_dict['ResNet101-HSATBCL'] = pd.read_excel(os.path.join(dir_path, "resnet101_loss", "Fold{}.csv".format(fold)))
    model_dict['Vgg16-HSATBCL'] = pd.read_excel(os.path.join(dir_path, "Vgg16_loss", "Fold{}.csv".format(fold)))
    model_dict['PAHNet(our)'] = pd.read_excel(os.path.join(dir_path, "densenet121_loss", "Fold{}.csv".format(fold)))

    # Iterate over models and calculate ROC curve for each fold
    for model, data in model_dict.items():
        fpr = data['fpr']
        tpr = data['tpr']
        auc = (tpr.diff() * (fpr + fpr.diff() / 2)).sum()
        lower_bound_list = data['confidence_interval'][0]
        upper_bound_list = data['confidence_interval'][1]

        # If model key not present in mean_fpr_dict, initialize lists
        if model not in mean_fpr_dict:
            mean_fpr_dict[model] = []
            mean_tpr_dict[model] = []
            mean_lower_bound_dict[model] = []
            mean_upper_bound_dict[model] = []
            mean_auc_dict[model] = 0.0

        # Append fpr, tpr, and auc values to the respective lists
        mean_fpr_dict[model].append(fpr)
        mean_tpr_dict[model].append(tpr)
        mean_auc_dict[model] += auc
        mean_lower_bound_dict[model].append(lower_bound_list)
        mean_upper_bound_dict[model].append(upper_bound_list)


# Initialize empty lists to store interpolated fpr and tpr values for each model
interpolated_fpr_dict = {}
interpolated_tpr_dict = {}


for model in mean_fpr_dict:
    interpolated_fpr_dict[model] = []
    interpolated_tpr_dict[model] = []

    # Interpolate fpr and tpr values for each fold to match the maximum length
    for fold_fpr, fold_tpr in zip(mean_fpr_dict[model], mean_tpr_dict[model]):
        max_length = max(len(fpr) for fpr in mean_fpr_dict[model])
        interpolated_fpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fold_fpr)), fold_fpr)
        interpolated_tpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fold_tpr)), fold_tpr)
        interpolated_fpr_dict[model].append(interpolated_fpr)
        interpolated_tpr_dict[model].append(interpolated_tpr)
# Calculate mean interpolated fpr and tpr values
mean_interpolated_fpr_dict = {model: np.mean(interpolated_fpr_dict[model], axis=0) for model in interpolated_fpr_dict}
mean_interpolated_tpr_dict = {model: np.mean(interpolated_tpr_dict[model], axis=0) for model in interpolated_tpr_dict}

# Plot ROC curves for each model
plt.figure(figsize=(8, 6))

for model in mean_interpolated_fpr_dict:
    mean_fpr = mean_interpolated_fpr_dict[model]
    mean_tpr = mean_interpolated_tpr_dict[model]
    auc = mean_auc_dict[model]
    lower_bound = np.mean(mean_lower_bound_dict[model])
    upper_bound = np.mean(mean_upper_bound_dict[model])

    if model == 'PAHNet(our)':
        plt.plot(mean_fpr, mean_tpr,
                 label=f'{model} (AUC = {1 - (auc / 5):.3f}) 95%CI=[{lower_bound:.3f},{upper_bound:.3f}]',
                 linewidth=2.5, color='red')
    else:
        plt.plot(mean_fpr, mean_tpr,
                 label=f'{model} (AUC = {1 - (auc / 5):.3f}) 95%CI=[{lower_bound:.3f},{upper_bound:.3f}]')


# Set plot labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curves for Different Models')
plt.legend(loc='lower right')  # Add legend in the lower right corner
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.grid(True)  # Add grid lines for better readability
# plt.show()  # Display the plot
plt.savefig(r"G:\CMR-res\muti_center_data0927\Data_reword\ismsm會議統計\fig3_20231206.svg")

