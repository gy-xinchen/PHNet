import os
import SimpleITK as sitk
import numpy as np

def zscore_normalize(image_array):
    mean_value = np.mean(image_array)
    std_dev = np.std(image_array)
    normalized_array = (image_array - mean_value) / std_dev
    return normalized_array

def process_nii_file(file_path):
    # Read nii file
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)

    # Perform Z-score normalization
    normalized_array = zscore_normalize(image_array)

    # Reconvert normalized array to SimpleITK image
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)

    # Save the normalized nii file
    output_path = file_path.replace('.nii.gz', '.nii.gz')
    sitk.WriteImage(normalized_image, output_path)

def process_folder(folder_path):
    # Iterate over files and subfolders in a folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                process_nii_file(file_path)

# Specify the folder to process
input_folder = r'D:\CMR-res\muti_center_data0927\test_qinyuan\wangfu\crop_flow\wangfu'

# Process all nii files in a folder
process_folder(input_folder)