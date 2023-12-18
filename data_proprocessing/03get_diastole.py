import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Read NIfTI files
nii_file_path = r'D:\CMR-res\muti_center_data0927\test_qinyuan\wangfu\nii'
output_path = r"D:\CMR-res\muti_center_data0927\test_qinyuan\wangfu\diastole"
patient_file = os.listdir(nii_file_path)
for patient_id in range(len(patient_file)):
    patient_path = os.path.join(nii_file_path, patient_file[patient_id])
    layer_file = os.listdir(patient_path)
    diastole_list = []
    for layer_id in range(len(layer_file)):
        layer_nii = os.path.join(patient_path, layer_file[layer_id])
        image = sitk.ReadImage(layer_nii)
        # Get image array
        image_array = sitk.GetArrayFromImage(image)
        # Extract the first image on the Z axis
        z_slice = image_array[0, :, :]
        diastole_list.append(z_slice)
    diastole_list = np.array(diastole_list)
    output_nii = sitk.GetImageFromArray(diastole_list)
    diastole_output_path = os.path.join(output_path, patient_file[patient_id]+".nii.gz")
    sitk.WriteImage(output_nii, diastole_output_path)