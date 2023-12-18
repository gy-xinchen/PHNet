import os
import SimpleITK as sitk
import numpy as np


def dicoms_to_nifti(input_folder, output_file):
    # Get all DICOM files in a folder
    dcm_files = os.listdir(input_folder)
    dcm_list = []
    for i in range(len(dcm_files)):
        # Read DICOM sequence
        dicom_reader = sitk.ReadImage(os.path.join(input_folder, dcm_files[i]))
        # Get image array
        image_array = sitk.GetArrayFromImage(dicom_reader)
        dcm_list.append(image_array[0,:,:])
    dcm_list = np.array(dcm_list)
    dcm_list = sitk.GetImageFromArray(dcm_list)

    # Save DICOM sequence as NIfTI file
    sitk.WriteImage(dcm_list, output_file)

    print(f"Converted DICOMs to 3D NIfTI: {output_file}")

# Input DICOM folder path and output NIfTI file path
input_path = r"D:\CMR-res\muti_center_data0927\test_qinyuan\wangfu"
output_path = r"D:\CMR-res\muti_center_data0927\test_qinyuan\wangfu\nii"
patient_path = os.listdir(input_path)

for patient_id in range(len(patient_path)):
    patient_layers_path = os.path.join(input_path, patient_path[patient_id])
    patient_layers = os.listdir(patient_layers_path)
    for layers in range(len(patient_layers)):
        input_folder_path = os.path.join(input_path, patient_path[patient_id], patient_layers[layers])
        output_nifti_path = os.path.join(output_path, patient_path[patient_id])
        output_nifti_files = os.path.join(output_path, patient_layers[layers]+".nii.gz")
        # Check if the output folder exists, if not create it
        if not os.path.exists(output_nifti_path):
            os.makedirs(output_nifti_path)
        dicoms_to_nifti(input_folder_path, output_nifti_files)
