import os
import SimpleITK as sitk

def dicoms_to_nifti(input_folder, output_file):
    # Get all DICOM files in a folder
    dicom_files = [os.path.join(input_folder, file) for file in sorted(os.listdir(input_folder)) if file.endswith('.dcm')]

    # Read DICOM sequence
    dicom_reader = sitk.ImageSeriesReader()
    dicom_reader.SetFileNames(dicom_files)

    # Get spatial information of DICOM sequence
    dicom_image = dicom_reader.Execute()

    # Apply the spatial information of the DICOM sequence to the output image
    output_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(dicom_image))
    output_image.SetSpacing(dicom_image.GetSpacing())
    output_image.SetOrigin(dicom_image.GetOrigin())
    output_image.SetDirection(dicom_image.GetDirection())

    # Save DICOM sequence as NIfTI file
    sitk.WriteImage(output_image, output_file)

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
        output_nifti_files = os.path.join(output_path, patient_path[patient_id], "combined.nii.gz")
        # Check if the output folder exists, if not create it
        if not os.path.exists(output_nifti_path):
            os.makedirs(output_nifti_path)
        dicoms_to_nifti(input_folder_path, output_nifti_files)
