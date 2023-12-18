import os
import SimpleITK as sitk

def resample_and_save_dcm_in_folder(input_folder, output_folder, target_spacing=(1.0, 1.0, 1.0)):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files and subfolders in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".dcm"):
                # Build full file path
                file_path = os.path.join(root, file_name)

                # Read DICOM files
                image = sitk.ReadImage(file_path)

                # Get the spacing of the original image
                original_spacing = image.GetSpacing()
                print(f"Processing {file_name} - Original Spacing: {original_spacing}")

                # Calculate resampled dimensions
                original_size = image.GetSize()
                target_size = [int(round(osz * osp / tsp)) for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)]

                # Create resampling object
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(target_size)
                resampler.SetOutputSpacing(target_spacing)
                resampler.SetOutputOrigin(image.GetOrigin())
                resampler.SetOutputDirection(image.GetDirection())
                resampler.SetInterpolator(sitk.sitkLinear)  # 使用线性插值

                # Perform resampling
                resampled_image = resampler.Execute(image)

                # Build the saved file path, keeping the filename the same
                relative_path = os.path.relpath(file_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)

                # Save the resampled image
                sitk.WriteImage(resampled_image, output_path)
                print(f"Resampled image saved at: {output_path}")

# Call the function, passing in the input folder and output folder paths
input_folder_path = r"D:\CMR-res\muti_center_data0927\test_qinyuan"
output_folder_path = r"D:\CMR-res\muti_center_data0927\test_qinyuan"
resample_and_save_dcm_in_folder(input_folder_path, output_folder_path)
