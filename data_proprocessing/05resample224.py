import os
import glob
import numpy as np
import SimpleITK as sitk

# Set folder path and output path
file_dir = r"D:\CMR-res\muti_center_data0927\test_qinyuan\wangfu\crop_flow\wangfu"

def resample_image_by_size(ori_image, target_size, mode):
    """

    :param ori_imgae: itk 读取的图像
    :param target_size: 列表形式保存的目标尺寸
    :param mode: "sitk.sitkLinear" OR "sitk.sitkNearestNeighbor"
    :return:
    """

    ori_size = np.array(ori_image.GetSize())
    ori_spacing = np.array(ori_image.GetSpacing())
    target_spacing = ori_spacing * ori_size / np.array(target_size)

    resampler = sitk.ResampleImageFilter()  # Initialize filter
    resampler.SetReferenceImage(ori_image)  # Pass in the target image that needs to be resampled
    resampler.SetOutputDirection(ori_image.GetDirection())
    resampler.SetOutputOrigin(ori_image.GetOrigin())
    resampler.SetInterpolator(mode)  # Set interpolation method
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing([float(s) for s in target_spacing])

    itk_img_resampled = resampler.Execute(ori_image)  # Get the resampled image
    return itk_img_resampled

# Define a recursive function to loop through all subfolders and process NIfTI files within them
def process_folder(folder_path):
    for file_name in glob.glob(os.path.join(folder_path, "*.nii*")):
        # Extract filename from filename and read image
        file_split = os.path.split(file_name)[-1]
        image = sitk.ReadImage(file_name)
        img_array = sitk.GetArrayFromImage(image)
        channel, W, H = img_array.shape
        save_img = resample_image_by_size(ori_image=image, target_size=(224, 224, channel), mode=sitk.sitkLinear)


        # Save the results as a file in NIfTI format
        output_dir = os.path.join(file_dir, os.path.relpath(folder_path, file_dir))  # Construct output folder path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # If the output folder does not exist, create it
        sitk.WriteImage(save_img, os.path.join(output_dir, file_split))
        print("{} done".format(file_name))

    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            process_folder(subfolder_path)


# Call a recursive function to process NIfTI files in all subfolders
process_folder(file_dir)
