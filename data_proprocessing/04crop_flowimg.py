import SimpleITK as sitk
import os
import cv2
import numpy as np


raw_path = r""
input_diastolemask_path = r""
input_flow_img = r""
crop_flow_output = r""

diastole_nii = os.listdir(input_diastolemask_path)
input_flow_nii = os.listdir(input_flow_img)
max_x, max_y, max_w, max_h = 0, 0, 0, 0
max_minrect = 0
# Define an empty list to save x, y, w, h
bounding_rect_list = []
for nii in range(len(diastole_nii)):
    diastole_nii_file = os.path.join(input_diastolemask_path, diastole_nii[nii])
    # Get a patient's diastole_mask
    diastole_nii_img = sitk.ReadImage(diastole_nii_file)
    diastole_nii_array = sitk.GetArrayFromImage(diastole_nii_img)
    depth, height, width = diastole_nii_array.shape
    for layer in range(depth):
        # Get the sequence mask pixel value and count the number of white pixels
        area_list = []
        for index in range(depth):
            mask_list = diastole_nii_array[index, :, :]
            t, rst = cv2.threshold(mask_list, 0, maxval=255, type=cv2.THRESH_BINARY)
            area = 0
            for j in range(width):
                for i in range(height):
                    if rst[i, j] == 255:
                        area += 1
            area_list.append(area)
        max_value = max(area_list)
        max_area_value = area_list.index(max_value)
        t, max_rst = cv2.threshold(diastole_nii_array[max_area_value,:,:], 0, maxval=1, type=cv2.THRESH_BINARY)
        # find contours
        contours, _ = cv2.findContours(max_rst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get the first contour (assuming there is only one contour,
        # if there are multiple contours, they need to be processed according to the actual situation)

        largest_contour = max(contours, key=cv2.contourArea)
        # Get the minimum bounding rectangle
        min_rect = cv2.minAreaRect(largest_contour)
        # Calculate the area of the smallest enclosing rectangle
        min_rect_area = min_rect[1][0] * min_rect[1][1]
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Save the current x, y, w, h to the list
        if layer < 1:
            bounding_rect_list.append((x, y, w, h))
        if min_rect_area > max_minrect:
            max_minrect = min_rect_area
            max_x, max_y, max_w, max_h = cv2.boundingRect(largest_contour)

for flow_patients in range(len(input_flow_nii)):
    flow_patients_path = os.path.join(input_flow_img, input_flow_nii[flow_patients])
    flow_layers = os.listdir(flow_patients_path)
    diastole_patient_num = os.listdir(input_diastolemask_path)
    diastole_patient_path = os.path.join(input_diastolemask_path, diastole_patient_num[flow_patients])
    diastole_patient_img = sitk.ReadImage(diastole_patient_path)
    diastole_patient_array = sitk.GetArrayFromImage(diastole_patient_img)
    for flow_nii in range(len(flow_layers)):
        flow_nii_path = os.path.join(flow_patients_path, flow_layers[flow_nii])
        flow_nii_img = sitk.ReadImage(flow_nii_path)
        flow_nii_array = sitk.GetArrayFromImage(flow_nii_img)
        diastole_patient_layer_array = diastole_patient_array[flow_nii, :, :]
        t, diastole_patient_layer_array_rst = cv2.threshold(diastole_patient_layer_array, 0, maxval=1, type=cv2.THRESH_BINARY)
        # Expansion mask operation
        # Define inflated structural elements (kernel)
        kernel = np.ones((7, 7), np.uint8)
        # Perform expansion operation
        dilated_mask = cv2.dilate(diastole_patient_layer_array_rst, kernel, iterations=4)
        multiply_img = np.multiply(flow_nii_array, dilated_mask)
        target_w = max_w
        target_h = max_h
        x, y, w, h = bounding_rect_list[flow_patients]
        start_y = (target_h - h) // 2
        start_x = (target_w - w) // 2
        # Crop image
        cropped_image = multiply_img[:, (y-start_y):(y-start_y) + target_h, (x-start_x):(x-start_x) + target_w]
        cropped_image = sitk.GetImageFromArray(cropped_image)
        cropped_save_path = os.path.join(crop_flow_output, input_flow_nii[flow_patients])
        if not os.path.exists(cropped_save_path):
            os.makedirs(cropped_save_path)
        save_name = os.path.join(cropped_save_path, flow_layers[flow_nii])
        sitk.WriteImage(cropped_image, save_name)








