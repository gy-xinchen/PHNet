from scipy import ndimage
import numpy as np
import cv2
from net_hub.Densenet3D_embedding import densenet121
import torch
import SimpleITK as sitk
from PIL import Image

# 修改图像尺寸
def ndarray_resample3D_by_size(ori_array, target_size, order=2):
    """

    :param ori_array:
    :param targer_size:
    :param order: 0 represents nearest neighbor interpolation, 1 represents bilinear, 3 represents trilinear
    :return:
    """
    new_resampled_arr = []
    for i in range(ori_array.shape[0]):
        resampled_arr = ndimage.zoom(ori_array[i], target_size / np.array(ori_array[0].shape), order=order)
        new_resampled_arr.append(resampled_arr)
    new_resampled_arr = np.array(new_resampled_arr)
    return new_resampled_arr

# 获取对应特征图
class ActivationsAndGradients:
    """Register forward and reverse hook functions and forward propagation"""

    def __init__(self, model, target_layer, reshape_transform):
        """
        Register the forward and reverse hook functions in the target layer of the model, and propagate the model forward in call
        :param model:
        :param target_layers:
        :param reshape_transform:
        """
        self.model = model
        self.reshape_transform = reshape_transform
        self.activation = None
        self.gradient = None

        """注册正、反向钩子函数"""
        target_layer.register_forward_hook(self.save_activation)    # Register a forward hook function for the target layer,
                                                                    # which will be executed during forward propagation

        # Reverse hook function compatible with new and old versions of pytorch
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)
        """Register forward and reverse hook functions"""

    def save_activation(self, module, input, output):
        self.activation = output.cpu().detach()  # ndarray:[batch=1, channel, H, W]

    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].cpu().detach()  # ndarray:[batch=1, channel, H, W]

    def __call__(self, x):
        self.gradients = None
        self.activations = None
        return self.model(x)  # Forward propagation to obtain feature maps


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()  #Set the model to validation mode
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """



    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):  # Loop over the length of len(target_category) which is batch
            # Calculate the loss of the corresponding category for each batch, but generally the batch data are of the same target class
            loss = loss + output[0][i, target_category[i]]  # output:[batch, num_class]，Here indexes the output of the corresponding category
        return loss

    def get_cam_image(self, activations, grad):


        # ndarray:[batch=1, channel, D, H, W]
        weights = np.mean(a=grad, axis=(2, 3, 4), keepdims=True)

        # weighted sum
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam


    def compute_cam(self, ori_size):

        activation = self.activations_and_grads.activation.cpu().data.numpy()   # ndarray:[batch=16, channel, D, H, W]
        grad = self.activations_and_grads.gradient.cpu().data.numpy()   # ndarray:[batch=16, channel, D, H, W]

        cam = self.get_cam_image(activation, grad)  # Calculate CAM heat map (weighted sum of feature maps) ndarray:[batch=16, D, H, W]
        cam[cam < 0] = 0  # Relu

        scaled_cam = (cam - np.min(cam))/(1e-7 + np.max(cam))  # Scale to 0-1
        scaled_cam = ndarray_resample3D_by_size(ori_array=scaled_cam, target_size=ori_size, order=1)
        # scaled_cam = cv2.resize(scaled_cam[0], ori_size)  # order=1

        # scaled_cam = np.expand_dims(a=scaled_cam, axis=0)  # Add batch dimension

        return scaled_cam



    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        #  [batch=1, channel, H, W]
        output = self.activations_and_grads(input_tensor)

        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output[0].cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))




        self.model.zero_grad()
        loss = self.get_loss(output, target_category)  # 计算指定类的损失

        """
            Note that the loss here is not the difference between prediction and label during training, but the prediction score of the specified class
            Because the direction in which the score decreases is the direction in which the most relevant features of the feature category become smaller and tend to 0.
            Then the gradient is positive. The more relevant the feature is, the smaller the score will be, the smaller the score will be, that is, the greater the gradient. This gradient can be used as the weight
            Therefore, using the prediction score as the loss can calculate the weight of the most relevant features.
        """
        # loss.requires_grad = True
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        D, H, W = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
        cam = self.compute_cam(ori_size=(D, H, W))
        return cam



def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    normalized_array = (mask - mask.min()) / (mask.max() - mask.min())
    # Convert grayscale image array to 8-bit unsigned integer type
    images_uint8 = (normalized_array * 255).astype(np.uint8)
    # Create an empty array to store the pseudocolor image
    heatmaps = []
    # Apply color map to each grayscale image
    for imgdata in images_uint8:
        heatmap = cv2.applyColorMap(imgdata, cv2.COLORMAP_JET)/255
        heatmaps.append(heatmap)
    # heatmaps now contains the color-mapped image data, which is an array of shape (25, 224, 224, 3)
    heatmaps = np.array(heatmaps)

    cam_list = []
    for num in range(heatmaps.shape[0]):
        img = np.squeeze(img)
        img = (img - img.min()) / (img.max() - img.min())
        img_slice = img[num,:,:].cpu().numpy()
        img_slice = np.stack([img_slice] * 3, axis=-1)
        cam =  heatmaps[num,:,:] + 0.7 * img_slice  # Add the two and divide by the maximum value, or scale to 0-1
        cam = cam / np.max(cam)
        cam_list.append(cam)
    cam_list = np.array(cam_list)
    return np.uint8(255 * cam_list)  # Scale to 0-255 Return

if __name__ == "__main__":
    device = "cuda:0"
    model = densenet121(pretrained=False)
    # Load pretrained weights
    pretrained_dict = torch.load(r'G:\CMR-res\muti_center_data0927\Data_reword\test_for_layer\slice02\weight\lr3_10-4\Fold1.pth')  # 下载预训练权重并指定文件路径
    model_dict = model.state_dict()
    # Only load weights from the pretrained model that match the current model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    # Set loaded weights to model
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    input_image = r"G:\CMR-res\muti_center_data0927\mix_train_data\slice02_224x224\PAH\patient016_GP.nii.gz"
    input_tensor = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(input_image)), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)# 添加batch维度

    target_layer = model.features.denseblock4[-1]
    # Create Grad-CAM object
    gradcam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
    # Generate heat map
    cam = gradcam(input_tensor)
    # Overlay heatmap onto original image
    cam_on_image = show_cam_on_image(np.squeeze(input_tensor), cam[0])

    # Show superimposed images
    for i in range(25):
        img_slice = cam_on_image[i,:,:,:]
        # plt.imshow(cam_on_image[i,:,:,:])
        # plt.axis('off')
        # plt.show()
        # Convert NumPy array to PIL image object
        slice_image = Image.fromarray(img_slice)
        # Save image as PNG file
        output_path = r"G:\CMR-res\muti_center_data0927\Data_reword\CAM_Grad\patient016_fold1_slice02\output_image_{}.png".format(i)
        slice_image.save(output_path)