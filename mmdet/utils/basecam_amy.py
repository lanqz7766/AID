import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from mmdet.utils import get_root_logger
logger = get_root_logger()

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        try:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        except ValueError:  #raised if 'img' is empty
            pass

    result = np.float32(result)

    return result



class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")
#        return np.mean(grads, axis=(2, 3))

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
#            cam = weighted_activations.sum(axis=1)
            cam = weighted_activations
        logger.info(f'\nIn basecam, get_cam_image: {cam.shape}')
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        # logger.info(f'output of activation and grads are: {outputs[0].activations[0].size()}')
        # logger.info(f'output of activation and grads are: {outputs[0].gradients[0].size()}')
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        
        '''  # for one target layer
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        cam_channel = self.get_cam_image(input_tensor,
                                         self.target_layers[0],   ############
                                         targets,
                                         activations_list[0],     ############ 
                                         grads_list[0],           ############
                                         eigen_smooth)
        cam_channel = np.maximum(cam_channel, 0)
#        print(f'\nIn basecam.py, after max(cam_channel, 0): {cam_channel.shape}')   ############

        cam = np.squeeze(cam_channel, axis=0)
#        print(f'\nAfter squeeze(cam_channel, 0): {cam.shape}')    #############
        scaled = scale_cam_image(cam, target_size)
#        print(f'\nAfter scale_came: {scaled.shape}')
#        cam_per_target_layer = scaled[:, None, :]
#        print(f'\ncam_per_target_layer.shape: {cam_per_target_layer.shape}')
#        print(f'\nscaled[0,:,:].shape: {scaled[0, :, :].shape}')        
        return scaled               ######################
        '''        

############################
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)

#        print(f'\nAfter compute_cam_per_layer, len(cam_per_layer): {len(cam_per_layer), cam_per_layer[0].shape, cam_per_layer[1].shape}')
###################################
        return cam_per_layer
###################################
#        return self.aggregate_multi_layers(cam_per_layer)
    

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)
        logger.info(f'output of activation list and grads list are: {len(activations_list), len(grads_list)}')
        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
#            print(f'\nIn compute_cam_per_layer, cam.shape for {target_layer} layer: {cam.shape}')   ###############
#            print(f'layer_activations: {layer_activations.shape}      layer_grads: {layer_grads.shape}')   #############
            
            cam = np.maximum(cam, 0)
            # logger.info(f'\nAfter max(cam, 0): {cam.shape}')    #############
            cam = np.squeeze(cam, axis=0)
            # logger.info(f'\nAfter squeeze(cam, 0): {cam.shape}')    #############

            scaled = scale_cam_image(cam, target_size)
            # logger.info(f'\nAfter scaled(cam): {scaled.shape}')    #############

#            cam_per_target_layer.append(scaled[:, None, :])   # original
            cam_per_target_layer.append(scaled)    # modified
#            print(f'\nAfter scaled, expanded, & appended: {len(cam_per_target_layer), cam_per_target_layer[0].shape}')    #############

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
