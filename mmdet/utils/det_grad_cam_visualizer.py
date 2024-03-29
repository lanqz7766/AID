# Copyright (c) OpenMMLab. All rights reserved.
import bisect

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
import copy
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes

try:
    from pytorch_grad_cam import AblationCAM, AblationLayer, EigenCAM, GradCAM, ActivationsAndGradients
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def reshape_transform(feats, max_shape=(20, 20)):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.
    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if isinstance(feats, torch.Tensor):
        feats = [feats]

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations


class DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.detector = None
        self.device = device
        self.score_thr = score_thr
        self.cfg = copy.deepcopy(cfg)
        self.checkpoint = checkpoint
        self.set_return_loss_and_model(cfg=cfg)
        self.input_data = None
        self.img = None
        self.return_loss = False

    def set_input_data(self, img, bboxes=None, labels=None):
        self.img = img
        cfg = self.cfg.copy()
        if self.return_loss is False:
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            data = dict(img=self.img)
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
            test_pipeline = Compose(cfg.data.test.pipeline)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]

            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            else:
                for m in self.detector.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'
        else:
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
            cfg.data.test.pipeline[1].transforms[-1] = dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            test_pipeline = Compose(cfg.data.test.pipeline)
            data = dict(img=self.img, gt_bboxes=bboxes, gt_labels=labels, bbox_fields=['gt_bboxes'])
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0][0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]
            data['gt_bboxes'] = [gt_bboxes.data[0] for gt_bboxes in data['gt_bboxes']]
            data['gt_labels'] = [gt_labels.data[0] for gt_labels in data['gt_labels']]
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]

            data['img'] = data['img'][0]
            data['gt_bboxes'] = data['gt_bboxes'][0]
            data['gt_labels'] = data['gt_labels'][0]

        self.input_data = data

    def set_return_loss_and_model(self, return_loss=False, cfg=None):
        self.return_loss = return_loss

        if self.return_loss is False:
            self.detector = init_detector(cfg, self.checkpoint, device=self.device)
        else:
            self.detector = build_detector(
                self.cfg.model,
                train_cfg=self.cfg.get('train_cfg'),
                test_cfg=self.cfg.get('test_cfg'))

            if self.checkpoint is not None:
                checkpoint = load_checkpoint(self.detector, self.checkpoint, map_location='cpu')
                if 'CLASSES' in checkpoint.get('meta', {}):
                    self.detector.CLASSES = checkpoint['meta']['CLASSES']
                else:
                    import warnings
                    warnings.simplefilter('once')
                    warnings.warn('Class names are not saved in the checkpoint\'s '
                                  'meta data, use COCO classes by default.')
                    self.detector.CLASSES = get_classes('coco')

            self.detector = self.detector.to(self.device)

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.return_loss is False:
            results = self.detector(
                return_loss=self.return_loss, rescale=True, **self.input_data)[0]

            if isinstance(results, tuple):
                bbox_result, segm_result = results
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = results, None

            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            segms = None
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy()
                else:
                    segms = np.stack(segms, axis=0)

            if self.score_thr > 0:
                assert bboxes is not None and bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > self.score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]
            return [{"bboxes": bboxes, 'labels': labels, 'segms': segms}]
        else:
            loss = self.detector(
                return_loss=self.return_loss, **self.input_data)
            return [loss]


class DetCAMVisualizer:
    """mmdet cam visualization class.
    Args:
        method (str):  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        ablation_layer (torch.nn.Module): The ablation layer. Only
            used by AblationCAM method. Defaults to None.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
        batch_size (int): Batch of inference of AblationCAM. Only
            used by AblationCAM method. Defaults to 1.
        ratio_channels_to_ablate (float): The parameter controls how
            many channels should be ablated. Only used by
            AblationCAM method. Defaults to 0.1.
    """

    def __init__(self,
                 method,
                 model,
                 target_layers,
                 ablation_layer=None,
                 reshape_transform=None,
                 batch_size=1,
                 ratio_channels_to_ablate=0.1):
        if method == 'ablationcam':
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=ablation_layer,
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        elif method == 'eigencam':
            self.cam = EigenCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
            )
        elif method == 'featmapam':
            self.cam = FeatmapAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform
            )
        elif method == 'gradcam':
            self.cam = GradCAM(model,
                               target_layers,
                               use_cuda=True if 'cuda' in model.device else False,
                               reshape_transform=reshape_transform
                               )

        else:
            raise NotImplementedError(
                f'{method} cam calculation method is not supported')

        self.classes = model.detector.CLASSES
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                         x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(boxes, labels,
                                                     cam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image


class DetBoxScoreTarget:
    """For every original detected bounding box specified in "bboxes",
    assign a score on how the current bounding boxes match it,
        1. In Bbox IoU
        2. In the classification score.
        3. In Mask IoU if ``segms`` exist.
    If there is not a large enough overlap, or the category changed,
    assign a score of 0.
    The total score is the sum of all the box scores.
    """

    def __init__(self,
                 bboxes,
                 labels,
                 segms=None,
                 match_iou_thr=0.5,
                 device='cuda:0'):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr

        self.device = device

    def __call__(self, results):
        if 'loss_cls' in results:
            loss = sum(_value for _key, _value in results.items()
                       if 'loss' in _key)
            return loss

        output = torch.tensor([0], device=self.device)
        if len(results["bboxes"]) == 0:
            return output

        pred_bboxes = torch.from_numpy(results["bboxes"]).to(self.device)
        pred_labels = results["labels"]
        pred_segms = results["segms"]

        if pred_segms is not None:
            pred_segms = torch.from_numpy(pred_segms).to(self.device)

        for focal_box, focal_label, focal_segm in zip(self.focal_bboxes,
                                                      self.focal_labels,
                                                      self.focal_segms):
            ious = torchvision.ops.box_iou(focal_box[None],
                                           pred_bboxes[..., :4])
            index = ious.argmax()
            if ious[0, index] > self.match_iou_thr and pred_labels[
                index] == focal_label:
                # TODO: Adaptive adjustment of weights based on algorithms
                score = ious[0, index] + pred_bboxes[..., 4][index]
                output = output + score

                if focal_segm is not None and pred_segms is not None:
                    segms_score = (focal_segm * pred_segms[index]).sum() / (
                            focal_segm.sum() + pred_segms[index].sum() + 1e-7)
                    output = output + segms_score
        return output


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.
    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)