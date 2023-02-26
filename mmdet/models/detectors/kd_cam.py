# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from .. import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.utils import collect_env, get_root_logger
import torch.nn as nn
import torch.nn.functional as F
import os.path
from functools import partial
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from mmcv import Config, DictAction
from collections import OrderedDict
import numpy as np
import cv2
from mmdet.utils.det_cam_visualizer import (DetAblationLayer, Student_DetCAMModel,
                                            DetBoxScoreTarget, DetCAMModel,
                                            DetCAMVisualizer, EigenCAM, BaseCAM,
                                            FeatmapAM, reshape_transform, GradCAM)

try:
    from pytorch_grad_cam import (AblationCAM, EigenGradCAM, GradCAMPlusPlus, LayerCAM, XGradCAM)
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

logger = get_root_logger()


GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
    'featmapam': FeatmapAM
}

GRAD_BASE_METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
    'basecam': BaseCAM
}

ALL_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys())

def init_model_cam(cfg,
                   detector,
                   checkpoint,
                   target_layers_, 
                   max_shape = -1, 
                   method = 'gradcam', 
                   score_thr = 0.3, 
                   device = 'cuda:0',
                   student = False):
    if student == False:
        model = DetCAMModel(
            cfg, checkpoint, score_thr, device=device)
    else:
        model = Student_DetCAMModel(
            cfg, checkpoint, detector, score_thr, device=device)

    # if args.preview_model:
    #     print(model.detector)
    #     print('\n Please remove `--preview-model` to get the CAM.')
    #     return

    target_layers = []
    for target_layer in target_layers_:
        try:
            target_layers.append(eval(f'model.detector.{target_layer}'))
        except Exception as e:
            print(model.detector)
            raise RuntimeError('layer does not exist', e)

    extra_params = {
        'batch_size': 1,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': 0.5
    }

    if method in GRAD_BASE_METHOD_MAP:
        method_class = GRAD_BASE_METHOD_MAP[method]
        is_need_grad = True
        no_norm_in_bbox = False
        assert no_norm_in_bbox is False, 'If not norm in bbox, the ' \
                                              'visualization result ' \
                                              'may not be reasonable.'
    else:
        method_class = GRAD_FREE_METHOD_MAP[method]
        is_need_grad = False

    max_shape = max_shape
    if not isinstance(max_shape, list):
        max_shape = [max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    det_cam_visualizer = DetCAMVisualizer(
        method_class,
        model,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=extra_params)
    return model, det_cam_visualizer


@DETECTORS.register_module()
class CAMKnowledgeDistillationSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 student_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])

        if isinstance(student_config, str):
            student_config = mmcv.Config.fromfile(student_config)
        self.student= build_detector(student_config['model'],
                                        train_cfg=student_config.get('train_cfg'),
                                        test_cfg=student_config.get('test_cfg'))
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.teacher_model.cuda()
        logger.info(f'keys are: {self.__dict__.keys()}')

        self.teacher_cam_model, self.teacher_det_cam_visualizer = init_model_cam(cfg = teacher_config,
                                                         detector = None,
                                                         checkpoint = teacher_ckpt,
                                                         target_layers_ = [
                                                         'neck.fpn_convs[4].conv', 
                                                         'neck.fpn_convs[3].conv',
                                                         'neck.fpn_convs[2].conv',
                                                         'neck.fpn_convs[1].conv',
                                                         'neck.fpn_convs[0].conv'
                                                         ], 
                                                         max_shape = -1, 
                                                         method = 'gradcam', 
                                                         score_thr = 0.3,
                                                         device = 'cuda:0',
                                                         student = False)
        self.student_cam_model, self.student_det_cam_visualizer = init_model_cam(cfg = teacher_config,
                                                         detector = self.student,
                                                         checkpoint = teacher_ckpt,
                                                         target_layers_ = [
                                                         'neck.fpn_convs[4].conv', 
                                                         'neck.fpn_convs[3].conv',
                                                         'neck.fpn_convs[2].conv',
                                                         'neck.fpn_convs[1].conv',
                                                         'neck.fpn_convs[0].conv'
                                                         ], 
                                                         max_shape = -1, 
                                                         method = 'gradcam', 
                                                         score_thr = 0.3,
                                                         device = 'cuda:0',
                                                         student = True)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        current_weight = self.student.state_dict()
        self.student_cam_model.update_model_weights(current_weight) 
        N, c, h, w = img.shape
        images = []
        teacher_cam_masks = []
        student_cam_masks = []
        topk=10
        method = 'gradcam'
        # logger.info(f"target_layer is: {self.det_cam_visualizer.target_layers}")
        for i in range(N):
            images.append(img_metas[i].get('filename'))
        for image_path in images:
            image = cv2.imread(image_path)
            self.teacher_cam_model.set_input_data(image)
            self.student_cam_model.set_input_data(image)
            teacher_result = self.teacher_cam_model()[0]
            student_result = self.student_cam_model()[0]

            teacher_bboxes = teacher_result['bboxes'][..., :4]
            teacher_scores = teacher_result['bboxes'][..., 4]
            teacher_labels = teacher_result['labels']
            teacher_segms = teacher_result['segms']

            student_bboxes = student_result['bboxes'][..., :4]
            student_scores = student_result['bboxes'][..., 4]
            student_labels = student_result['labels']
            student_segms = student_result['segms']
            # assert bboxes is not None and len(bboxes) > 0
            if topk > 0:
                teacher_idxs = np.argsort(-teacher_scores)
                student_idxs = np.argsort(-student_scores)
                teacher_bboxes = teacher_bboxes[teacher_idxs[:topk]]
                student_bboxes = student_bboxes[student_idxs[:topk]]
                teacher_labels = teacher_labels[teacher_idxs[:topk]]
                student_labels = student_labels[student_idxs[:topk]]
                if teacher_segms is not None:
                    teacher_segms = teacher_segms[teacher_idxs[:topk]]
                    student_segms = student_segms[student_idxs[:topk]]
            teacher_targets = [
                DetBoxScoreTarget(bboxes=teacher_bboxes, labels=teacher_labels, segms=teacher_segms)
            ]
            student_targets = [
                DetBoxScoreTarget(bboxes=student_bboxes, labels=student_labels, segms=student_segms)
            ]

            if method in GRAD_BASE_METHOD_MAP:
                self.teacher_cam_model.set_return_loss(True)
                self.teacher_cam_model.set_input_data(image, bboxes=teacher_bboxes, labels=teacher_labels)
                self.teacher_det_cam_visualizer.switch_activations_and_grads(self.teacher_cam_model)
                self.student_cam_model.set_return_loss(True)
                self.student_cam_model.set_input_data(image, bboxes=student_bboxes, labels=student_labels)
                self.student_det_cam_visualizer.switch_activations_and_grads(self.student_cam_model)

            teacher_grayscale_cam = self.teacher_det_cam_visualizer(
                image,
                targets=teacher_targets,
                aug_smooth=False,
                eigen_smooth=False) # (1, 375, 1242)

            student_grayscale_cam = self.student_det_cam_visualizer(
                image,
                targets=student_targets,
                aug_smooth=False,
                eigen_smooth=False) # (1, 375, 1242)


            if method in GRAD_BASE_METHOD_MAP:
                self.teacher_cam_model.set_return_loss(False)
                self.teacher_det_cam_visualizer.switch_activations_and_grads(self.teacher_cam_model)
                self.student_cam_model.set_return_loss(False)
                self.student_det_cam_visualizer.switch_activations_and_grads(self.student_cam_model)

            teacher_cam_masks.append(teacher_grayscale_cam)
            student_cam_masks.append(student_grayscale_cam)

        teacher_cam_mask_per_layer = []
        student_cam_mask_per_layer = []
        for i in range(0,5):
            teacher_cam_mask_per_layer.append(torch.tensor([item[i] for item in teacher_cam_masks]))
            student_cam_mask_per_layer.append(torch.tensor([item[i] for item in student_cam_masks]))


        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        ######
        t=0.5

        with torch.no_grad():
            t_feats = self.teacher_model.extract_feat(img)
               
        if t_feats is not None:
            for _i in range(len(t_feats)):
                N,C,H,W = preds_S.shape
                s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size = s_attention_mask.size()
                s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
                S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

                Mask_fg = torch.zeros_like(S_attention)

                wmin,wmax,hmin,hmax = [],[],[],[]
                big_map = torch.zeros(N, H*2, W*2).cuda()
                for i in range(N):
                    new_boxxes = torch.ones_like(gt_bboxes[i])
                    new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
                    new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
                    new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
                    new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

                    wmin.append(torch.floor(new_boxxes[:, 0]).int())
                    wmax.append(torch.ceil(new_boxxes[:, 2]).int())
                    hmin.append(torch.floor(new_boxxes[:, 1]).int())
                    hmax.append(torch.ceil(new_boxxes[:, 3]).int())
                    for j in range(len(gt_bboxes[i])):
                        w = (wmax[i][j]-wmin[i][j]).int()
                        h = (hmax[i][j]-hmin[i][j]).int()
                        big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]] = \
                                torch.maximum(big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]], self.calculate_gaussian(w, h))
                    gap_W = np.floor((2*W-W)/2).astype(int)
                    gap_H = np.floor((2*H-H)/2).astype(int)
                    Mask_fg[i] = big_map[i][gap_H:gap_H+H, gap_W:gap_W+W]
                Mask_cam_T = torch.squeeze(teacher_cam_mask_per_layer[_i]).cuda()
                Mask_cam_S = torch.squeeze(student_cam_mask_per_layer[_i]).cuda()
                # fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, Mask_cam_T, Mask_cam_S,
                                   # C_attention_s, C_attention_t, S_attention_s, S_attention_t)
                # mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
                # rela_loss = self.get_rela_loss(preds_S, preds_T)
                cam_mask_loss += self.get_cam_mask_loss(Mask_cam_T, Mask_cam_S, Mask_fg)
        losses.update({'kd_cam_mask_loss': 0.05*cam_mask_loss})
        return losses

    def get_cam_mask_loss(self, Mask_cam_T, Mask_cam_S, Mask_fg):

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_cam_T = Mask_cam_T.unsqueeze(dim=1)
        Mask_cam_S = Mask_cam_S.unsqueeze(dim=1)

        weighted_mask_cam_T = torch.mul(Mask_cam_T, torch.sqrt(Mask_fg))
        weighted_mask_cam_S = torch.mul(Mask_cam_S, torch.sqrt(Mask_fg))
        cam_mask_loss = torch.sum(torch.abs(weighted_mask_cam_T - weighted_mask_cam_S))/len(weighted_mask_cam_S)

        return cam_mask_loss

    def calculate_gaussian(self, old_w, old_h):
        w = 2*old_w
        h = 2*old_h
        cx = w / 2
        cy = h / 2
        if cx == 0:
            cx += 1
        if cy == 0:
            cy += 1
        x0 = cx.repeat(1, w)
        y0 = cy.repeat(h, 1)
        x = torch.arange(w).cuda()
        y = torch.unsqueeze(torch.arange(h), dim=1).cuda()
        gaussian_mask = torch.exp(-0.5*((x-x0)/cx)**2) * torch.exp(-0.5*((y-y0)/cy)**2)
        bbox = torch.ones((old_h,old_w)).cuda()
        gap_w = torch.floor((w-old_w)/2).int()
        gap_h = torch.floor((h-old_h)/2).int()
        gaussian_mask[gap_h:gap_h+old_h, gap_w:gap_w+old_w] = bbox
        gaussian_mask = 1/w/h*gaussian_mask.double().cuda()
        return gaussian_mask

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        self.stundent_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
