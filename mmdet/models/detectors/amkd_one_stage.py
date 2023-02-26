# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector
import torch.nn as nn
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.utils import collect_env, get_root_logger
#        add loogers to debug

#logger = get_root_logger()
def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff

@DETECTORS.register_module()
class AMKnowledgeDistillationSingleStageDetector(SingleStageDetector):
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
                 teacher_config_1,
                 teacher_config_2,
                 teacher_config_3,
                 teacher_ckpt_1=None,
                 teacher_ckpt_2=None,
                 teacher_ckpt_3=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        
        if isinstance(teacher_config_1, str):
            teacher_config_1 = mmcv.Config.fromfile(teacher_config_1)
        self.teacher_model_1 = build_detector(teacher_config_1['model'])
        
        if isinstance(teacher_config_2, str):
            teacher_config_2 = mmcv.Config.fromfile(teacher_config_2)
        self.teacher_model_2 = build_detector(teacher_config_2['model'])
        
        if isinstance(teacher_config_3, str):
            teacher_config_3 = mmcv.Config.fromfile(teacher_config_3)
        self.teacher_model_3 = build_detector(teacher_config_3['model'])  

        if teacher_ckpt_1 is not None:
            load_checkpoint(
                self.teacher_model_1, teacher_ckpt_1, map_location='cpu')
                
        if teacher_ckpt_2 is not None:
            load_checkpoint(
                self.teacher_model_2, teacher_ckpt_2, map_location='cpu')
                
        if teacher_ckpt_3 is not None:
            load_checkpoint(
                self.teacher_model_3, teacher_ckpt_3, map_location='cpu')

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
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x_1 = self.teacher_model_1.extract_feat(img)
            losses_T_1 = self.teacher_model_1.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            out_teacher_1 = self.teacher_model_1.bbox_head(teacher_x_1)

        with torch.no_grad():
            teacher_x_2 = self.teacher_model_2.extract_feat(img)
            losses_T_2 = self.teacher_model_2.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            out_teacher_2 = self.teacher_model_2.bbox_head(teacher_x_2)

        with torch.no_grad():
            teacher_x_3 = self.teacher_model_3.extract_feat(img)
            losses_T_3 = self.teacher_model_3.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
            out_teacher_3 = self.teacher_model_3.bbox_head(teacher_x_3)
        
            
        losses = self.bbox_head.forward_train(x, out_teacher_1, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
                                              
        losses_2 = self.bbox_head.forward_train(x, out_teacher_2, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)  
                                              
        losses_3 = self.bbox_head.forward_train(x, out_teacher_3, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
                                              
        
        for i in range(len(x)):
            T_loss_1 = losses_T_1['loss_cls'][i] + losses_T_1['loss_bbox'][i]
            T_loss_2 = losses_T_2['loss_cls'][i] + losses_T_2['loss_bbox'][i]
            T_loss_3 = losses_T_3['loss_cls'][i] + losses_T_3['loss_bbox'][i]
            if T_loss_1 <= T_loss_2 and T_loss_1 <= T_loss_3:
                losses['loss_ld'][i] = torch.exp(-0.1*(losses_T_1['loss_cls'][i]+ losses_T_1['loss_bbox'][i])) * losses['loss_ld'][i]
            elif T_loss_2 <= T_loss_1 and T_loss_2 <= T_loss_3:
                losses['loss_ld'][i] = torch.exp(-0.1*(losses_T_2['loss_cls'][i]+ losses_T_2['loss_bbox'][i])) * losses_2['loss_ld'][i]
            else:
                losses['loss_ld'][i] = torch.exp(-0.1*(losses_T_3['loss_cls'][i]+ losses_T_3['loss_bbox'][i])) * losses_3['loss_ld'][i]
        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model_1.cuda(device=device)
        self.teacher_model_2.cuda(device=device)
        self.teacher_model_3.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model_1.train(False)
            self.teacher_model_2.train(False)
            self.teacher_model_3.train(False)
        else:
            self.teacher_model_1.train(mode)
            self.teacher_model_2.train(mode)
            self.teacher_model_3.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model_1' or name == 'teacher_model_2':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
