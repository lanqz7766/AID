# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from .. import build_detector
from mmdet.core import bbox2result

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from mmdet.utils import collect_env, get_root_logger

# logger = get_root_logger()

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


@DETECTORS.register_module()
class ATTLAD(KnowledgeDistillationSingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 # teacher_config,
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_ckpt,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KnowledgeDistillationSingleStageDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)
        self.eval_teacher = eval_teacher
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_model.bbox_head = build_head(teacher_bbox_head)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.adaptation_type = '1x1conv'
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        #   self.roi_adaptation_layer = nn.Conv2d(256, 256, kernel_size=1)
        if self.adaptation_type == '3x3conv':
            #   3x3 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ])
        if self.adaptation_type == '1x1conv':
            #   1x1 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            ])

        if self.adaptation_type == '3x3conv+bn':
            #   3x3 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        if self.adaptation_type == '1x1conv+bn':
            #   1x1 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])

        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])

    @property
    def with_teacher_neck(self):
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self.teacher_model, 'neck') and \
            self.teacher_model.neck is not None

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

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
        # get label assignment from the teacher
        with torch.no_grad():
            x_teacher = self.extract_teacher_feat(img)
            outs_teacher = self.teacher_model.bbox_head(x_teacher)
            label_assignment_results = \
                self.teacher_model.bbox_head.get_label_assignment(
                    *outs_teacher, gt_bboxes, gt_labels, img_metas,
                    gt_bboxes_ignore)

        # the student use the label assignment from the teacher to learn
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, label_assignment_results,
                                              img_metas, gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        # logger.info(f"losses are: {losses.keys()}")
        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0

        #   for channel attention
        c_t = 0.1
        c_s_ratio = 1.0
        with torch.no_grad():
            t_feats = self.extract_teacher_feat(img)
               
        if t_feats is not None:
            for _i in range(len(t_feats)):
                t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [1], keepdim=True)
                size = t_attention_mask.size()
                t_attention_mask = t_attention_mask.view(x[0].size(0), -1)
                t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
                t_attention_mask = t_attention_mask.view(size)

                s_attention_mask = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size = s_attention_mask.size()
                s_attention_mask = s_attention_mask.view(x[0].size(0), -1)
                s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
                s_attention_mask = s_attention_mask.view(size)

                c_t_attention_mask = torch.mean(torch.abs(t_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_t_attention_mask.size()
                c_t_attention_mask = c_t_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * 256
                c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                c_s_attention_mask = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size = c_s_attention_mask.size()
                c_s_attention_mask = c_s_attention_mask.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * 256
                c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
                sum_attention_mask = sum_attention_mask.detach()

                c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask = c_sum_attention_mask.detach()

                kd_feat_loss += dist2(t_feats[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask,
                                      channel_attention_mask=c_sum_attention_mask) * 7e-5*4 #7e-5 * 6
                kd_channel_loss += torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                              self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3*4 #4e-3 * 6
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                   t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))
                kd_spatial_loss += torch.dist(t_spatial_pool,
                                              self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3*4 #4e-3 * 6

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        return losses
