# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector
from .single_stage import SingleStageDetector

import torch.nn as nn
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.utils import collect_env, get_root_logger
import time
import os.path as osp
#        add loogers to debug
#        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#        log_file = osp.join(.work_dir, f'{timestamp}.log')
#logger = get_root_logger()

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


@DETECTORS.register_module()
class KDAttention(SingleStageDetector):
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
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        ## add adaptation layers
        self.adaptation_type = '1x1conv'
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #   self.cls_adaptation = nn.Linear(1024, 1024)
        #   self.reg_adaptation = nn.Linear(1024, 1024)
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

        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])

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
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
#        losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
#                                              gt_bboxes, gt_labels,
#                                              gt_bboxes_ignore)

        losses = self.bbox_head.forward_train(x, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
#        logger.info(f"losses are: {losses.keys()}") 
#        logger.info(f"losses values are: {losses['loss_cls'], losses['loss_bbox']}")
#        weight = torch.exp(-losses['loss_cls'][0] - losses ['loss_bbox'][0])
#        logger.info(f"weight values are: {weight}")
#        with torch.no_grad():
#            teacher_x = self.teacher_model.extract_feat(img)
#            out_teacher = self.teacher_model.bbox_head(teacher_x)

#        cls_quality_score, bbox_pred = self.teacher_model.bbox_head.forward(teacher_x)
#        loss_test = self.teacher_model.bbox_head.loss(cls_quality_score,
#                                                      bbox_pred,
#                                                      gt_bboxes, gt_labels,
#                                                      img_metas, gt_bboxes_ignore)
        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0

        #   for channel attention
        c_t = 0.1
        c_s_ratio = 1.0
        with torch.no_grad():
            t_feats = self.teacher_model.extract_feat(img)
               
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
                                      channel_attention_mask=c_sum_attention_mask) * 7e-5*6 #7e-5 * 6
                kd_channel_loss += torch.dist(torch.mean(t_feats[_i], [2, 3]),
                                              self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3*6 #4e-3 * 6
                t_spatial_pool = torch.mean(t_feats[_i], [1]).view(t_feats[_i].size(0), 1, t_feats[_i].size(2),
                                                                   t_feats[_i].size(3))
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))
                kd_spatial_loss += torch.dist(t_spatial_pool,
                                              self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3*6 #4e-3 * 6

        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})

        kd_nonlocal_loss = 0
        with torch.no_grad():
            t_feats = self.teacher_model.extract_feat(img)

        if t_feats is not None:
            for _i in range(len(t_feats)):
#                w = torch.exp(-0.1*(losses['loss_cls'][_i] + losses ['loss_bbox'][_i]))
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                #   print(s_relation.size())
                kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
        losses.update(kd_nonlocal_loss=kd_nonlocal_loss * 7e-5*6) ##7e-5 * 6

        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
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
