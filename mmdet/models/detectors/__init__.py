# Copyright (c) OpenMMLab. All rights reserved.
from .atss import ATSS
from .autoassign import AutoAssign
from .base import BaseDetector
from .cascade_rcnn import CascadeRCNN
from .centernet import CenterNet
from .cornernet import CornerNet
from .deformable_detr import DeformableDETR
from .detr import DETR
from .fast_rcnn import FastRCNN
from .faster_rcnn import FasterRCNN
from .fcos import FCOS
from .fovea import FOVEA
from .fsaf import FSAF
from .gfl import GFL
from .grid_rcnn import GridRCNN
from .htc import HybridTaskCascade
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from .lad import LAD
from .mask_rcnn import MaskRCNN
from .mask_scoring_rcnn import MaskScoringRCNN
from .nasfcos import NASFCOS
from .paa import PAA
from .panoptic_fpn import PanopticFPN
from .panoptic_two_stage_segmentor import TwoStagePanopticSegmentor
from .point_rend import PointRend
from .queryinst import QueryInst
from .reppoints_detector import RepPointsDetector
from .retinanet import RetinaNet
from .rpn import RPN
from .scnet import SCNet
from .single_stage import SingleStageDetector
from .solo import SOLO
from .sparse_rcnn import SparseRCNN
from .tood import TOOD
from .trident_faster_rcnn import TridentFasterRCNN
from .two_stage import TwoStageDetector
from .vfnet import VFNet
from .yolact import YOLACT
from .yolo import YOLOV3
from .yolof import YOLOF
from .yolox import YOLOX
from .kd_attention import KDAttention ## added
from .kd_att_sw import KDAttentionSW ## added
from .kd_two_stage_attention import KDTwoStageATTDetector ##added
from .kd_two_stage_att_sw import KDTwoStageATTDetectorSW ##added
from .mkd_att_sw import MKDAttentionSW ## added
from .mkd_att_sw_test import MKDAttentionSW_test ## added
from .mkd_one_stage import MKnowledgeDistillationSingleStageDetector
from .amkd_one_stage import AMKnowledgeDistillationSingleStageDetector
from .att_kd_one_stage import ATTKnowledgeDistillationSingleStageDetector
from .att_kd_one_stage_aid import ATTAIDKDSingleStageDetector
from .pad_one_stage import SingleStagePAD
from .pad_two_stage import PADTwoStage
from .m3kd_one_stage import M3KnowledgeDistillationSingleStageDetector
from .aid_ld_one_stage import AIDLD
from .Cot_kd import CotKnowledgeDistillationSingleStageDetector
from .aid_lad import AID_LAD
from .colad import CoLAD
from .kd import KnowledgeDistiller
from .solad import SoLAD
from .kd_two_stage import KnowledgeDistillationTwoStageDetector
from .att_lad import ATTLAD
from .kd_cam import CAMKnowledgeDistillationSingleStageDetector
from .gfl_lda import GFL_LDA

__all__ = [
    'ATSS', 'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    'KnowledgeDistillationSingleStageDetector', 'FastRCNN', 'FasterRCNN',
    'MaskRCNN', 'CascadeRCNN', 'HybridTaskCascade', 'RetinaNet', 'FCOS',
    'GridRCNN', 'MaskScoringRCNN', 'RepPointsDetector', 'FOVEA', 'FSAF',
    'NASFCOS', 'PointRend', 'GFL', 'CornerNet', 'PAA', 'YOLOV3', 'YOLACT',
    'VFNet', 'DETR', 'TridentFasterRCNN', 'SparseRCNN', 'SCNet', 'SOLO',
    'DeformableDETR', 'AutoAssign', 'YOLOF', 'CenterNet', 'YOLOX',
    'TwoStagePanopticSegmentor', 'PanopticFPN', 'QueryInst', 'LAD', 'TOOD',
    'KDAttention', 'KDAttentionSW', 'KDTwoStageATTDetector', 'KDTwoStageATTDetectorSW',
    'MKDAttentionSW', 'MKDAttentionSW_test', 'MKnowledgeDistillationSingleStageDetector',
    "ATTKnowledgeDistillationSingleStageDetector", "ATTAIDKDSingleStageDetector",
    "AMKnowledgeDistillationSingleStageDetector", "SingleStagePAD", "PADTwoStage",
    "M3KnowledgeDistillationSingleStageDetector", "AIDLD", "CotKnowledgeDistillationSingleStageDetector",
    "AID_LAD", "CoLAD", 'KnowledgeDistiller', 'SoLAD', 'KnowledgeDistillationTwoStageDetector', 'ATTLAD',
    'CAMKnowledgeDistillationSingleStageDetector', 'GFL_LDA'
]
