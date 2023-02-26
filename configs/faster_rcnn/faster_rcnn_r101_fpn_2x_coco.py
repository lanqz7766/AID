_base_ = './faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)