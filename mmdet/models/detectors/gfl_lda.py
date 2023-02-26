# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.utils import get_root_logger
from functools import partial


logger = get_root_logger()

def lda(X, y, n_classes, lamb):
    # flatten X
    X = X.view(X.shape[0], -1)
    N, D = X.shape

    # count unique labels in y
    labels, counts = torch.unique(y, return_counts=True)
    assert len(labels) == n_classes  # require X,y cover all classes

    # compute mean-centered observations and covariance matrix
    X_bar = X - torch.mean(X, 0)
    Xc_mean = torch.zeros((n_classes, D), dtype=X.dtype, device=X.device, requires_grad=False)
    St = X_bar.t().matmul(X_bar) / (N - 1)  # total scatter matrix
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device, requires_grad=True)  # within-class scatter matrix
    for c, Nc in zip(labels, counts):
        Xc = X[y == c]
        Xc_mean[int(c), :] = torch.mean(Xc, 0)
        Xc_bar = Xc - Xc_mean[int(c), :]
        Sw = Sw + Xc_bar.t().matmul(Xc_bar) / (Nc - 1)
    Sw /= n_classes
    Sb = St - Sw  # between scatter matrix

    # cope for numerical instability
    Sw += torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * lamb
    # logger.info(f'Sw is {Sw.shape}, Sw have nan: {torch.isnan(Sw).any()}')
    # logger.info(f'Sb is {Sb.shape}, Sw have nan: {torch.isnan(Sb).any()}')
    # compute eigen decomposition
    # temp = Sw.pinverse().matmul(Sb)
    # temp = torch.linalg.pinv(Sw, hermitian = True).matmul(Sb)
    # compute eigen decomposition
    temp = Sw.pinverse().matmul(Sb)
    # evals, evecs = torch.symeig(temp, eigenvectors=True) # only works for symmetric matrix
    evals, evecs = torch.eig(temp, eigenvectors=True) # shipped from nightly-built version (1.8.0.dev20201015)
    print(evals.shape, evecs.shape)

    # remove complex eigen values and sort
    noncomplex_idx = evals[:, 1] == 0
    evals = evals[:, 0][noncomplex_idx] # take real part of eigen values
    evecs = evecs[:, noncomplex_idx]
    evals, inc_idx = torch.sort(evals) # sort by eigen values, in ascending order
    evecs = evecs[:, inc_idx]
    print(evals.shape, evecs.shape)

    # flag to indicate if to skip backpropagation
    hasComplexEVal = evecs.shape[1] < evecs.shape[0]

    return hasComplexEVal, Xc_mean, evals, evecs


def lda_loss(evals, n_classes, n_eig=None, margin=None):
    n_components = n_classes - 1
    evals = evals[-n_components:]
    # evecs = evecs[:, -n_components:]
    print('evals', evals.shape, evals)
    # print('evecs', evecs.shape)

    # calculate loss
    if margin is not None:
        threshold = torch.min(evals) + margin
        n_eig = torch.sum(evals < threshold)
    loss = -torch.mean(evals[:n_eig]) # small eigen values are on left
    return loss

class LDA(nn.Module):
    def __init__(self, n_classes, lamb):
        super(LDA, self).__init__()
        self.n_classes = n_classes
        self.n_components = n_classes - 1
        self.lamb = lamb
        self.lda_layer = partial(lda, n_classes=n_classes, lamb=lamb)

    def forward(self, X, y):
        # perform LDA
        hasComplexEVal, Xc_mean, evals, evecs = self.lda_layer(X, y)  # CxD, D, DxD

        # compute LDA statistics
        self.scalings_ = evecs  # projection matrix, DxD
        self.coef_ = Xc_mean.matmul(evecs).matmul(evecs.t())  # CxD
        self.intercept_ = -0.5 * torch.diagonal(Xc_mean.matmul(self.coef_.t())) # C

        # return self.transform(X)
        return hasComplexEVal, evals

    def transform(self, X):
        """ transform data """
        X_new = X.matmul(self.scalings_)
        return X_new[:, :self.n_components]

    def predict(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return torch.argmax(logit, dim=1)

    def predict_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        proba = nn.functional.softmax(logit, dim=1)
        return proba

    def predict_log_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        log_proba = nn.functional.log_softmax(logit, dim=1)
        return log_proba

@DETECTORS.register_module()
class GFL_LDA(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFL_LDA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)

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
        # super(GFL_LDA, self).forward_train(img, img_metas)
        # print_log('this is the theteteetetewt log printed')
        

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)



        margin = 0.1
        lamb = 0.5
        loss_lda = 0
        y = torch.cat(gt_labels)
        n_classes = len(torch.unique(y))
        for _i in range(len(x)):
            N, C, H, W= x[_i].shape
            # X = torch.ones_like(n_classes, C)
            instances = []
            wmin,wmax,hmin,hmax = [],[],[],[]
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
                    #channel-wise get feature values, get maximum value for each bbox.
                    # (C, 1, 1)
                    value_per_channel = torch.amax(x[_i][i][:, hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], dim = (1,2))
                    # logger.info(f'value_per_channel shape is: {value_per_channel.shape}')
                    instances.append(value_per_channel)
            X_lda = torch.stack(instances)
            # logger.info(f'X_LAD shape is {X_lda.shape}')
            # logger.info(f'gt_labels are {gt_labels}')
            lda = LDA(n_classes, lamb = lamb)
            _, evals = lda(X_lda, y)
            loss_lda += lda_loss(evals, n_classes, n_eig = None, margin = margin)
            
        losses.update(LDA_loss = loss_lda)
            # lda_loss = self.lda_loss(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)


        return losses
