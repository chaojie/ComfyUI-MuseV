# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference, fast_rcnn_inference_single_image, FastRCNNOutputLayers

from detectron2.layers.nms import batched_nms
from .box_regression import BUABoxes

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""

class BUACaffeFastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, image_scales, attr_on=False
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.image_scales = image_scales
        self.attr_on = attr_on

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def fast_rcnn_inference(self, boxes, scores, image_shapes, image_scales, score_thresh, nms_thresh, topk_per_image):
        """
        Call `fast_rcnn_inference_single_image` for all images.

        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
            image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, image_scale, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, image_shape, image_scale in zip(scores, boxes, image_shapes, image_scales)
        ]
        return tuple(list(x) for x in zip(*result_per_image))

    def fast_rcnn_inference_single_image(
        self, boxes, scores, image_shape, image_scale, score_thresh, nms_thresh, topk_per_image
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.

        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """
        scores = scores[:, 1:]
        boxes = boxes[:, 4:]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = BUABoxes(boxes.reshape(-1, 4))
        boxes.clip((image_shape[0]/image_scale, image_shape[1]/image_scale))
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # Filter results based on detection scores
        filter_mask = scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        # Apply per-class NMS
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

        result = Instances(image_shape)
        result.pred_boxes = BUABoxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        return result, filter_inds[:, 0]

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        # Always use 1 image per worker during inference since this is the
        # standard when reporting inference time in papers.
        self.proposals.scale(1.0/self.image_scales[0], 1.0/self.image_scales[0])
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas,
            self.proposals.tensor,
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        image_scales = self.image_scales

        return self.fast_rcnn_inference(
            boxes, scores, image_shapes, image_scales, score_thresh, nms_thresh, topk_per_image
        )

class BUACaffeFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, attr_on=False, num_attr_classes=401):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(BUACaffeFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.attr_on = attr_on

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.attr_on:
            self.cls_embed = nn.Embedding(num_classes, 256)
            self.attr_linear1 = nn.Linear(input_size + 256, 512)
            self.attr_linear2 = nn.Linear(512, num_attr_classes)

            nn.init.normal_(self.cls_embed.weight, std=0.01)
            nn.init.normal_(self.attr_linear1.weight, std=0.01)
            nn.init.normal_(self.attr_linear2.weight, std=0.01)
            nn.init.constant_(self.attr_linear1.bias, 0)
            nn.init.constant_(self.attr_linear2.bias, 0)

    def forward(self, x, proposal_boxes=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        if self.attr_on:
            
            # get labels and indices of proposals with foreground
            all_labels = torch.argmax(scores, dim=1)

            # get embeddings of indices using gt cls labels
            cls_embed_out = self.cls_embed(all_labels)

            # concat with fc7 feats
            concat_attr = cat([x, cls_embed_out], dim=1)

            # pass through attr head layers
            fc_attr = self.attr_linear1(concat_attr)
            attr_score =  F.softmax(self.attr_linear2(F.relu(fc_attr)), dim=-1)
            return scores, proposal_deltas, attr_score

        return scores, proposal_deltas

class BUADetection2FastRCNNOutputs(FastRCNNOutputLayers):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
            
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, attr_on=False, pred_attribute_logits=None, num_attr_classes=400, gt_attributes=None
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            pred_attribute_logits (Tensor:) A tensor of shape (R, C) storing the predicted attribute
                logits for all R predicted object instances.
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.attr_on = attr_on
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas

        if self.attr_on:
            self.pred_attribute_logits = pred_attribute_logits
            self.gt_attributes = gt_attributes
        self.smooth_l1_beta = smooth_l1_beta

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]
        self.num_attr_classes = num_attr_classes

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def attribute_loss(self):
        fg_gt_attributes = self.gt_attributes
        n_boxes = self.pred_attribute_logits.shape[0]
        self.pred_attribute_logits = self.pred_attribute_logits.unsqueeze(1)
        self.pred_attribute_logits = self.pred_attribute_logits.expand(n_boxes, 16, self.num_attr_classes).contiguous().view(-1, self.num_attr_classes)

        inv_per_box_weights = (
            (fg_gt_attributes >= 0).sum(dim=1).repeat(16, 1).transpose(0, 1).flatten()
        )
        per_box_weights = inv_per_box_weights.float().reciprocal()
        per_box_weights[per_box_weights > 1] = 0.0

        fg_gt_attributes = fg_gt_attributes.view(-1)
        attributes_loss = 0.5 * F.cross_entropy(
            self.pred_attribute_logits, fg_gt_attributes, reduction="none", ignore_index=-1
        )

        attributes_loss = (attributes_loss * per_box_weights).view(n_boxes, -1).sum(dim=1)

        n_valid_boxes = len(attributes_loss.nonzero())

        if n_valid_boxes > 0:
            attributes_loss = (attributes_loss / n_valid_boxes).sum()
        else:
            attributes_loss = (attributes_loss * 0.0).sum()
        return attributes_loss

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
            "loss_attr": self.attribute_loss() if self.attr_on else 0.,
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

class BUADetectron2FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4, attr_on=False, num_attr_classes=400):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(BUADetectron2FastRCNNOutputLayers, self).__init__()
        self.attr_on = attr_on
        self.num_classes = num_classes
        self.num_attr_classes = num_attr_classes

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.attr_on:
            self.cls_embed = nn.Embedding(num_classes+1, 256)
            self.attr_linear1 = nn.Linear(input_size + 256, 512)
            self.attr_linear2 = nn.Linear(512, num_attr_classes)

            # nn.init.normal_(self.cls_embed.weight, std=0.01)
            nn.init.normal_(self.attr_linear1.weight, std=0.01)
            nn.init.normal_(self.attr_linear2.weight, std=0.01)
            nn.init.constant_(self.attr_linear1.bias, 0)
            nn.init.constant_(self.attr_linear2.bias, 0)

    def forward(self, x, proposal_boxes=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        if self.attr_on:
            if self.training:
                assert proposal_boxes is not None, "Proposals are None while attr=True"
                proposals, fg_selection_atrributes = select_foreground_proposals(proposal_boxes, self.num_classes)
                attribute_features = x[torch.cat(fg_selection_atrributes, dim=0)]
                cls_labels = torch.cat([prop.gt_classes for prop in proposals])
                
            else:
                # get labels and indices of proposals with foreground
                cls_labels = torch.argmax(scores, dim=1)
                attribute_features = x

            # get embeddings of indices using gt cls labels
            cls_embed_out = self.cls_embed(cls_labels)

            # concat with fc7 feats
            concat_attr = cat([attribute_features, cls_embed_out], dim=1)

            # pass through attr head layers
            fc_attr = self.attr_linear1(concat_attr)
            attr_score = self.attr_linear2(F.relu(fc_attr))
            return scores, proposal_deltas, attr_score, cat([p.gt_attributes for p in proposals], dim=0) if self.training else None

        return scores, proposal_deltas
