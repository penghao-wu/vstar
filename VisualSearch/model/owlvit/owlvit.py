import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
from typing import Any, Dict, Optional, Tuple, Union

from transformers import OwlViTForObjectDetection, OwlViTConfig

from .util import box_ops
from .util.misc import (nested_tensor_from_tensor_list,
                       accuracy, interpolate, inverse_sigmoid)

from .matcher import HungarianMatcher
from .segmentation import dice_loss, sigmoid_focal_loss

from .matcher import HungarianMatcher
import copy

class OwlViT(nn.Module):
    def __init__(self, num_classes, is_eval=False):
        super().__init__()
        if is_eval:
            owlViT_config = OwlViTConfig.from_pretrained("google/owlvit-base-patch16")
            model_owlViT = OwlViTForObjectDetection(owlViT_config)
        else:
            model_owlViT = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
        self.vision_model = model_owlViT.owlvit.vision_model
        self.class_head = model_owlViT.class_head
        self.box_head = model_owlViT.box_head
        self.layer_norm = model_owlViT.layer_norm
        self.sigmoid = nn.Sigmoid()
        del model_owlViT

        self.matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
        self.weight_dict = {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 2}

        self.losses = ['labels', 'boxes']
        # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
        self.criterion = SetCriterion(num_classes, self.matcher, self.weight_dict, self.losses, focal_alpha=0.25)

    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # Computes normalized xy corner coordinates from feature_map.
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        device = feature_map.device
        num_patches = feature_map.shape[1]

        box_coordinates = np.stack(
            np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1
        ).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

    def get_visual_embs(self, image):
        vision_outputs = self.vision_model(
        pixel_values=image,
            output_hidden_states=self.vision_model.config.output_hidden_states,
            return_dict=self.vision_model.config.use_return_dict,
        )

        # Get image embeddings
        last_hidden_state = vision_outputs[0]
        image_embeds = self.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        feature_map = image_embeds.reshape(new_size)
        return feature_map

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        ):

        feature_map = image_embeddings

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        query_embeds = prompt_embeddings.reshape(batch_size, 1, prompt_embeddings.shape[-1])

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)

        out = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
        return out


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduce=None)
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, num_boxes):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses