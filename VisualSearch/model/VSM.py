from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from VisualSearch.model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
													 LlavaLlamaModel)

from .segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer

from .owlvit.owlvit import OwlViT

def dice_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
	num_masks: float,
	scale=1000,  # 100000.0,
	eps=1e-6,
):
	"""
	Compute the DICE loss, similar to generalized IOU for masks
	Args:
		inputs: A float tensor of arbitrary shape.
				The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs
				(0 for the negative class and 1 for the positive class).
	"""
	inputs = inputs.sigmoid()
	inputs = inputs.flatten(1, 2)
	targets = targets.flatten(1, 2)
	numerator = 2 * (inputs / scale * targets).sum(-1)
	denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
	loss = 1 - (numerator + eps) / (denominator + eps)
	loss = loss / (num_masks + 1e-8)
	return loss

def sigmoid_ce_loss(
	inputs: torch.Tensor,
	targets: torch.Tensor,
	num_masks: float,
):
	"""
	Args:
		inputs: A float tensor of arbitrary shape.
				The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs
				(0 for the negative class and 1 for the positive class).
	Returns:
		Loss tensor
	"""
	loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
	loss = loss.flatten(1, 2).mean(1) / (num_masks + 1e-8)
	return loss

class VSMMetaModel:
	def __init__(
		self,
		config,
		**kwargs,
	):
		super(VSMMetaModel, self).__init__(config)

		self.config = config
		if not hasattr(self.config, "train_mask_decoder"):
			self.config.train_mask_decoder = kwargs["train_mask_decoder"]
			self.config.out_dim = kwargs["out_dim"]
		else:
			is_eval = kwargs.get('is_eval', False)			
			self.initialize_lisa_modules(self.config, is_eval)

	def initialize_lisa_modules(self, config, is_eval=False):
		# OWL-ViT
		self.owlvit = OwlViT(1, is_eval)
		self.owlvit.train()
		for param in self.owlvit.parameters():
			param.requires_grad = True

		for param in self.owlvit.vision_model.parameters():
			param.requires_grad = False
		self.owlvit.vision_model.eval()

		for param in self.owlvit.box_head.parameters():
			param.requires_grad = False

		self.visual_projection = nn.Linear(self.owlvit.vision_model.config.hidden_size, 256, bias=False)
		for param in self.visual_projection.parameters():
			param.requires_grad = True

		self.prompt_encoder=PromptEncoder(
			embed_dim=256,
			image_embedding_size=(48, 48),
			input_image_size=(768, 768),
			mask_in_chans=16,
		)
		self.prompt_encoder.train()
		for param in self.prompt_encoder.parameters():
			param.requires_grad = True
		self.mask_decoder=MaskDecoder(
			num_multimask_outputs=3,
			transformer=TwoWayTransformer(
				depth=2,
				embedding_dim=256,
				mlp_dim=2048,
				num_heads=8,
			),
			transformer_dim=256,
			iou_head_depth=3,
			iou_head_hidden_dim=256,
		)
		self.mask_decoder.train()
		for param in self.mask_decoder.parameters():
			param.requires_grad = True

		# Projection layer
		in_dim = config.hidden_size
		out_dim = config.out_dim
		text_fc_det = [
			nn.Linear(in_dim, in_dim),
			nn.ReLU(inplace=True),
			nn.Linear(in_dim, out_dim),
			nn.Dropout(0.0),
		]
		self.text_hidden_fcs_det = nn.ModuleList([nn.Sequential(*text_fc_det)])
		self.text_hidden_fcs_det.train()
		for param in self.text_hidden_fcs_det.parameters():
			param.requires_grad = True

		text_fc_seg = [
			nn.Linear(in_dim, in_dim),
			nn.ReLU(inplace=True),
			nn.Linear(in_dim, 256),
			nn.Dropout(0.0),
		]
		self.text_hidden_fcs_seg = nn.ModuleList([nn.Sequential(*text_fc_seg)])
		self.text_hidden_fcs_seg.train()
		for param in self.text_hidden_fcs_seg.parameters():
			param.requires_grad = True


class VSMModel(VSMMetaModel, LlavaLlamaModel):
	def __init__(
		self,
		config,
		**kwargs,
	):
		super(VSMModel, self).__init__(config, **kwargs)

		self.config.use_cache = False
		self.config.vision_tower = self.config.mm_vision_tower
		self.config.mm_vision_select_feature = "patch"
		self.config.image_aspect_ratio = "square"
		self.config.image_grid_pinpoints = None
		self.config.tune_mm_mlp_adapter = False
		self.config.freeze_mm_mlp_adapter = True
		self.config.pretrain_mm_mlp_adapter = None
		self.config.mm_use_im_patch_token = False


class VSMForCausalLM(LlavaLlamaForCausalLM):
	def __init__(
		self,
		config,
		**kwargs,
	):
		if not hasattr(config, "train_mask_decoder"):
			config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
			config.mm_vision_tower = kwargs.get(
				"vision_tower", "openai/clip-vit-large-patch14"
			)
			self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
			self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
			self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
			self.det_loss_weight = kwargs.pop("det_loss_weight", None)
		else:
			config.mm_vision_tower = config.vision_tower

		self.loc_token_idx = kwargs.pop("loc_token_idx")

		super().__init__(config)

		self.model = VSMModel(config, **kwargs)

		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def get_visual_embs(self, pixel_values: torch.FloatTensor):
		with torch.no_grad():
			image_embeddings = self.model.owlvit.get_visual_embs(pixel_values)
		return image_embeddings

	def forward(self, **kwargs):
		if "past_key_values" in kwargs:
			return super().forward(**kwargs)
		return self.model_forward(**kwargs)

	def model_forward(
		self,
		images: torch.FloatTensor,
		images_clip: torch.FloatTensor,
		input_ids: torch.LongTensor,
		labels: torch.LongTensor,
		attention_masks: torch.LongTensor,
		offset: torch.LongTensor,
		masks_list: List[torch.FloatTensor],
		label_list: List[torch.Tensor],
		bboxes_labels_list: List[torch.FloatTensor],
		bboxes_valid_list: torch.Tensor,
		masks_valid_list: List[torch.Tensor],
		resize_list: List[tuple],
		inference: bool = False,
		**kwargs,
	):
		image_embeddings = self.get_visual_embs(images)
		batch_size = image_embeddings.shape[0]
		assert batch_size == len(offset) - 1

		loc_token_mask = input_ids[:, 1:] == self.loc_token_idx
		loc_token_mask = torch.cat(
			[
				loc_token_mask,
				torch.zeros((loc_token_mask.shape[0], 1)).bool().cuda(),
			],
			dim=1,
		)
		# hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
		loc_token_mask = torch.cat(
			[torch.zeros((loc_token_mask.shape[0], 255)).bool().cuda(), loc_token_mask],
			dim=1,
		)

		if inference:
			n_batch = 1
			length = input_ids.shape[0]
			assert images_clip.shape[0] == 1
			images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

			output_hidden_states = []
			for i in range(n_batch):
				start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
				output_i = super().forward(
					images=images_clip_extend[: end_i - start_i],
					attention_mask=attention_masks[start_i:end_i],
					input_ids=input_ids[start_i:end_i],
					output_hidden_states=True,
				)
				output_hidden_states.append(output_i.hidden_states)
				torch.cuda.empty_cache()

			output_hidden_states_list = []
			output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
			output_hidden_states_list.append(output_hidden_states_level)
			output_hidden_states = output_hidden_states_list
			output = None

		else:
			images_clip_list = []
			for i in range(len(offset) - 1):
				start_i, end_i = offset[i], offset[i + 1]
				images_clip_i = (
					images_clip[i]
					.unsqueeze(0)
					.expand(end_i - start_i, -1, -1, -1)
					.contiguous()
				)
				images_clip_list.append(images_clip_i)
			images_clip = torch.cat(images_clip_list, dim=0)

			output = super().forward(
				images=images_clip,
				attention_mask=attention_masks,
				input_ids=input_ids,
				labels=labels,
				output_hidden_states=True,
			)
			output_hidden_states = output.hidden_states

		# seg
		hidden_states_seg = []
		assert len(self.model.text_hidden_fcs_seg) == 1
		hidden_states_seg.append(self.model.text_hidden_fcs_seg[0](output_hidden_states[-1]))

		last_hidden_state_seg = torch.stack(hidden_states_seg, dim=-1).sum(dim=-1)

		# det
		hidden_states_det = []

		assert len(self.model.text_hidden_fcs_det) == 1
		hidden_states_det.append(self.model.text_hidden_fcs_det[0](output_hidden_states[-1]))
		last_hidden_state_det = torch.stack(hidden_states_det, dim=-1).sum(dim=-1)

		pred_embeddings_seg = last_hidden_state_seg[loc_token_mask]
		pred_embeddings_det = last_hidden_state_det[loc_token_mask]
		loc_token_counts = loc_token_mask.int().sum(-1)  # [bs, ]

		loc_token_offset = loc_token_counts.cumsum(-1)
		loc_token_offset = torch.cat(
			[torch.zeros(1).long().cuda(), loc_token_offset], dim=0
		)

		loc_token_offset = loc_token_offset[offset]

		pred_embeddings_seg_ = []
		for i in range(len(loc_token_offset) - 1):
			start_i, end_i = loc_token_offset[i], loc_token_offset[i + 1]
			pred_embeddings_seg_.append(pred_embeddings_seg[start_i:end_i])
		pred_embeddings_seg = pred_embeddings_seg_

		pred_embeddings_det_ = []
		for i in range(len(loc_token_offset) - 1):
			start_i, end_i = loc_token_offset[i], loc_token_offset[i + 1]
			pred_embeddings_det_.append(pred_embeddings_det[start_i:end_i])
		pred_embeddings_det = pred_embeddings_det_

		# seg branch 
		multimask_output = False
		pred_masks = []
		for i in range(len(pred_embeddings_seg)):
			(
				sparse_embeddings,
				dense_embeddings,
			) = self.model.prompt_encoder(
				points=None,
				boxes=None,
				masks=None,
				text_embeds=pred_embeddings_seg[i].unsqueeze(1),
			)
			sparse_embeddings = sparse_embeddings.to(pred_embeddings_seg[i].dtype)
			low_res_masks, iou_predictions = self.model.mask_decoder(
				image_embeddings=self.model.visual_projection(image_embeddings[i].unsqueeze(0)).permute(0, 3, 1, 2),
				image_pe=self.model.prompt_encoder.get_dense_pe(),
				sparse_prompt_embeddings=sparse_embeddings,
				dense_prompt_embeddings=dense_embeddings,
				multimask_output=multimask_output,
			)
			pred_mask = F.interpolate(
			low_res_masks, label_list[i].shape, mode="bilinear", align_corners=False
		)
			pred_masks.append(pred_mask[:, 0])

		gt_masks = masks_list

		# det branch
		detection_result_batch = []
		for i in range(len(pred_embeddings_det)):
			bs = pred_embeddings_det[i].shape[0]
			detection_result = self.model.owlvit(image_embeddings[i].unsqueeze(0).repeat(bs, 1, 1, 1), pred_embeddings_det[i].unsqueeze(1))
			detection_result_batch.append(detection_result)


		pred_logits = torch.cat([detection_result['pred_logits'] for detection_result in detection_result_batch], 0)
		pred_boxes = torch.cat([detection_result['pred_boxes'] for detection_result in detection_result_batch], 0)
		if inference:
			return {
				"pred_masks": pred_masks,
				"gt_masks": gt_masks,
				"pred_logits": pred_logits,
				"pred_boxes": pred_boxes,
				"gt_bboxes": bboxes_labels_list
			}
		
		num_boxes = 0
		for bboxes_labels, bboxes_valid in zip(bboxes_labels_list, bboxes_valid_list):
			if bboxes_valid:
				num_boxes += len(bboxes_labels)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=image_embeddings.device)
		num_boxes = torch.clamp(num_boxes, min=1).item()
		
		detection_result_batch = {'pred_logits':pred_logits, 'pred_boxes':pred_boxes}

		target_det = []
		all_bboxes_valid = []
		for bboxes_label, bboxes_valid in zip(bboxes_labels_list, bboxes_valid_list):
			target_det.append({"labels":torch.zeros(len(bboxes_label)).to(bboxes_label.device, torch.long), "boxes":bboxes_label})
			if bboxes_valid:
				all_bboxes_valid.append(torch.ones((min(24*24, len(bboxes_label)), 1)).to(bboxes_label.device, torch.long))
			else:
				all_bboxes_valid.append(torch.zeros((min(24*24, len(bboxes_label)), 1)).to(bboxes_label.device, torch.long))
		all_bboxes_valid = torch.cat(all_bboxes_valid, 0)
		
		loss_dict = self.model.owlvit.criterion(detection_result_batch, target_det, num_boxes)

		for loss_k, loss_v in loss_dict.items():
			if "loss_ce" in loss_k:
				loss_dict[loss_k] = (loss_v*bboxes_valid_list.unsqueeze(-1)).mean()
			else:
				loss_dict[loss_k] = (loss_v*all_bboxes_valid).sum()

		weight_dict = self.model.owlvit.criterion.weight_dict
		detection_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
		detection_loss = detection_loss*self.det_loss_weight

		model_output = output
		output = model_output.logits

		ce_loss = model_output.loss
		ce_loss = ce_loss * self.ce_loss_weight
		mask_bce_loss = 0
		mask_dice_loss = 0
		num_masks = 0
		for batch_idx in range(len(pred_masks)):
			gt_mask = gt_masks[batch_idx]
			pred_mask = pred_masks[batch_idx]
			masks_valid = masks_valid_list[batch_idx]

			mask_bce_loss += (
				sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
				* gt_mask.shape[0] * masks_valid
			).sum()
			mask_dice_loss += (
				dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
				* gt_mask.shape[0] * masks_valid
			).sum()
			num_masks += masks_valid.sum()

		mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
		mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
		mask_loss = mask_bce_loss + mask_dice_loss

		loss = ce_loss + mask_loss + detection_loss

		return {
			"loss": loss,
			"ce_loss": ce_loss,
			"mask_bce_loss": mask_bce_loss,
			"mask_dice_loss": mask_dice_loss,
			"mask_loss": mask_loss,
			"detection_loss": detection_loss,
			"detection_loss_ce": loss_dict['loss_ce'],
			"detection_loss_bbox": loss_dict['loss_bbox'],
			"detection_loss_giou": loss_dict['loss_giou'],
		}

	def inference(
		self,
		images_clip,
		images,
		input_ids,
		resize_list,
		original_size_list,
		max_new_tokens=32,
		tokenizer=None,
		mode = 'vqa'
	):
		assert mode in ['vqa', 'segmentation', 'detection']
		with torch.no_grad():
			outputs = self.generate(
				images=images_clip,
				input_ids=input_ids,
				max_new_tokens=max_new_tokens,
				num_beams=1,
				output_hidden_states=True,
				return_dict_in_generate=True,
			)
			output_hidden_states = outputs.hidden_states[-1]
			output_ids = outputs.sequences

			if mode == 'vqa':
				return output_ids, None, None

			loc_token_mask = output_ids[:, 1:] == self.loc_token_idx
			# hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
			loc_token_mask = torch.cat(
				[
					torch.zeros((loc_token_mask.shape[0], 255)).bool().cuda(),
					loc_token_mask,
				],
				dim=1,
			)

			# seg
			hidden_states_seg = []
			assert len(self.model.text_hidden_fcs_seg) == 1
			hidden_states_seg.append(self.model.text_hidden_fcs_seg[0](output_hidden_states))

			last_hidden_state_seg = torch.stack(hidden_states_seg, dim=-1).sum(dim=-1)

			# det
			hidden_states_det = []

			assert len(self.model.text_hidden_fcs_det) == 1
			hidden_states_det.append(self.model.text_hidden_fcs_det[0](output_hidden_states))
			last_hidden_state_det = torch.stack(hidden_states_det, dim=-1).sum(dim=-1)

			pred_embeddings_seg = last_hidden_state_seg[loc_token_mask]
			pred_embeddings_det = last_hidden_state_det[loc_token_mask]
			loc_token_counts = loc_token_mask.int().sum(-1)  # [bs, ]

			loc_token_offset = loc_token_counts.cumsum(-1)
			loc_token_offset = torch.cat(
				[torch.zeros(1).long().cuda(), loc_token_offset], dim=0
			)


			pred_embeddings_seg_ = []
			for i in range(len(loc_token_offset) - 1):
				start_i, end_i = loc_token_offset[i], loc_token_offset[i + 1]
				pred_embeddings_seg_.append(pred_embeddings_seg[start_i:end_i])
			pred_embeddings_seg = pred_embeddings_seg_

			pred_embeddings_det_ = []
			for i in range(len(loc_token_offset) - 1):
				start_i, end_i = loc_token_offset[i], loc_token_offset[i + 1]
				pred_embeddings_det_.append(pred_embeddings_det[start_i:end_i])
			pred_embeddings_det = pred_embeddings_det_

			image_embeddings = self.get_visual_embs(images)

			multimask_output = False
			pred_masks = []
			for i in range(len(pred_embeddings_seg)):
				(
					sparse_embeddings,
					dense_embeddings,
				) = self.model.prompt_encoder(
					points=None,
					boxes=None,
					masks=None,
					text_embeds=pred_embeddings_seg[i].unsqueeze(1),
				)

				sparse_embeddings = sparse_embeddings.to(pred_embeddings_seg[i].dtype)
				low_res_masks, iou_predictions = self.model.mask_decoder(
					image_embeddings=self.model.visual_projection(image_embeddings[i].unsqueeze(0)).permute(0, 3, 1, 2),
					image_pe=self.model.prompt_encoder.get_dense_pe(),
					sparse_prompt_embeddings=sparse_embeddings,
					dense_prompt_embeddings=dense_embeddings,
					multimask_output=multimask_output,
				)
				pred_mask = F.interpolate(
				low_res_masks.float(), original_size_list[i], mode="bilinear", align_corners=False
			)
				pred_masks.append(pred_mask[:, 0])

			if mode == 'segmentation':
				return None, pred_masks, None

			# detection model
			detection_result_batch = []
			for i in range(len(pred_embeddings_det)):
				bs = pred_embeddings_det[i].shape[0]
				detection_result = self.model.owlvit(image_embeddings[i].unsqueeze(0).repeat(bs, 1, 1, 1), pred_embeddings_det[i].unsqueeze(1))
				detection_result_batch.append(detection_result)


			pred_logits = torch.cat([detection_result['pred_logits'] for detection_result in detection_result_batch], 0)
			pred_boxes = torch.cat([detection_result['pred_boxes'] for detection_result in detection_result_batch], 0)
			detection_result_batch = {'pred_logits':pred_logits, 'pred_boxes':pred_boxes}

		return None, pred_masks, detection_result_batch