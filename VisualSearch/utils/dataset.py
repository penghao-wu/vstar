import glob
import os
import random
from PIL import Image
import cv2
cv2.setNumThreads(1)
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from transformers import OwlViTProcessor

from VisualSearch.model.llava import conversation as conversation_lib
from VisualSearch.model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
								   IMAGE_TOKEN_INDEX)
from VisualSearch.model.llava.mm_utils import tokenizer_image_token

from VisualSearch.utils.data_processing import get_mask_from_json
from VisualSearch.utils.refer import REFER
from VisualSearch.utils.refer_seg_dataset import ReferSegDataset
from VisualSearch.utils.general_segdet_dataset import SegDetDataset
from VisualSearch.utils.mixed_grounding_dataset import MixedGroundingDataset
from VisualSearch.utils.vqa_dataset import VQADataset
from VisualSearch.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
					DEFAULT_IMAGE_TOKEN)
from VisualSearch.utils.utils import box_xyxy_to_cxcywh, expand2square


def collate_fn(
	batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
	image_path_list = []
	images_list = []
	images_clip_list = []
	conversation_list = []
	masks_list = []
	label_list = []
	bboxes_labels_list = []
	bboxes_valid_list = []
	masks_valid_list = []
	resize_list = []
	questions_list = []
	sampled_classes_list = []
	offset_list = [0]
	cnt = 0
	inferences = []
	for (
		image_path,
		images,
		images_clip,
		conversations,
		masks,
		label,
		bboxes_labels,
		bboxes_valid,
		masks_valid,
		resize,
		questions,
		sampled_classes,
		inference,
	) in batch:
		image_path_list.append(image_path)
		images_list.append(images)
		images_clip_list.append(images_clip)
		conversation_list.extend(conversations)
		label_list.append(label)
		masks_list.append(masks.float())
		bboxes_labels_list.extend(bboxes_labels)
		bboxes_valid_list.extend(bboxes_valid)
		masks_valid_list.append(torch.tensor(masks_valid))
		resize_list.append(resize)
		questions_list.append(questions)
		sampled_classes_list.append(sampled_classes)
		cnt += len(conversations)
		offset_list.append(cnt)
		inferences.append(inference)

	if use_mm_start_end:
		# replace <image> token
		for i in range(len(conversation_list)):
			replace_token = DEFAULT_IMAGE_TOKEN
			replace_token = (
				DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
			)
			conversation_list[i] = conversation_list[i].replace(
				DEFAULT_IMAGE_TOKEN, replace_token
			)
	input_ids = [
		tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
		for prompt in conversation_list
	]
	input_ids = torch.nn.utils.rnn.pad_sequence(
		input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
	)
	attention_masks = input_ids.ne(tokenizer.pad_token_id)

	for i in range(len(bboxes_valid_list)):
		bboxes_valid = bboxes_valid_list[i]
		attention_mask = attention_masks[i]
		if not bboxes_valid:
			attention_mask = attention_mask & input_ids[i].ne(tokenizer("[LOC]", add_special_tokens=False).input_ids[0])
			attention_masks[i] = attention_mask

	conv = conversation_lib.default_conversation.copy()
	targets = input_ids.clone()

	if conv_type == "llava_v1":
		sep = conv.sep + conv.roles[1] + ": "
	else:
		sep = "[/INST] "
	for conversation, target in zip(conversation_list, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep2)
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			# if len(parts) != 2:
			#     break
			assert len(parts) == 2, (len(parts), rou)
			parts[0] += sep

			if DEFAULT_IMAGE_TOKEN in conversation:
				round_len = len(tokenizer_image_token(rou, tokenizer))
				instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 2

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if False:
			z = target.clone()
			z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
			if local_rank == 0:
				print(
					"conversation: ",
					conversation,
					"tokenizer.decode(z): ",
					tokenizer.decode(z),
				)

		if cur_len < tokenizer.model_max_length:
			assert cur_len == total_len

	if inferences[0] == False:
		truncate_len = tokenizer.model_max_length - 255

		if input_ids.shape[1] > truncate_len:
			input_ids = input_ids[:, :truncate_len]
			targets = targets[:, :truncate_len]
			attention_masks = attention_masks[:, :truncate_len]

	return {
		"image_paths": image_path_list,
		"images": torch.stack(images_list, dim=0),
		"images_clip": torch.stack(images_clip_list, dim=0),
		"input_ids": input_ids,
		"labels": targets,
		"bboxes_labels_list": bboxes_labels_list,
		"bboxes_valid_list": torch.tensor(bboxes_valid_list),
		"masks_valid_list": masks_valid_list,
		"attention_masks": attention_masks,
		"masks_list": masks_list,
		"label_list": label_list,
		"resize_list": resize_list,
		"offset": torch.LongTensor(offset_list),
		"questions_list": questions_list,
		"sampled_classes_list": sampled_classes_list,
		"inference": inferences[0],
		"conversation_list": conversation_list,
	}


class HybridDataset(torch.utils.data.Dataset):
	pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
	pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
	img_size = 1024
	ignore_label = 255

	def __init__(
		self,
		base_dir,
		tokenizer,
		vision_tower,
		samples_per_epoch=500 * 8 * 2 * 10,
		precision: str = "fp32",
		num_classes_per_sample: int = 3,
		exclude_val=False,
		dataset="general_segdet||refer_seg||vqa||reason_seg",
		sample_rate=[9, 3, 3, 1],
		general_segdet_data="objects365||cocostuff||paco_lvis",
		general_segdet_sample_rate=[2,1,1],
		refer_seg_data="refclef||refcoco||refcoco+||refcocog",
		vqa_data="possible_locations_conv_86k||llava_instruct_80k",
		vqa_sample_rate=[2,1],
	):
		self.exclude_val = exclude_val
		self.dataset = dataset
		self.samples_per_epoch = samples_per_epoch
		self.num_classes_per_sample = num_classes_per_sample
		sample_rate = np.array(sample_rate)
		self.sample_rate = sample_rate / sample_rate.sum()

		self.base_dir = base_dir
		self.tokenizer = tokenizer
		self.precision = precision

		self.datasets = dataset.split("||")

		self.all_datasets = []
		for dataset in self.datasets:
			if dataset == "general_segdet":
				self.all_datasets.append(
					SegDetDataset(
						base_dir,
						tokenizer,
						vision_tower,
						samples_per_epoch,
						precision,
						num_classes_per_sample,
						exclude_val,
						general_segdet_data,
						general_segdet_sample_rate,
					)
				)
			elif dataset == "refer_seg":
				self.all_datasets.append(
					ReferSegDataset(
						base_dir,
						tokenizer,
						vision_tower,
						samples_per_epoch,
						precision,
						num_classes_per_sample,
						exclude_val,
						refer_seg_data,
					)
				)
			elif dataset == "vqa":
				self.all_datasets.append(
					VQADataset(
						base_dir,
						tokenizer,
						vision_tower,
						samples_per_epoch,
						precision,
						num_classes_per_sample,
						exclude_val,
						vqa_data,
						vqa_sample_rate,
					)
				)
			elif dataset == "mixed_grounding":
				self.all_datasets.append(
					MixedGroundingDataset(
						base_dir,
						tokenizer,
						vision_tower,
						samples_per_epoch,
						precision,
						num_classes_per_sample,
						exclude_val,
					)
				)

	def __len__(self):
		return self.samples_per_epoch

	def __getitem__(self, idx):
		ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
		data = self.all_datasets[ind]
		inference = False
		return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
	pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
	pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
	img_size = 1024
	ignore_label = 255

	def __init__(
		self,
		base_dir,
		tokenizer,
		vision_tower,
		val_dataset,
	):
		self.base_dir = base_dir
		splits = val_dataset.split("|")
		if len(splits) == 2:
			ds, split = splits
			images = glob.glob(
				os.path.join(self.base_dir, "reason_seg", ds, split, "*.jpg")
			)
			self.images = images
			self.data_type = "reason_seg"
		elif len(splits) == 3:
			self.base_dir = os.path.join(self.base_dir, 'refer_seg')
			ds, splitBy, split = splits
			refer_api = REFER(self.base_dir, ds, splitBy)
			ref_ids_val = refer_api.getRefIds(split=split)
			images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
			refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
			refer_seg_ds = {}
			refer_seg_ds["images"] = []
			loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
			for item in loaded_images:
				item = item.copy()
				if ds == "refclef":
					item["file_name"] = os.path.join(
						self.base_dir, "images/saiapr_tc-12", item["file_name"]
					)
				elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
					item["file_name"] = os.path.join(
						self.base_dir,
						"images/mscoco/images/train2014",
						item["file_name"],
					)
				refer_seg_ds["images"].append(item)
			refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

			img2refs = {}
			for ref in refs_val:
				image_id = ref["image_id"]
				img2refs[image_id] = img2refs.get(image_id, []) + [
					ref,
				]
			refer_seg_ds["img2refs"] = img2refs
			self.refer_seg_ds = refer_seg_ds
			self.data_type = "refer_seg"

		self.ds = ds
		self.tokenizer = tokenizer
		self.transform = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
		self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

	def __len__(self):
		if self.data_type == "refer_seg":
			return len(self.refer_seg_ds["images"])
		else:
			return len(self.images)

	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize pixel values and pad to a square input."""
		# Normalize colors
		x = (x - self.pixel_mean) / self.pixel_std

		# Pad
		h, w = x.shape[-2:]
		padh = self.img_size - h
		padw = self.img_size - w
		x = F.pad(x, (0, padw, 0, padh))
		return x

	def __getitem__(self, idx):
		if self.data_type == "refer_seg":
			refer_seg_ds = self.refer_seg_ds
			images = refer_seg_ds["images"]
			annotations = refer_seg_ds["annotations"]
			img2refs = refer_seg_ds["img2refs"]

			image_info = images[idx]
			image_path = image_info["file_name"]
			image_id = image_info["id"]

			refs = img2refs[image_id]
			if len(refs) == 0:
				raise ValueError("image {} has no refs".format(image_id))

			sents = []
			ann_ids = []
			for ref in refs:
				for sent in ref["sentences"]:
					sents.append(sent["sent"].strip().lower())
					ann_ids.append(ref["ann_id"])

			sampled_sents = sents
			sampled_ann_ids = ann_ids
			image = cv2.imread(image_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			is_sentence = False
		else:
			image_path = self.images[idx]
			image = cv2.imread(image_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			json_path = image_path.replace(".jpg", ".json")
			mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
			sampled_sents = [sampled_sents[0]]

		conversations = []
		conv = conversation_lib.default_conversation.copy()
		i = 0
		while i < len(sampled_sents):
			conv.messages = []
			text = sampled_sents[i].strip()
			if is_sentence:
				conv.append_message(
					conv.roles[0],
					DEFAULT_IMAGE_TOKEN
					+ "\n {} Please output segmentation mask.".format(text),
				)
				conv.append_message(conv.roles[1], "[LOC].")
			else:
				conv.append_message(
					conv.roles[0],
					DEFAULT_IMAGE_TOKEN
					+ "\n Please locate the {} in this image.".format(
						text
					),
				)
				conv.append_message(conv.roles[1], "Sure, [LOC].")
			conversations.append(conv.get_prompt())
			i += 1

		# preprocess image for clip
		image_clip = self.clip_image_processor.preprocess(
				expand2square(Image.open(image_path).convert('RGB'), tuple(int(x*255) for x in self.clip_image_processor.image_mean)), return_tensors="pt")["pixel_values"][0]
		original_size = image.shape[:2]

		image = self.transform(images=image, return_tensors="pt")['pixel_values'][0]
		resize = image.shape[:2]

		if self.data_type == "refer_seg":
			masks = []
			bboxes_labels = []
			for i, ann_id in enumerate(sampled_ann_ids):
				ann = annotations[ann_id]
				cur_bboxes = [ann['bbox']]
				cur_bboxes = torch.tensor(cur_bboxes).view(-1, 4)
				# xywh to x1y1x2y2
				cur_bboxes[:, 2:] += cur_bboxes[:, :2]
				cur_bboxes[:, 0::2].clamp_(min=0, max=original_size[1])
				cur_bboxes[:, 1::2].clamp_(min=0, max=original_size[0])
				keep = (cur_bboxes[:, 3] > cur_bboxes[:, 1]) & (cur_bboxes[:, 2] > cur_bboxes[:, 0])
				cur_bboxes = cur_bboxes[keep]
				cur_bboxes = box_xyxy_to_cxcywh(cur_bboxes)
				cur_bboxes = cur_bboxes / torch.tensor([original_size[1], original_size[0], original_size[1], original_size[0]], dtype=torch.float32)
				if len(cur_bboxes) == 0:
					return self.__getitem__(0)
				bboxes_labels.append(cur_bboxes)
				if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
					m = np.zeros((image_info["height"], image_info["width"], 1))
				else:
					if type(ann["segmentation"][0]) == list:  # polygon
						rle = mask.frPyObjects(
							ann["segmentation"],
							image_info["height"],
							image_info["width"],
						)
					else:
						rle = ann["segmentation"]
						for i in range(len(rle)):
							if not isinstance(rle[i]["counts"], bytes):
								rle[i]["counts"] = rle[i]["counts"].encode()
					m = mask.decode(rle)
				m = np.sum(
					m, axis=2
				)  # sometimes there are multiple binary map (corresponding to multiple segs)
				m = m.astype(np.uint8)  # convert to np.uint8
				masks.append(m)
		else:
			masks = [mask_json]
		bboxes_valid = [1]*len(bboxes_labels)
		masks_valid = [1]*len(bboxes_labels)
		masks = np.stack(masks, axis=0)
		masks = torch.from_numpy(masks)
		labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
		inference = True

		return (
			image_path,
			image,
			image_clip,
			conversations,
			masks,
			labels,
			bboxes_labels,
			bboxes_valid,
			masks_valid,
			resize,
			None,
			None,
			inference,
		)