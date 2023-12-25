import argparse
import os
import sys
import json
import tqdm
import copy
from queue import PriorityQueue
import functools
import spacy
nlp = spacy.load("en_core_web_sm")

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import OwlViTProcessor

from VisualSearch.model.VSM import VSMForCausalLM
from VisualSearch.model.llava import conversation as conversation_lib
from VisualSearch.model.llava.mm_utils import tokenizer_image_token
from VisualSearch.utils.utils import expand2square
from VisualSearch.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
						 DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
	parser = argparse.ArgumentParser(description="Visual Search Evaluation")
	parser.add_argument("--version", default="craigwu/seal_vsm_7b")
	parser.add_argument("--benchmark-folder", default="vstar_bench", type=str)
	parser.add_argument("--visualization", action="store_true", default=False)
	parser.add_argument("--output_path", default="", type=str)
	parser.add_argument("--confidence_low", default=0.3, type=float)
	parser.add_argument("--confidence_high", default=0.5, type=float)
	parser.add_argument("--target_cue_threshold", default=6.0, type=float)
	parser.add_argument("--target_cue_threshold_decay", default=0.7, type=float)
	parser.add_argument("--target_cue_threshold_minimum", default=3.0, type=float)
	parser.add_argument("--minimum_size_scale", default=4.0, type=float)
	parser.add_argument("--minimum_size", default=224, type=int)
	parser.add_argument("--model_max_length", default=512, type=int)
	parser.add_argument(
		"--vision-tower", default="openai/clip-vit-large-patch14", type=str
	)
	parser.add_argument("--use_mm_start_end", action="store_true", default=True)
	parser.add_argument(
		"--conv_type",
		default="llava_v1",
		type=str,
		choices=["llava_v1", "llava_llama_2"],
	)
	return parser.parse_args(args)

def tranverse(token):
	children = [_ for _ in token.children]
	if len(children) == 0:
		return token.i, token.i
	left_i = token.i
	right_i = token.i
	for child in children:
		child_left_i, child_right_i = tranverse(child)
		left_i = min(left_i, child_left_i)
		right_i = max(right_i, child_right_i)
	return left_i, right_i
def get_noun_chunks(token):
	left_children = []
	right_children = []
	for child in token.children:
		if child.i < token.i:
			left_children.append(child)
		else:
			right_children.append(child)

	start_token_i = token.i
	for left_child in left_children[::-1]:
		if left_child.dep_ in ['amod', 'compound', 'poss']:
			start_token_i, _ = tranverse(left_child)
		else:
			break
	end_token_i = token.i
	for right_child in right_children:
		if right_child.dep_ in ['relcl', 'prep']:
			_, end_token_i = tranverse(right_child)
		else:
			break
	return start_token_i, end_token_i

def filter_chunk_list(chunks):
	def overlap(min1, max1, min2, max2):
		return min(max1, max2) - max(min1, min2)
	chunks = sorted(chunks, key=lambda chunk: chunk[1]-chunk[0], reverse=True)
	filtered_chunks = []
	for chunk in chunks:
		flag=True
		for exist_chunk in filtered_chunks:
			if overlap(exist_chunk[0], exist_chunk[1], chunk[0], chunk[1]) >= 0:
				flag = False
				break
		if flag:
			filtered_chunks.append(chunk)
	return sorted(filtered_chunks, key=lambda chunk: chunk[0])

def extract_noun_chunks(expression):
	doc = nlp(expression)
	cur_chunks = []
	for token in doc:
		if token.pos_ not in ["NOUN", "PRON"]:
			continue
		cur_chunks.append(get_noun_chunks(token))
	cur_chunks = filter_chunk_list(cur_chunks)
	cur_chunks = [doc[chunk[0]:chunk[1]+1].text for chunk in cur_chunks]
	return cur_chunks

def preprocess(
	x,
	pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
	pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
	img_size=1024,
) -> torch.Tensor:
	"""Normalize pixel values and pad to a square input."""
	# Normalize colors
	x = (x - pixel_mean) / pixel_std
	# Pad
	h, w = x.shape[-2:]
	padh = img_size - h
	padw = img_size - w
	x = F.pad(x, (0, padw, 0, padh))
	return x

def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		 (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
	img_w, img_h = size
	b = box_cxcywh_to_xyxy(out_bbox)
	b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	return b

class VSM:
	def __init__(self, args):
		kwargs = {}
		kwargs['torch_dtype'] = torch.bfloat16
		kwargs['device_map'] = 'cuda'
		kwargs['is_eval'] = True
		vsm_tokenizer = AutoTokenizer.from_pretrained(
				args.version,
				cache_dir=None,
				model_max_length=args.model_max_length,
				padding_side="right",
				use_fast=False,
			)
		vsm_tokenizer.pad_token = vsm_tokenizer.unk_token
		loc_token_idx = vsm_tokenizer("[LOC]", add_special_tokens=False).input_ids[0]
		vsm_model = VSMForCausalLM.from_pretrained(
				args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, loc_token_idx=loc_token_idx, **kwargs
			)
		vsm_model.get_model().initialize_vision_modules(vsm_model.get_model().config)
		vision_tower = vsm_model.get_model().get_vision_tower().cuda().to(dtype=torch.bfloat16)
		vsm_image_processor = vision_tower.image_processor
		vsm_model.eval()
		clip_image_processor = CLIPImageProcessor.from_pretrained(vsm_model.config.vision_tower)
		transform = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
		self.model = vsm_model
		self.vsm_tokenizer = vsm_tokenizer
		self.vsm_image_processor = vsm_image_processor
		self.clip_image_processor = clip_image_processor
		self.transform = transform
		self.conv_type = args.conv_type
		self.use_mm_start_end = args.use_mm_start_end
	
	@torch.inference_mode()
	def inference(self, image, question, mode='segmentation'):
		conv = conversation_lib.conv_templates[self.conv_type].copy()
		conv.messages = []
		prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
		if self.use_mm_start_end:
			replace_token = ( DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
			prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
		conv.append_message(conv.roles[0], prompt)
		conv.append_message(conv.roles[1], "")
		prompt = conv.get_prompt()

		background_color = tuple(int(x*255) for x in self.clip_image_processor.image_mean)
		image_clip = self.clip_image_processor.preprocess(expand2square(image, background_color), return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()

		image_clip = image_clip.bfloat16()
		image = np.array(image)
		original_size_list = [image.shape[:2]]
		image = self.transform(images=image, return_tensors="pt")['pixel_values'].cuda()
		resize_list = [image.shape[:2]]
		image = image.bfloat16()
		input_ids = tokenizer_image_token(prompt, self.vsm_tokenizer, return_tensors="pt")
		input_ids = input_ids.unsqueeze(0).cuda()

		output_ids, pred_masks, det_result = self.model.inference(
			image_clip,
			image,
			input_ids,
			resize_list,
			original_size_list,
			max_new_tokens=100,
			tokenizer=self.vsm_tokenizer,
			mode = mode
		)
		if mode == 'segmentation':
			pred_mask = pred_masks[0]
			pred_mask = torch.clamp(pred_mask, min=0)
			return pred_mask[-1]

		elif mode == 'vqa':
			input_token_len = input_ids.shape[1]
			n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
			if n_diff_input_output > 0:
				print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
			text_output = self.vsm_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
			text_output = text_output.replace("\n", "").replace("  ", " ").strip()
			return text_output
		
		elif mode == 'detection':
			pred_mask = pred_masks[0]
			pred_mask = torch.clamp(pred_mask, min=0)
			return det_result['pred_boxes'][0].cpu(), det_result['pred_logits'][0].sigmoid().cpu(), pred_mask[-1]

def refine_bbox(bbox, image_width, image_height):
	bbox[0] = max(0, bbox[0])
	bbox[1] = max(0, bbox[1])
	bbox[2] = min(bbox[2], image_width-bbox[0])
	bbox[3] = min(bbox[3], image_height-bbox[1])
	return bbox

def split_4subpatches(current_patch_bbox):
	hw_ratio = current_patch_bbox[3] / current_patch_bbox[2]
	if hw_ratio >= 2:
		return 1, 4
	elif hw_ratio <= 0.5:
		return 4, 1
	else:
		return 2, 2

def get_sub_patches(current_patch_bbox, num_of_width_patches, num_of_height_patches):
	width_stride = int(current_patch_bbox[2]//num_of_width_patches)
	height_stride = int(current_patch_bbox[3]/num_of_height_patches)
	sub_patches = []
	for j in range(num_of_height_patches):
		for i in range(num_of_width_patches):
			sub_patch_width = current_patch_bbox[2] - i*width_stride if i == num_of_width_patches-1 else width_stride
			sub_patch_height = current_patch_bbox[3] - j*height_stride if j == num_of_height_patches-1 else height_stride
			sub_patch = [current_patch_bbox[0]+i*width_stride, current_patch_bbox[1]+j*height_stride, sub_patch_width, sub_patch_height]
			sub_patches.append(sub_patch)
	return sub_patches, width_stride, height_stride

def get_subpatch_scores(score_heatmap, current_patch_bbox, sub_patches):
	total_sum = (score_heatmap/(current_patch_bbox[2]*current_patch_bbox[3])).sum()
	sub_scores = []
	for sub_patch in sub_patches:
		bbox = [(sub_patch[0]-current_patch_bbox[0]), sub_patch[1]-current_patch_bbox[1], sub_patch[2], sub_patch[3]]
		score = (score_heatmap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]/(current_patch_bbox[2]*current_patch_bbox[3])).sum()
		if total_sum > 0:
			score /= total_sum
		else:
			score *= 0
		sub_scores.append(score)
	return sub_scores

def normalize_score(score_heatmap):
	max_score = score_heatmap.max()
	min_score = score_heatmap.min()
	if max_score != min_score:
		score_heatmap = (score_heatmap - min_score) / (max_score - min_score)
	else:
		score_heatmap = score_heatmap * 0
	return score_heatmap

def iou(bbox1, bbox2):
	x1 = max(bbox1[0], bbox2[0])
	y1 = max(bbox1[1], bbox2[1])
	x2 = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
	y2 = min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
	inter_area = max(0, x2 - x1) * max(0, y2 - y1)
	return inter_area/(bbox1[2]*bbox1[3]+bbox2[2]*bbox2[3]-inter_area)

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
import cv2
from matplotlib import pyplot as plt
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
	"""Visualizes a single bounding box on the image"""
	x_min, y_min, w, h = bbox
	x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
	
	((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    
	cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
	cv2.putText(
		img,
		text=class_name,
		org=(x_min, y_min - int(0.3 * text_height)),
		fontFace=cv2.FONT_HERSHEY_SIMPLEX,
		fontScale=0.5, 
		color=TEXT_COLOR, 
		lineType=cv2.LINE_AA,
	)
	return img
def show_heatmap_on_image(img: np.ndarray,
					  mask: np.ndarray,
					  use_rgb: bool = False,
					  colormap: int = cv2.COLORMAP_JET,
					  image_weight: float = 0.5) -> np.ndarray:
	mask = np.clip(mask, 0, 1)
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
	if use_rgb:
		heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
	heatmap = np.float32(heatmap) / 255

	if np.max(img) > 1:
		raise Exception(
			"The input image should np.float32 in the range [0, 1]")

	if image_weight < 0 or image_weight > 1:
		raise Exception(
			f"image_weight should be in the range [0, 1].\
				Got: {image_weight}")

	cam = (1 - image_weight) * heatmap + image_weight * img
	cam = cam / np.max(cam)
	return np.uint8(255 * cam)
def vis_heatmap(image, heatmap, use_rgb=False):
	max_v =  np.max(heatmap)
	min_v =  np.min(heatmap)
	if max_v != min_v:
		heatmap = (heatmap - min_v) / (max_v - min_v)
	heatmap_image = show_heatmap_on_image(image.astype(float)/255., heatmap, use_rgb=use_rgb)
	return heatmap_image

def visualize_search_path(image, search_path, search_length, target_bbox, label, save_path):
	context_cue_list = []
	whole_image = image	
	os.makedirs(save_path, exist_ok=True)
	whole_image.save(os.path.join(save_path, 'whole_image.jpg'))

	whole_image = np.array(whole_image)
	if target_bbox is not None:
		whole_image = visualize_bbox(whole_image.copy(), target_bbox, class_name="gt: "+label, color=(255,0,0))
	for step_i, node in enumerate(search_path):
		if step_i + 1 > search_length:
			break
		current_patch_box = node['bbox']
		if 'detection_result' in node:
			final_patch_image = image.crop((current_patch_box[0],current_patch_box[1],current_patch_box[0]+current_patch_box[2], current_patch_box[1]+current_patch_box[3]))
			final_patch_image.save(os.path.join(save_path, 'final_patch_image.jpg'))
			final_search_result = visualize_bbox(np.array(final_patch_image), node['detection_result'], class_name='search result', color=(255,0,0))
			final_search_result = cv2.cvtColor(final_search_result, cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(save_path, 'search_result.jpg'), final_search_result)
		cur_whole_image = visualize_bbox(whole_image.copy(), current_patch_box, class_name="step-{}".format(step_i+1), color=(0,0,255))
		# if step_i != len(search_path)-1:
		# 	next_patch_box = search_path[step_i+1]['bbox']
		# 	cur_whole_image = visualize_bbox(cur_whole_image, next_patch_box, class_name="next-step", color=(0,255,0))
		cur_whole_image = cv2.cvtColor(cur_whole_image, cv2.COLOR_RGB2BGR)
		cv2.imwrite(os.path.join(save_path, 'step_{}.jpg'.format(step_i+1)), cur_whole_image)
		
		cur_patch_image = image.crop((current_patch_box[0],current_patch_box[1],current_patch_box[0]+current_patch_box[2], current_patch_box[1]+current_patch_box[3]))
		if 'context_cue' in node:
			context_cue = node['context_cue']
			context_cue_list.append('step{}: {}'.format(step_i+1, context_cue)+'\n')
		if 'final_heatmap' in node:
			score_map = node['final_heatmap']
			score_map = vis_heatmap(np.array(cur_patch_image), score_map, use_rgb=True)
			score_map = cv2.cvtColor(score_map, cv2.COLOR_RGB2BGR)
			cv2.imwrite(os.path.join(save_path, 'step_{}_heatmap.jpg'.format(step_i+1)), score_map)

	with open(os.path.join(save_path, 'context_cue.txt'),"w") as f:
		f.writelines(context_cue_list)
		
@functools.total_ordering
class Prioritize:

	def __init__(self, priority, item):
		self.priority = priority
		self.item = item

	def __eq__(self, other):
		return self.priority == other.priority

	def __lt__(self, other):
		return self.priority < other.priority
def visual_search_queue(vsm, image, target_object_name, current_patch, search_path, queue,  smallest_size=224, confidence_high=0.5, target_cue_threshold=6.0, target_cue_threshold_decay=0.7, target_cue_threshold_minimum=3.0):
	current_patch_bbox = current_patch['bbox']
	current_patch_scale_level = current_patch['scale_level']

	image_patch = image.crop((int(current_patch_bbox[0]), int(current_patch_bbox[1]), int(current_patch_bbox[0]+current_patch_bbox[2]), int(current_patch_bbox[1]+current_patch_bbox[3])))
	# whehter we can detect the target object on the current image patch
	question = "Please locate the {} in this image.".format(target_object_name)
	pred_bboxes, pred_logits, target_cue_heatmap = vsm.inference(copy.deepcopy(image_patch), question, mode='detection')
	if len(pred_logits) > 0:
		top_index = pred_logits.view(-1).argmax()
		top_logit = pred_logits.view(-1).max()
		final_bbox = pred_bboxes[top_index].view(4)
		final_bbox = final_bbox * torch.Tensor([image_patch.width, image_patch.height, image_patch.width, image_patch.height])
		final_bbox[:2] -= final_bbox[2:] / 2
		if top_logit > confidence_high:
			search_path[-1]['detection_result'] = final_bbox
			# only return multiple detected instances on the whole image
			if len(search_path) == 1:
				all_valid_boxes = pred_bboxes[pred_logits.view(-1)>0.5].view(-1, 4)
				all_valid_boxes = all_valid_boxes * torch.Tensor([[image_patch.width, image_patch.height, image_patch.width, image_patch.height]])
				all_valid_boxes[:, :2] -= all_valid_boxes[:, 2:] / 2
				return True, search_path, all_valid_boxes
			return True, search_path, None
		else:
			search_path[-1]['temp_detection_result'] = (top_logit, final_bbox)

	### current patch is already the smallest unit
	if min(current_patch_bbox[2], current_patch_bbox[3]) <= smallest_size:
		return False, search_path, None

	target_cue_heatmap = target_cue_heatmap.view(current_patch_bbox[3], current_patch_bbox[2], 1)
	score_max = target_cue_heatmap.max().item()
	# check whether the target cue is prominent
	threshold = max(target_cue_threshold_minimum, target_cue_threshold*(target_cue_threshold_decay)**(current_patch_scale_level-1))
	if score_max > threshold:
		target_cue_heatmap = normalize_score(target_cue_heatmap)
		final_heatmap = target_cue_heatmap
	else:
		question = "According to the common sense knowledge and possible visual cues, what is the most likely location of the {} in the image?".format(target_object_name)
		vqa_results = vsm.inference(copy.deepcopy(image_patch), question, mode='vqa')

		possible_location_phrase = vqa_results.split('most likely to appear')[-1].strip()
		if possible_location_phrase.endswith('.'):
			possible_location_phrase = possible_location_phrase[:-1]
		possible_location_phrase = possible_location_phrase.split(target_object_name)[-1]
		noun_chunks = extract_noun_chunks(possible_location_phrase)
		if len(noun_chunks) == 1:
			possible_location_phrase = noun_chunks[0]
		else:
			possible_location_phrase = "region {}".format(possible_location_phrase)
		question = "Please locate the {} in this image.".format(possible_location_phrase)
		context_cue_heatmap = vsm.inference(copy.deepcopy(image_patch), question, mode='segmentation').view(current_patch_bbox[3], current_patch_bbox[2], 1)
		context_cue_heatmap = normalize_score(context_cue_heatmap)
		final_heatmap = context_cue_heatmap

	current_patch_index = len(search_path)-1
	if score_max <= threshold:
		search_path[current_patch_index]['context_cue'] = vqa_results + "#" + possible_location_phrase
	search_path[current_patch_index]['final_heatmap'] = final_heatmap.cpu().numpy()
	
	### split the current patch into 4 sub-patches
	basic_sub_patches, sub_patch_width, sub_patch_height = get_sub_patches(current_patch_bbox, *split_4subpatches(current_patch_bbox))

	tmp_patch = current_patch
	basic_sub_scores = [0]*len(basic_sub_patches)
	while True:
		tmp_score_heatmap = tmp_patch['final_heatmap']
		tmp_sub_scores = get_subpatch_scores(tmp_score_heatmap, tmp_patch['bbox'],  basic_sub_patches)
		basic_sub_scores = [basic_sub_scores[patch_i]+tmp_sub_scores[patch_i]/(4**tmp_patch['scale_level']) for patch_i in range(len(basic_sub_scores))]
		if  tmp_patch['parent_index'] == -1:
			break
		else:
			tmp_patch = search_path[tmp_patch['parent_index']]

	sub_patches = basic_sub_patches
	sub_scores = basic_sub_scores

	for sub_patch, sub_score in zip(sub_patches, sub_scores):
		new_patch_info = dict()
		new_patch_info['bbox'] = sub_patch
		new_patch_info['scale_level'] = current_patch_scale_level + 1
		new_patch_info['score'] = sub_score
		new_patch_info['parent_index'] = current_patch_index
		queue.put(Prioritize(-new_patch_info['score'], new_patch_info))
	
	while(not queue.empty()):
		patch_chosen = queue.get().item
		search_path.append(patch_chosen)
		success, search_path, all_valid_boxes = visual_search_queue(vsm, image, target_object_name, patch_chosen, search_path, queue, smallest_size=smallest_size, confidence_high=confidence_high, target_cue_threshold=target_cue_threshold, target_cue_threshold_decay=target_cue_threshold_decay, target_cue_threshold_minimum=target_cue_threshold_minimum)
		if success:
			return success, search_path, all_valid_boxes
	return False, search_path, None


def visual_search(vsm, image, target_object_name, target_bbox, smallest_size, confidence_high=0.5, confidence_low=0.3, target_cue_threshold=6.0, target_cue_threshold_decay=0.7, target_cue_threshold_minimum=3.0, visualize=False, save_path=None):
	if visualize:
		assert save_path is not None
	init_patch = dict()
	init_patch['bbox'] = [0,0,image.width,image.height]
	init_patch['scale_level'] = 1
	init_patch['score'] = None
	init_patch['parent_index'] = -1
	search_path = [init_patch]

	queue = PriorityQueue()
	search_successful, search_path, all_valid_boxes = visual_search_queue(vsm, image, target_object_name, init_patch, search_path, queue, smallest_size=smallest_size, confidence_high=confidence_high, target_cue_threshold=target_cue_threshold, target_cue_threshold_decay=target_cue_threshold_decay, target_cue_threshold_minimum=target_cue_threshold_minimum)
	path_length = len(search_path)
	final_step = search_path[-1]
	if not search_successful:
		# if no target is found with confidence passing confidence_high, select the target with the highest confidence during search and compare its confidence with confidence_low
		max_logit = 0
		final_step = None
		path_length = 0
		for i, search_step in enumerate(search_path):
			if 'temp_detection_result' in search_step:
				if search_step['temp_detection_result'][0] > max_logit:
					max_logit = search_step['temp_detection_result'][0]
					final_step = search_step
					path_length = i+1
		final_step['detection_result'] = final_step['temp_detection_result'][1]
		if max_logit >= confidence_low:
			search_successful = True
	if visualize:
		vis_path_length = path_length if search_successful else len(search_path)
		visualize_search_path(image, search_path, vis_path_length, target_bbox, target_object_name, save_path)
	del queue
	return final_step, path_length, search_successful, all_valid_boxes



def main(args):
	args = parse_args(args)
	vsm = VSM(args)

	benchmark_folder = args.benchmark_folder

	acc_list = []
	search_path_length_list = []

	for test_type in ['direct_attributes', 'relative_position']:
		folder = os.path.join(benchmark_folder, test_type)
		output_folder = None
		if args.visualization:
			output_folder =  os.path.join(args.output_path, test_type)
			os.makedirs(output_folder, exist_ok=True)
		image_files = filter(lambda file: '.json' not in file, os.listdir(folder))
		for image_file in tqdm.tqdm(image_files):
			image_path = os.path.join(folder, image_file)
			annotation_path = image_path.split('.')[0] + '.json'
			annotation = json.load(open(annotation_path))
			bboxs = annotation['bbox']
			object_names = annotation['target_object']

			for i, (gt_bbox, object_name) in enumerate(zip(bboxs, object_names)):
				image = Image.open(image_path).convert('RGB')
				smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
				if args.visualization:
					vis_path = os.path.join(output_folder, "{}_{}".format(image_file.split('.')[0],i))
				else:
					vis_path = None
				final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name, target_bbox=gt_bbox, smallest_size=smallest_size, confidence_high=args.confidence_high, confidence_low=args.confidence_low, target_cue_threshold=args.target_cue_threshold, target_cue_threshold_decay=args.target_cue_threshold_decay, target_cue_threshold_minimum=args.target_cue_threshold_minimum, save_path=vis_path, visualize=args.visualization)
				if search_successful:
					search_bbox = final_step['detection_result']
					search_final_patch = final_step['bbox']
					search_bbox[0] += search_final_patch[0]
					search_bbox[1] += search_final_patch[1]
					iou_i = iou(search_bbox, gt_bbox).item()
					det_acc = 1.0 if iou_i > 0.5 else 0.0
					acc_list.append(det_acc)
					search_path_length_list.append(path_length)
				else:
					acc_list.append(0)
					search_path_length_list.append(0)
	print('Avg search path length:', np.mean([search_path_length_list[i] for i in range(len(search_path_length_list)) if acc_list[i]]))
	print('Top 1 Acc:', np.mean(acc_list))

if __name__ == "__main__":
	main(sys.argv[1:])