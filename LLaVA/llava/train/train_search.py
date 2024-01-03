# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import numpy as np

import torch


import transformers

from LLaVA.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_OBJECT_TOKEN,OBJECT_TOKEN_INDEX
from torch.utils.data import Dataset
from LLaVA.llava.train.llava_trainer import LLaVATrainer

from LLaVA.llava import conversation as conversation_lib
from LLaVA.llava.model import *
from LLaVA.llava.mm_utils import tokenizer_image_token, tokenizer_image_object_token

from LLaVA.llava.utils import get_patch

from PIL import Image


local_rank = None


def rank0_print(*args):
	if local_rank == 0:
		print(*args)


@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
	version: Optional[str] = field(default="v0")
	freeze_backbone: bool = field(default=False)
	tune_mm_mlp_adapter: bool = field(default=False)
	vision_tower: Optional[str] = field(default=None)
	mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
	pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
	pretrain_mm_perceiver_adapter: Optional[str] = field(default=None)
	mm_projector_type: Optional[str] = field(default='linear')
	object_mm_projector_type: Optional[str] = field(default='perceiver')
	mm_use_im_start_end: bool = field(default=False)
	mm_use_im_patch_token: bool = field(default=True)
	mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
	data_path: str = field(default=None,
						   metadata={"help": "Path to the training data."})
	lazy_preprocess: bool = False
	is_multimodal: bool = False
	image_folder: Optional[str] = field(default=None)
	image_aspect_ratio: str = 'square'
	image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	remove_unused_columns: bool = field(default=False)
	freeze_mm_mlp_adapter: bool = field(default=False)
	mpt_attn_impl: Optional[str] = field(default="triton")
	model_max_length: int = field(
		default=512,
		metadata={
			"help":
			"Maximum sequence length. Sequences will be right padded (and possibly truncated)."
		},
	)
	double_quant: bool = field(
		default=True,
		metadata={"help": "Compress the quantization statistics through double quantization."}
	)
	quant_type: str = field(
		default="nf4",
		metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
	)
	bits: int = field(
		default=16,
		metadata={"help": "How many bits to use."}
	)
	lora_enable: bool = False
	lora_r: int = 64
	lora_alpha: int = 16
	lora_dropout: float = 0.05
	lora_weight_path: str = ""
	lora_bias: str = "none"
	group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
	from deepspeed import zero
	from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
	if hasattr(param, "ds_id"):
		if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
			if not ignore_status:
				logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
		with zero.GatheredParameters([param]):
			param = param.data.detach().cpu().clone()
	else:
		param = param.detach().cpu().clone()
	return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
	if bias == "none":
		to_return = {k: t for k, t in named_params if "lora_" in k}
	elif bias == "all":
		to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
	elif bias == "lora_only":
		to_return = {}
		maybe_lora_bias = {}
		lora_bias_names = set()
		for k, t in named_params:
			if "lora_" in k:
				to_return[k] = t
				bias_name = k.split("lora_")[0] + "bias"
				lora_bias_names.add(bias_name)
			elif "bias" in k:
				maybe_lora_bias[k] = t
		for k, t in maybe_lora_bias:
			if bias_name in lora_bias_names:
				to_return[bias_name] = t
	else:
		raise NotImplementedError
	to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
	return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
	to_return = {k: t for k, t in named_params if "lora_" not in k}
	if require_grad_only:
		to_return = {k: t for k, t in to_return.items() if t.requires_grad}
	to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
	return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
	to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
	to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
	return to_return


def find_all_linear_names(model):
	cls = torch.nn.Linear
	lora_module_names = set()
	multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
	for name, module in model.named_modules():
		if any(mm_keyword in name for mm_keyword in multimodal_keywords):
			continue
		if isinstance(module, cls):
			names = name.split('.')
			lora_module_names.add(names[0] if len(names) == 1 else names[-1])

	if 'lm_head' in lora_module_names: # needed for 16-bit
		lora_module_names.remove('lm_head')
	return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
								   output_dir: str):
	"""Collects the state dict and dump to disk."""

	if getattr(trainer.args, "tune_mm_mlp_adapter", False):
		# Only save Adapter
		keys_to_match = ['mm_projector']
		if getattr(trainer.args, "use_im_start_end", False):
			keys_to_match.extend(['embed_tokens', 'embed_in'])

		weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
		trainer.model.config.save_pretrained(output_dir)

		current_folder = output_dir.split('/')[-1]
		parent_folder = os.path.dirname(output_dir)
		if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
			if current_folder.startswith('checkpoint-'):
				mm_projector_folder = os.path.join(parent_folder, "mm_projector")
				os.makedirs(mm_projector_folder, exist_ok=True)
				torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
			else:
				torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
		return

	if trainer.deepspeed:
		torch.cuda.synchronize()
		trainer.save_model(output_dir)
		return

	state_dict = trainer.model.state_dict()
	if trainer.args.should_save:
		cpu_state_dict = {
			key: value.cpu()
			for key, value in state_dict.items()
		}
		del state_dict
		trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
	special_tokens_dict: Dict,
	tokenizer: transformers.PreTrainedTokenizer,
	model: transformers.PreTrainedModel,
):
	"""Resize tokenizer and embedding.

	Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
	"""
	num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer))

	if num_new_tokens > 0:
		input_embeddings = model.get_input_embeddings().weight.data
		output_embeddings = model.get_output_embeddings().weight.data

		input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
			dim=0, keepdim=True)
		output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
			dim=0, keepdim=True)

		input_embeddings[-num_new_tokens:] = input_embeddings_avg
		output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
				 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
	"""Tokenize a list of strings."""
	tokenized_list = [
		tokenizer(
			text,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		) for text in strings
	]
	input_ids = labels = [
		tokenized.input_ids[0] for tokenized in tokenized_list
	]
	input_ids_lens = labels_lens = [
		tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
		for tokenized in tokenized_list
	]
	return dict(
		input_ids=input_ids,
		labels=labels,
		input_ids_lens=input_ids_lens,
		labels_lens=labels_lens,
	)


def _mask_targets(target, tokenized_lens, speakers):
	# cur_idx = 0
	cur_idx = tokenized_lens[0]
	tokenized_lens = tokenized_lens[1:]
	target[:cur_idx] = IGNORE_INDEX
	for tokenized_len, speaker in zip(tokenized_lens, speakers):
		if speaker == "human":
			target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
		cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
	"""Add speaker and start/end signal on each round."""
	BEGIN_SIGNAL = "### "
	END_SIGNAL = "\n"
	conversation = header
	for sentence in source:
		from_str = sentence["from"]
		if from_str.lower() == "human":
			from_str = conversation_lib.default_conversation.roles[0]
		elif from_str.lower() == "gpt":
			from_str = conversation_lib.default_conversation.roles[1]
		else:
			from_str = 'unknown'
		sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
							 sentence["value"] + END_SIGNAL)
		if get_conversation:
			conversation += sentence["value"]
	conversation += BEGIN_SIGNAL
	return conversation


def replace_nth(sub,repl,txt,nth):
	arr=txt.split(sub)
	part1=sub.join(arr[:nth])
	part2=sub.join(arr[nth:])
	
	return part1+repl+part2

def preprocess_multimodal(
	sources: Sequence[str],
	data_args: DataArguments,
	object_str_list=None
) -> Dict:
	is_multimodal = data_args.is_multimodal
	if not is_multimodal:
		return sources

	for source in sources:
		for sentence in source:
			if DEFAULT_IMAGE_TOKEN in sentence['value']:
				sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
				sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
				sentence['value'] = sentence['value'].strip()
				if "mmtag" in conversation_lib.default_conversation.version:
					sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
			replace_token = DEFAULT_IMAGE_TOKEN
			if data_args.mm_use_im_start_end:
				replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
			sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

			if DEFAULT_OBJECT_TOKEN in sentence["value"]:
				num = sentence["value"].count(DEFAULT_OBJECT_TOKEN)
				for i in range(num):
					sentence["value"] = replace_nth(DEFAULT_OBJECT_TOKEN, object_str_list[i], sentence["value"], i+1)

	return sources


def preprocess_llama_2(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False,
	has_object: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations

	if has_image:
		if has_object:
			input_ids = torch.stack([tokenizer_image_object_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
		else:
			input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()

	assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

	# Mask targets
	sep = "[/INST] "
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep2)
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep

			if has_image:
				if has_object:
					round_len = len(tokenizer_image_object_token(rou, tokenizer))
					instruction_len = len(tokenizer_image_object_token(parts[0], tokenizer)) - 2
				else:
					round_len = len(tokenizer_image_token(rou, tokenizer))
					instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 2

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess_v1(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False,
	has_object: bool = False
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations

	if has_image:
		if has_object:
			input_ids = torch.stack([tokenizer_image_object_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
		else:
			input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	else:
		input_ids = tokenizer(
			conversations,
			return_tensors="pt",
			padding="longest",
			max_length=tokenizer.model_max_length,
			truncation=True,
		).input_ids

	targets = input_ids.clone()

	assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

	# Mask targets
	sep = conv.sep + conv.roles[1] + ": "
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep2)
		cur_len = 1
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep

			if has_image:
				if has_object:
					round_len = len(tokenizer_image_object_token(rou, tokenizer))
					instruction_len = len(tokenizer_image_object_token(parts[0], tokenizer)) - 2
				else:
					round_len = len(tokenizer_image_token(rou, tokenizer))
					instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
			else:
				round_len = len(tokenizer(rou).input_ids)
				instruction_len = len(tokenizer(parts[0]).input_ids) - 2

			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess_mpt(
	sources,
	tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
	conv = conversation_lib.default_conversation.copy()
	roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

	# Apply prompt templates
	conversations = []
	for i, source in enumerate(sources):
		if roles[source[0]["from"]] != conv.roles[0]:
			# Skip the first one if it is not from human
			source = source[1:]

		conv.messages = []
		for j, sentence in enumerate(source):
			role = roles[sentence["from"]]
			assert role == conv.roles[j % 2], f"{i}"
			conv.append_message(role, sentence["value"])
		conversations.append(conv.get_prompt())

	# Tokenize conversations
	input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
	targets = input_ids.clone()
	assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

	# Mask targets
	sep = conv.sep + conv.roles[1]
	for conversation, target in zip(conversations, targets):
		total_len = int(target.ne(tokenizer.pad_token_id).sum())

		rounds = conversation.split(conv.sep)
		re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
		for conv_idx in range(3, len(rounds), 2):
			re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
		cur_len = 0
		target[:cur_len] = IGNORE_INDEX
		for i, rou in enumerate(re_rounds):
			if rou == "":
				break

			parts = rou.split(sep)
			if len(parts) != 2:
				break
			parts[0] += sep
			round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
			instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
			target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

			cur_len += round_len
		target[cur_len:] = IGNORE_INDEX

		if cur_len < tokenizer.model_max_length:
			if cur_len != total_len:
				target[:] = IGNORE_INDEX
				print(
					f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
					f" (ignored)"
				)

	return dict(
		input_ids=input_ids,
		labels=targets,
	)


def preprocess_plain(
	sources: Sequence[str],
	tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
	# add end signal and concatenate together
	conversations = []
	for source in sources:
		assert len(source) == 2
		assert DEFAULT_IMAGE_TOKEN in source[0]['value']
		source[0]['value'] = DEFAULT_IMAGE_TOKEN
		conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
		conversations.append(conversation)
	# tokenize conversations
	input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
	targets = copy.deepcopy(input_ids)
	for target, source in zip(targets, sources):
		tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
		target[:tokenized_len] = IGNORE_INDEX

	return dict(input_ids=input_ids, labels=targets)


def preprocess(
	sources: Sequence[str],
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False,
	has_object: bool = False,
) -> Dict:
	"""
	Given a list of sources, each is a conversation list. This transform:
	1. Add signal '### ' at the beginning each sentence, with end signal '\n';
	2. Concatenate conversations together;
	3. Tokenize the concatenated conversation;
	4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
	"""
	if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
		return preprocess_plain(sources, tokenizer)
	if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
		return preprocess_llama_2(sources, tokenizer, has_image=has_image, has_object=has_object)
	if conversation_lib.default_conversation.version.startswith("v1"):
		return preprocess_v1(sources, tokenizer, has_image=has_image, has_object=has_object)
	if conversation_lib.default_conversation.version == "mpt":
		return preprocess_mpt(sources, tokenizer)
	# add end signal and concatenate together
	conversations = []
	for source in sources:
		header = f"{conversation_lib.default_conversation.system}\n\n"
		conversation = _add_speaker_and_signal(header, source)
		conversations.append(conversation)
	# tokenize conversations
	def get_tokenize_len(prompts):
		return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

	if has_image:
		input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
	else:
		conversations_tokenized = _tokenize_fn(conversations, tokenizer)
		input_ids = conversations_tokenized["input_ids"]

	targets = copy.deepcopy(input_ids)
	for target, source in zip(targets, sources):
		if has_image:
			tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
		else:
			tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
		speakers = [sentence["from"] for sentence in source]
		_mask_targets(target, tokenized_lens, speakers)

	return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(Dataset):
	"""Dataset for supervised fine-tuning."""

	def __init__(self, data_path: str,
				 tokenizer: transformers.PreTrainedTokenizer,
				 data_args: DataArguments):
		super(LazySupervisedDataset, self).__init__()
		llava_data = json.load(open(os.path.join(data_path, 'llava_instruct_data.json'), "r"))
		GQA_search_data = json.load(open(os.path.join(data_path, 'GQA_data.json'), "r"))
		vaw_search_data = json.load(open(os.path.join(data_path, 'vaw_attribute_data.json'), "r"))
		negative_data = json.load(open(os.path.join(data_path, 'negative_data.json'), "r"))
		llava_focus_40k = json.load(open(os.path.join(data_path, 'llava_focus_data.json'), "r"))
		spatial = json.load(open(os.path.join(data_path, 'spatial_relation_data.json'), "r"))
		spatial = spatial + copy.deepcopy(spatial)
		list_data_dict =  vaw_search_data + llava_data + GQA_search_data + llava_focus_40k + spatial + negative_data 

		rank0_print("Formatting inputs...Skip in lazy mode")
		self.tokenizer = tokenizer
		self.list_data_dict = list_data_dict
		self.data_args = data_args

	def __len__(self):
		return len(self.list_data_dict)
	@property
	def lengths(self):
		length_list = []
		for sample in self.list_data_dict:
			img_tokens = 128 if 'image' in sample else 0
			length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
		return length_list

	@property
	def modality_lengths(self):
		length_list = []
		for sample in self.list_data_dict:
			cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
			cur_len = cur_len if 'image' in sample else -cur_len
			length_list.append(cur_len)
		return length_list
	
	def normalize_bbox(self, bbox, image_width, image_height):
		normalized_bbox = [bbox[0]/image_width, bbox[1]/image_height, (bbox[0]+bbox[2])/image_width, (bbox[1]+bbox[3])/image_height]
		normalized_bbox = [np.clip(_, 0, 1) for _ in normalized_bbox]
		return normalized_bbox

	def __getitem__(self, i) -> Dict[str, torch.Tensor]:
		sources = self.list_data_dict[i]
		if isinstance(i, int):
			sources = [sources]
		assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
		crop_size = self.data_args.image_processor.crop_size
		is_search = False # whether the sample contains target object
		if 'image' in sources[0]:
			image_file = self.list_data_dict[i]['image']
			image_folder = self.data_args.image_folder
			processor = self.data_args.image_processor
			is_search = 'search' in sources[0]
			# indicate using the linear projection (long token sequence) or re-sampler projection (short sequence) for images and targe object crops
			# always pad to three target objects
			images_long = 1
			objects_long = [0, 0, 0]
			
			image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
			object_features = []
			if is_search:
				target_instances = sources[0]['target_instances']
				bbox_list = [instance['bbox'] for instance in target_instances]
				instance_name_list = [instance['name'] for instance in target_instances]
				for target_instance in target_instances:
					bbox = target_instance['bbox']
					image_width = image.width
					image_height = image.height
					# crop a larger bbox to include some context
					resized_bbox = get_patch(bbox, image_width, image_height, patch_scale=1.2)
					image_patch = image.crop((resized_bbox[0], resized_bbox[1], resized_bbox[2], resized_bbox[3]))
					image_patch = image_patch.resize((self.data_args.image_processor.crop_size['width'],self.data_args.image_processor.crop_size['height']))
					object_features.append(processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0])
			# re-sampler projection for image and linear projection for object when there is a single target object 
			if len(object_features) == 1:
				objects_long[-1] = 1 
				images_long = 0
			while len(object_features) < 3:
				object_features.insert(0, torch.zeros(3, crop_size['height'], crop_size['width']))

			if self.data_args.image_aspect_ratio == 'pad':
				def expand2square(pil_img, background_color):
					width, height = pil_img.size
					if width == height:
						return pil_img, 0, 0
					elif width > height:
						result = Image.new(pil_img.mode, (width, width), background_color)
						result.paste(pil_img, (0, (width - height) // 2))
						return result, 0, (width - height) // 2
					else:
						result = Image.new(pil_img.mode, (height, height), background_color)
						result.paste(pil_img, ((height - width) // 2, 0))
						return result, (height - width) // 2, 0
				image, left, top = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
				if is_search:
					for bbox in bbox_list:
						bbox[0] += left
						bbox[1] += top
					bbox_list = [self.normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
					object_str_list = []
					for name, bbox in zip(instance_name_list, bbox_list):
						object_str_list.append("{} {} at location [{:.3f},{:.3f},{:.3f},{:.3f}]".format(name, DEFAULT_OBJECT_TOKEN, bbox[0], bbox[1], bbox[2], bbox[3]))
				image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
			else:
				image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
			if is_search:
				sources = preprocess_multimodal(
				copy.deepcopy([e["conversations"] for e in sources]),
				self.data_args, object_str_list=object_str_list)
			else:
				sources = preprocess_multimodal(
				copy.deepcopy([e["conversations"] for e in sources]),
				self.data_args)
		else:
			sources = copy.deepcopy([e["conversations"] for e in sources])
		data_dict = preprocess(
			sources,
			self.tokenizer,
			has_image=('image' in self.list_data_dict[i]),
			has_object = is_search)
		if isinstance(i, int):
			data_dict = dict(input_ids=data_dict["input_ids"][0],
							 labels=data_dict["labels"][0])

		if 'image' in self.list_data_dict[i]:
			data_dict['object_features'] = object_features
			data_dict['images_long'] = images_long
			data_dict['objects_long'] = objects_long
		elif self.data_args.is_multimodal:
			data_dict['object_features'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]*3
			data_dict['images_long'] = 1
			data_dict['objects_long'] = [0, 0, 0]
		# image exist in the data
		if 'image' in self.list_data_dict[i]:
			data_dict['image'] = image
		elif self.data_args.is_multimodal:
			# image does not exist in the data, but the model is multimodal
			data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
		return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
	"""Collate examples for supervised fine-tuning."""

	tokenizer: transformers.PreTrainedTokenizer

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		input_ids, labels = tuple([instance[key] for instance in instances]
								  for key in ("input_ids", "labels"))
		pad_image_list = []
		pad_object_lens = []
		object_padded_input_ids = []
		object_padded_labels = []
		for i, (input_id, label) in enumerate(zip(input_ids, labels)):
			pad_object_len = 0
			if (input_id == IMAGE_TOKEN_INDEX).sum() == 0:
				input_id = torch.cat([input_id[0:1], torch.tensor([IMAGE_TOKEN_INDEX]*1, dtype=torch.long), input_id[1:]], 0)
				label = torch.cat([label[0:1], torch.tensor([IGNORE_INDEX]*1, dtype=torch.long), label[1:]], 0)
				pad_image_list.append(True)
			else:
				pad_image_list.append(False)
			image_token_pos = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
			count = torch.sum(input_id==OBJECT_TOKEN_INDEX).item()
			if (input_id == IMAGE_TOKEN_INDEX).sum() > 0:
				if count < 3:
					pad_object_len = 3-count
					input_id = torch.cat([input_id[:image_token_pos+1], torch.tensor([OBJECT_TOKEN_INDEX]*pad_object_len, dtype=torch.long), input_id[image_token_pos+1:]], 0)
					label = torch.cat([label[:image_token_pos+1], torch.tensor([IGNORE_INDEX]*pad_object_len, dtype=torch.long), label[image_token_pos+1:]], 0)
			pad_object_lens.append((image_token_pos, pad_object_len))
			object_padded_input_ids.append(input_id)
			object_padded_labels.append(label)
		input_ids = object_padded_input_ids
		labels = object_padded_labels
		input_ids = torch.nn.utils.rnn.pad_sequence(
			input_ids,
			batch_first=True,
			padding_value=self.tokenizer.pad_token_id)
		labels = torch.nn.utils.rnn.pad_sequence(labels,
												 batch_first=True,
												 padding_value=IGNORE_INDEX)
		input_ids = input_ids[:, :self.tokenizer.model_max_length]
		labels = labels[:, :self.tokenizer.model_max_length]
		batch = dict(
			input_ids=input_ids,
			labels=labels,
			attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
		)

		for i, pad_image in enumerate(pad_image_list):
			if pad_image:
				batch['attention_mask'][i, 1] = False

		for i, (image_token_pos, pad_object_len) in enumerate(pad_object_lens):
			if pad_object_len > 0:
				batch['attention_mask'][i, image_token_pos+1:image_token_pos+1+pad_object_len] = False

		if 'image' in instances[0]:
			images = [instance['image'] for instance in instances]
			if all(x is not None and x.shape == images[0].shape for x in images):
				batch['images'] = torch.stack(images)
			else:
				batch['images'] = images

			images_long = [instance['images_long'] for instance in instances]
			batch['images_long'] = torch.tensor(images_long).view(-1, 1).to(torch.bool)
			objects_long = [instance['objects_long'] for instance in instances]
			batch['objects_long'] = torch.tensor(objects_long).view(-1, 1).to(torch.bool)
			object_features = []
			for instance in instances:
				object_features.extend(instance['object_features'])
			if len(object_features) > 0:
				batch['object_features'] = torch.stack(object_features)
			else:
				batch['object_features'] = object_features
		return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
								data_args) -> Dict:
	"""Make dataset and collator for supervised fine-tuning."""
	train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
								data_path=data_args.data_path,
								data_args=data_args)
	data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
	return dict(train_dataset=train_dataset,
				eval_dataset=None,
				data_collator=data_collator)


def train():
	global local_rank

	parser = transformers.HfArgumentParser(
		(ModelArguments, DataArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	local_rank = training_args.local_rank
	compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

	bnb_model_from_pretrained_args = {}
	if training_args.bits in [4, 8]:
		from transformers import BitsAndBytesConfig
		bnb_model_from_pretrained_args.update(dict(
			device_map={"": training_args.device},
			load_in_4bit=training_args.bits == 4,
			load_in_8bit=training_args.bits == 8,
			quantization_config=BitsAndBytesConfig(
				load_in_4bit=training_args.bits == 4,
				load_in_8bit=training_args.bits == 8,
				llm_int8_threshold=6.0,
				llm_int8_has_fp16_weight=False,
				bnb_4bit_compute_dtype=compute_dtype,
				bnb_4bit_use_double_quant=training_args.double_quant,
				bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
			)
		))

	if model_args.vision_tower is not None:
		if 'mpt' in model_args.model_name_or_path:
			config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
			config.attn_config['attn_impl'] = training_args.mpt_attn_impl
			model = LlavaMPTForCausalLM.from_pretrained(
				model_args.model_name_or_path,
				config=config,
				cache_dir=training_args.cache_dir,
				**bnb_model_from_pretrained_args
			)
		else:
			model = LlavaSearchLlamaForCausalLM.from_pretrained(
				model_args.model_name_or_path,
				cache_dir=training_args.cache_dir,
				**bnb_model_from_pretrained_args
			)
	else:
		model = transformers.LlamaForCausalLM.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=training_args.cache_dir,
			**bnb_model_from_pretrained_args
		)
	model.config.use_cache = False

	if model_args.freeze_backbone:
		model.model.requires_grad_(False)

	if training_args.bits in [4, 8]:
		from peft import prepare_model_for_kbit_training
		model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
		model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

	if training_args.gradient_checkpointing:
		if hasattr(model, "enable_input_require_grads"):
			model.enable_input_require_grads()
		else:
			def make_inputs_require_grad(module, input, output):
				output.requires_grad_(True)
			model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

	if training_args.lora_enable:
		from peft import LoraConfig, get_peft_model
		lora_config = LoraConfig(
			r=training_args.lora_r,
			lora_alpha=training_args.lora_alpha,
			target_modules=find_all_linear_names(model),
			lora_dropout=training_args.lora_dropout,
			bias=training_args.lora_bias,
			task_type="CAUSAL_LM",
		)
		if training_args.bits == 16:
			if training_args.bf16:
				model.to(torch.bfloat16)
			if training_args.fp16:
				model.to(torch.float16)
		rank0_print("Adding LoRA adapters...")
		model = get_peft_model(model, lora_config)

	if 'mpt' in model_args.model_name_or_path:
		tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=training_args.cache_dir,
			model_max_length=training_args.model_max_length,
			padding_side="right"
		)
	else:
		tokenizer = transformers.AutoTokenizer.from_pretrained(
			model_args.model_name_or_path,
			cache_dir=training_args.cache_dir,
			model_max_length=training_args.model_max_length,
			padding_side="right",
			use_fast=False,
		)

	if model_args.version == "v0":
		if tokenizer.pad_token is None:
			smart_tokenizer_and_embedding_resize(
				special_tokens_dict=dict(pad_token="[PAD]"),
				tokenizer=tokenizer,
				model=model,
			)
	elif model_args.version == "v0.5":
		tokenizer.pad_token = tokenizer.unk_token
	else:
		tokenizer.pad_token = tokenizer.unk_token
		if model_args.version in conversation_lib.conv_templates:
			conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
		else:
			conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

	if model_args.vision_tower is not None:
		model.get_model().initialize_vision_modules(
			model_args=model_args,
			fsdp=training_args.fsdp
		)
		
		vision_tower = model.get_vision_tower()
		vision_tower.to(dtype=torch.float16, device=training_args.device)

		data_args.image_processor = vision_tower.image_processor
		data_args.is_multimodal = True

		model.config.image_aspect_ratio = data_args.image_aspect_ratio
		model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

		model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
		if model_args.tune_mm_mlp_adapter:
			model.requires_grad_(False)
			for p in model.get_model().mm_projector.parameters():
				p.requires_grad = True

		model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
		if training_args.freeze_mm_mlp_adapter:
			for p in model.get_model().mm_projector.parameters():
				p.requires_grad = False

		if training_args.bits in [4, 8]:
			model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

		model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
		training_args.use_im_start_end = model_args.mm_use_im_start_end
		model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
		model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

	if training_args.bits in [4, 8]:
		from peft.tuners.lora import LoraLayer
		for name, module in model.named_modules():
			if isinstance(module, LoraLayer):
				if training_args.bf16:
					module = module.to(torch.bfloat16)
			if 'norm' in name:
				module = module.to(torch.float32)
			if 'lm_head' in name or 'embed_tokens' in name:
				if hasattr(module, 'weight'):
					if training_args.bf16 and module.weight.dtype == torch.float32:
						module = module.to(torch.bfloat16)

	data_module = make_supervised_data_module(tokenizer=tokenizer,
											  data_args=data_args)
	trainer = LLaVATrainer(model=model,
					tokenizer=tokenizer,
					args=training_args,
					**data_module)

	if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
		trainer.train(resume_from_checkpoint=True)
	else:
		trainer.train()
	trainer.save_state()

	model.config.use_cache = True

	if training_args.lora_enable:
		state_dict = get_peft_state_maybe_zero_3(
			model.named_parameters(), training_args.lora_bias
		)
		non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
			model.named_parameters()
		)
		if training_args.local_rank == 0 or training_args.local_rank == -1:
			model.config.save_pretrained(training_args.output_dir)
			model.save_pretrained(training_args.output_dir, state_dict=state_dict)
			torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
	else:
		safe_save_model_for_hf_trainer(trainer=trainer,
									   output_dir=training_args.output_dir)


if __name__ == "__main__":
	train()