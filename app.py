import argparse
from copy import deepcopy
import re
import os

import bleach
import cv2
import gradio as gr
from PIL import Image
import numpy as np
import torch


from visual_search import parse_args, VSM, visual_search
from vstar_bench_eval import normalize_bbox, expand2square, VQA_LLM

import cv2
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
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

def parse_args_vqallm(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("--vqa-model-path", type=str, default="craigwu/visual_search_vqa_aug")
	parser.add_argument("--vqa-model-base", type=str, default=None)
	parser.add_argument("--conv_type", default="v1", type=str,)
	parser.add_argument("--vsm-model-path", type=str, default="craigwu/LISA_owlvit_detseg")
	parser.add_argument("--minimum_size_scale", default=4.0, type=float)
	parser.add_argument("--minimum_size", default=224, type=int)
	return parser.parse_args(args)

args = parse_args_vqallm({})
# init VQA LLM
vqa_llm = VQA_LLM(args)
# init VSM
vsm_args = parse_args({})
vsm_args.version = args.vsm_model_path
vsm = VSM(vsm_args)

missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
focus_msg = "Additional visual information to focus on: "

# Gradio
examples = [
	[
		"Based on the exact content of the flag on the roof, what can we know about its owner?",
		"./assets/example_images/flag.JPG",
	],
	[
		"What is the logo on that bag of bottles carried by the man?",
		"./assets/example_images/bag_of_bottle.jpeg",
	],
	[
		"At which conference did someone get that black mug?",
		"./assets/example_images/blackmug.JPG",
	],	
	[
		"Where to buy a mug like this based on its logo?",
		"./assets/example_images/desktop.webp",
	],
	[
		"Which company does that little doll belong to?",
		"./assets/example_images/doll.JPG",
	],
	[
		"What is the instrument held by an ape?",
		"./assets/example_images/instrument.webp",
	],
	[
		"What color is the liquid in the glass?",
		"./assets/example_images/animate_glass.jpg",
	],
	[
		"Tell me the number of that player who is shooting.",
		"./assets/example_images/nba.png",
	],
	[
		"From the information on the black framed board, how long do we have to wait in line for this attraction?",
		"./assets/example_images/queue_time.jpg",
	],
	[
		"What animal is drawn on that red signicade?",
		"./assets/example_images/signicade.JPG",
	],
	[
		"What kind of drink can we buy from that vending machine?",
		"./assets/example_images/vending_machine.jpg",
	]
]

title = "V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs"

description = """
<font size=4>
This is the demo of our SEAL framework with V* visual search mechanism. \n
**Note**: The current framework is built on top of **LLaVA-7b**. \n
**Note**: The current visual search model and search algorithm mainly focus on common objects and single instance cases.\n
</font>
"""

article = """
<p style='text-align: center'>
<a href='' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='' target='_blank'>   Github </a></p>
"""


def inference(input_str, input_image):
	## filter out special chars
	input_str = bleach.clean(input_str)

	print("input_str: ", input_str, "input_image: ", input_image)

	## input valid check
	if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
		output_str = "[Error] Invalid input: ", input_str
		return output_str, None

	# Model Inference
	# check whether we need additional visual information
	question = input_str
	image = Image.open(input_image).convert('RGB')
	image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
	prediction = vqa_llm.free_form_inference(image, question, max_new_tokens=512)
	missing_objects = []
	if missing_objects_msg in prediction:
		missing_objects = prediction.split(missing_objects_msg)[-1]
		if missing_objects.endswith('.'):
			missing_objects = missing_objects[:-1]
		missing_objects = missing_objects.split(',')
		missing_objects = [missing_object.strip() for missing_object in missing_objects]

	if len(missing_objects) == 0:
		return prediction, None, None, None

	search_result = []
	failed_objects = []
	# visual search
	for object_name in missing_objects:
		image = Image.open(input_image).convert('RGB')
		smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
		final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name, confidence_low=0.3, target_bbox=None, smallest_size=smallest_size)
		if not search_successful:
			failed_objects.append(object_name)
		if all_valid_boxes is not None:
			# might exist multiple target instances
			for search_bbox in all_valid_boxes:
				search_final_patch = final_step['bbox']
				search_bbox[0] += search_final_patch[0]
				search_bbox[1] += search_final_patch[1]
				search_result.append({'bbox':search_bbox.tolist(),'name':object_name})
		else:
			search_bbox = final_step['detection_result']
			search_final_patch = final_step['bbox']
			search_bbox[0] += search_final_patch[0]
			search_bbox[1] += search_final_patch[1]
			search_result.append({'bbox':search_bbox.tolist(),'name':object_name})

	# answer based on the searched results
	image = Image.open(input_image).convert('RGB')
	object_names = [_['name'] for _ in search_result]
	bboxs = deepcopy([_['bbox'] for _ in search_result])

	search_result_image = np.array(image).copy()
	for object_name, bbox in zip(object_names, bboxs):
		search_result_image = visualize_bbox(search_result_image, bbox, class_name=object_name, color=(255,0,0))

	if len(object_names) <= 2:
		images_long = [False]
		objects_long = [True]*len(object_names)
	else:
		images_long = [False]
		objects_long = [False]*len(object_names)
	object_crops = []
	for bbox in bboxs:
		object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=1.2)
		object_crops.append(object_crop)
	object_crops = torch.stack(object_crops, 0)
	image, left, top = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
	bbox_list = []
	for bbox in bboxs:
		bbox[0] += left
		bbox[1] += top
		bbox_list.append(bbox)
	bbox_list = [normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
	cur_focus_msg = focus_msg
	for i, (object_name, bbox) in enumerate(zip(object_names, bbox_list)):
		cur_focus_msg = cur_focus_msg + "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]".format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
		if i != len(bbox_list)-1:
			cur_focus_msg = cur_focus_msg+"; "
		else:
			cur_focus_msg = cur_focus_msg +'.'
	if len(failed_objects) > 0:
		if len(object_names) > 0:
			cur_focus_msg = cur_focus_msg[:-1] + "; "
		for i, failed_object in enumerate(failed_objects):
			cur_focus_msg = cur_focus_msg + "{} not existent in the image".format(object_name)
			if i != len(failed_objects)-1:
				cur_focus_msg = cur_focus_msg+"; "
			else:
				cur_focus_msg = cur_focus_msg +'.'
	question_with_focus = cur_focus_msg+"\n"+question
	response = vqa_llm.free_form_inference(image, question_with_focus, object_crops=object_crops, images_long=images_long, objects_long=objects_long, temperature=0.0, max_new_tokens=512)

	search_result_str = ""
	if len(object_names) > 0:
		search_result_str += "Targets located after search: {}.".format(', '.join(object_names))
	if len(failed_objects) > 0:
		search_result_str += "Targets unable to locate after search: {}.".format(', '.join(failed_objects))

	return "Need to conduct visual search to search for: {}.".format(', '.join(missing_objects)), search_result_str, search_result_image, response
	
demo = gr.Interface(
	inference,
	inputs=[
		gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
		gr.Image(type="filepath", label="Input Image"),
	],
	outputs=[
		gr.Textbox(lines=1, placeholder=None, label="Direct Answer"),
		gr.Textbox(lines=1, placeholder=None, label="Visual Search Results"),
		gr.Image(type="pil", label="Visual Search Results"),
		gr.Textbox(lines=1, placeholder=None, label="Final Answer"),
	],
	examples=examples,
	title=title,
	description=description,
	article=article,
	allow_flagging="auto",
)

demo.queue()
demo.launch()