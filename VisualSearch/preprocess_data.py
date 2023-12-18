import os
import argparse
import json
import numpy as np
from collections import defaultdict

# images exist in annotations but not in image folder.
objv2_ignore_list = [
	os.path.join('patch16', 'objects365_v2_00908726.jpg'),
	os.path.join('patch6', 'objects365_v1_00320532.jpg'),
	os.path.join('patch6', 'objects365_v1_00320534.jpg'),
]


def process_coco(data_dir):
	things_instances = json.load(open(os.path.join(data_dir, 'coco2017', 'annotations', 'instances_train2017.json')))
	stuff_instances = json.load(open(os.path.join(data_dir, 'cocostuff', 'annotations', 'stuff_train2017.json')))

	image_info = dict()
	for image in things_instances['images'] + stuff_instances['images']:
		image_id = image['id']
		if image_id not in image_info:
			image_info[image_id] = image
		else:
			assert image_info[image_id]['file_name'] == image['file_name']

	category_info = dict()
	for category in things_instances['categories'] + stuff_instances['categories']:
		category_info[category['id']] = category['name']

	image2annotations = defaultdict(list)
	for annotation in things_instances['annotations'] + stuff_instances['annotations']:
		image_id = annotation['image_id']
		image_file_name = image_info[image_id]['file_name']
		image2annotations[image_file_name].append({'category_id':annotation['category_id'], 'bbox':annotation['bbox']})

	with open(os.path.join(data_dir, 'cocostuff', 'annotations', 'image2bboxes.json'), 'w') as f:
		json.dump(image2annotations, f)

def process_objects365(data_dir):
	instances = json.load(open(os.path.join(data_dir, 'object365', 'zhiyuan_objv2_train.json')))

	image_info = dict()
	for image in instances['images']:
		image_id = image['id']
		if image_id not in image_info:
			image_info[image_id] = image
		else:
			assert image_info[image_id]['file_name'] == image['file_name']

	category_info = dict()
	for category in instances['categories']:
		category_info[category['id']] = category['name']

	image2annotations = defaultdict(list)
	for annotation in instances['annotations']:
		image_id = annotation['image_id']
		image_file_name = image_info[image_id]['file_name']
		image_file_name = (os.sep).join(image_file_name.split(os.sep)[2:])
		if image_file_name in objv2_ignore_list:
			continue
		image2annotations[image_file_name].append({'category_id':annotation['category_id'], 'bbox':annotation['bbox']})

	with open(os.path.join(data_dir, 'object365', 'image2bboxes.json'), 'w') as f:
		json.dump(image2annotations, f)


def process_goldG(data_dir):
	instances = json.load(open(os.path.join(data_dir, 'MixedGrounding', 'final_mixed_train.json')))
	flickr_instances = json.load(open(os.path.join(data_dir, 'MixedGrounding', 'final_flickr_separateGT_train.json')))
	image_info = []
	for image in instances['images']:
		image_info.append({'file_name':image['file_name'], 'caption':image['caption'], 'data_source':image['data_source'], 'instances':[]})
	for annotation in instances['annotations']:
		image_id = annotation['image_id']
		image_info[image_id]['instances'].append(annotation)

	for image in flickr_instances['images']:
		image_info.append({'file_name':image['file_name'], 'caption':image['caption'], 'data_source':'flickr', 'instances':[]})
	for annotation in flickr_instances['annotations']:
		image_id = annotation['image_id']
		image_info[image_id]['instances'].append(annotation)

	with open(os.path.join(data_dir, 'MixedGrounding', 'goldG_train.json'), 'w') as f:
		json.dump(image_info, f)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="../data")
	args = parser.parse_args()
	process_coco(args.data_dir)
	process_objects365(args.data_dir)
	process_goldG(args.data_dir)