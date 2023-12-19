import os
import random
import json
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from VisualSearch.model.llava import conversation as conversation_lib


from transformers import OwlViTProcessor

from VisualSearch.utils.utils import box_xyxy_to_cxcywh, expand2square
from VisualSearch.utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST


class MixedGroundingDataset(torch.utils.data.Dataset):
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
    ):
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        with open(os.path.join(base_dir, 'MixedGrounding', 'goldG_train.json')) as f:
            self.images = json.load(f)

    def __len__(self):
        return self.samples_per_epoch

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

        idx = random.randint(0, len(self.images) - 1)
        image_info = self.images[idx]
        image_data_source = image_info['data_source']
        file_name = image_info["file_name"]
        assert image_data_source in ['coco', 'vg', 'flickr']
        if image_data_source == 'coco':
            image_path = os.path.join(self.base_dir, 'coco2014/train2014', file_name)
        elif image_data_source == 'vg':
            image_path = os.path.join(self.base_dir, 'MixedGrounding/GQA/images', file_name)
        else:
            image_path = os.path.join(self.base_dir, 'MixedGrounding/flickr30k-images', file_name)
        caption = image_info['caption']
        instances = image_info['instances']
        if len(instances) == 0:
            return self.__getitem__(0)

        if len(instances) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(instances))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(instances)))

        sampled_classes = sampled_inds
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
                expand2square(Image.open(image_path).convert('RGB'), tuple(int(x*255) for x in self.clip_image_processor.image_mean)), return_tensors="pt")["pixel_values"][0]
        original_size = image.shape[:2]
        image = self.transform(images=image, return_tensors="pt")['pixel_values'][0]
        resize = image.shape[:2]

        questions = []
        answers = []
        bboxes_labels = []
        for sample_ind in sampled_inds:
            text = []
            tokens_positive = instances[sample_ind]['tokens_positive']
            for token in tokens_positive:
                text.append(caption[token[0]:token[1]])
            text = " ".join(text)
            text = text.strip()
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

            cur_bboxes = [instances[sample_ind]['bbox']]
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

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1
            
        bboxes_valid = [1]*len(bboxes_labels)
        masks_valid = [0]*len(bboxes_labels)
        masks = torch.rand(len(bboxes_labels), *original_size)
        label = torch.ones(original_size) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            bboxes_labels,
            bboxes_valid,
            masks_valid,
            resize,
            questions,
            sampled_classes,
        )
