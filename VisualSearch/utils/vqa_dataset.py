import json
import os
import random
import copy
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from transformers import OwlViTProcessor


from VisualSearch.model.llava import conversation as conversation_lib
from VisualSearch.utils.utils import box_xyxy_to_cxcywh, expand2square
from VisualSearch.utils.utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "[LOC]"+"\n" + sentence["value"]
            # sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source

class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="possible_locations_conv_86k||llava_instruct_150k",
        vqa_sample_rate=[2,1],
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        DATA_DIR = os.path.join(base_image_dir, "vsm_vqa_data")
        self.vqa_image_root = os.path.join(base_image_dir, "coco2017/train2017")
        vqa_datas = vqa_data.split("||")
        self.vqa_datas = []
        for data in vqa_datas:
            with open(os.path.join(DATA_DIR, "{}.json".format(data))) as f:
                data = json.load(f)
                self.vqa_datas.append(data)
        sample_rate = np.array(vqa_sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

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
        ds = np.random.choice(list(range(len(self.vqa_datas))), p=self.sample_rate)
        ds = self.vqa_datas[ds]
        idx = random.randint(0, len(ds) - 1)
        item = ds[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(
                expand2square(Image.open(image_path).convert('RGB'), tuple(int(x*255) for x in self.clip_image_processor.image_mean)), return_tensors="pt")["pixel_values"][0]

        image = self.transform(images=image, return_tensors="pt")['pixel_values'][0]
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            copy.deepcopy(source),
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        masks = torch.rand(1, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
        bboxes_labels = [torch.tensor([[0.5,0.5,1.0,1.0]])]
        bboxes_valid = [0]
        masks_valid = [0]

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
