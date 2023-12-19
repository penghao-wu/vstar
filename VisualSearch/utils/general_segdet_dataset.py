import glob
import json
import os
import random
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from transformers import OwlViTProcessor

from VisualSearch.model.llava import conversation as conversation_lib

from VisualSearch.utils.utils import box_xyxy_to_cxcywh, expand2square
from VisualSearch.utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST

parent_dir = os.path.dirname(os.path.abspath(__file__))

def init_objects365(base_dir):
    objects365_classes = []
    with open(os.path.join(parent_dir, "objects365_classes.txt")) as f:
        for line in f.readlines():
            objects365_classes.append(line.strip().split(": ")[-1])
    objects365_classes = np.array(objects365_classes)

    with open(os.path.join(base_dir, "object365", "image2bboxes.json")) as f:
        image2bboxes = json.load(f)

    objects365_images = list(image2bboxes.keys())

    objects365_bboxes = []

    for file_name in objects365_images:
        bboxes = image2bboxes[file_name]
        objects365_bboxes.append(bboxes)

    print("objects365: ", len(objects365_images))

    objects365_images = [os.path.join(base_dir, 'object365/images/train', file_name) for file_name in objects365_images]
    return objects365_classes, objects365_images, objects365_bboxes


def init_cocostuff(base_dir):
    cocostuff_classes = []
    with open(os.path.join(parent_dir, "cocostuff_classes.txt")) as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_dir, "cocostuff", "train2017", "*.png")
    )

    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco2017") for x in cocostuff_labels
    ]

    with open(os.path.join(base_dir, "cocostuff", "annotations", "image2bboxes.json")) as f:
        image2bboxes = json.load(f)

    cocostuff_bboxes = []

    delete_index_list = []
    for i, image_path in enumerate(cocostuff_images):
        file_name = image_path.split(os.sep)[-1]
        if file_name not in image2bboxes:
            delete_index_list.append(i)
            continue
        bboxes = image2bboxes[file_name]
        cocostuff_bboxes.append(bboxes)

    for index in sorted(delete_index_list, reverse=True):
        del cocostuff_labels[index]
        del cocostuff_images[index]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels, cocostuff_bboxes

def init_paco_lvis(base_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis

class SegDetDataset(torch.utils.data.Dataset):
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
        general_segdet_data="objects365||cocostuff||paco_lvis",
        general_segdet_sample_rate=[2,1,1]
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.general_segdet_datas = general_segdet_data.split("||")
        num_images = []
        for ds in self.general_segdet_datas:
            if ds == "cocostuff":
                classes, images, labels, bboxes = eval("init_{}".format(ds))(base_dir)
                self.data2list[ds] = (images, labels, bboxes)
            elif ds == "objects365":
                classes, images, bboxes = eval("init_{}".format(ds))(base_dir)
                self.data2list[ds] = (images, bboxes)
            else:
                classes, images, labels = eval("init_{}".format(ds))(base_dir)
                self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes
            num_images.append(len(images))
        sample_rate = np.array(general_segdet_sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        if "cocostuff" in self.general_segdet_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

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
        ds = np.random.choice(list(range(len(self.general_segdet_datas))), p=self.sample_rate)
        ds = self.general_segdet_datas[ds]

        if ds in ["paco_lvis"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_dir, "coco2017", file_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                expand2square(Image.open(image_path).convert('RGB'), tuple(int(x*255) for x in self.clip_image_processor.image_mean)), return_tensors="pt"
            )["pixel_values"][0]
            original_size = image.shape[:2]
            image = self.transform(images=image, return_tensors="pt")['pixel_values'][0]
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            anns_category2instances = dict()
            for ann in anns:
                category_id = ann['category_id']
                if category_id not in anns_category2instances:
                    anns_category2instances[category_id] = []
                anns_category2instances[category_id].append(ann)
            if len(anns_category2instances) == 0:
                return self.__getitem__(0)
            if len(anns_category2instances) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    list(anns_category2instances.keys()), size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = list(anns_category2instances.keys())
            sampled_classes = []
            for category_id in sampled_anns:
                sampled_cls = class_map[category_id]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                name = name.replace('_', ' ')
                sampled_classes.append(name)

        elif ds in ["cocostuff"]:
            image, labels, bboxes_all = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            bboxes = bboxes_all[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                expand2square(Image.open(image_path).convert('RGB'), tuple(int(x*255) for x in self.clip_image_processor.image_mean)), return_tensors="pt"
            )["pixel_values"][0]
            original_size = image.shape[:2]
            image = self.transform(images=image, return_tensors="pt")['pixel_values'][0]
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes

        elif ds in ['objects365']:
            image, bboxes_all = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            bboxes = bboxes_all[idx]
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocess image for clip
            image_clip = self.clip_image_processor.preprocess(
                expand2square(Image.open(image_path).convert('RGB'), tuple(int(x*255) for x in self.clip_image_processor.image_mean)), return_tensors="pt"
            )["pixel_values"][0]
            original_size = image.shape[:2]
            image = self.transform(images=image, return_tensors="pt")['pixel_values'][0]
            resize = image.shape[:2]
            unique_label = set()
            for bbox_info in bboxes:
                unique_label.add(bbox_info['category_id'])
            unique_label = list(unique_label)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes


        questions = []
        answers = []
        class_ids = []
        bboxes_labels = []
        for i, sampled_cls in enumerate(sampled_classes):
            text = sampled_cls
            if ds in ['objects365']:
                text = random.sample(text.split('/'), 1)[0]
                
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

            if ds in ["paco_lvis", "pascal_part"]:
                category_id = sampled_anns[i]
                cur_bboxes = [instance['bbox'] for instance in anns_category2instances[category_id]]
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
                continue

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)
            if ds in ['objects365']:
                cur_bboxes = [bbox['bbox'] for bbox in bboxes if bbox['category_id'] == class_id]
            else:
                cur_bboxes = [bbox['bbox'] for bbox in bboxes if bbox['category_id']-1 == class_id]
            cur_bboxes = cur_bboxes[:100]
            assert len(cur_bboxes) > 0
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
        bboxes_valid = [1]*len(bboxes_labels)
        masks_valid = [1]*len(bboxes_labels)
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for category_id in sampled_anns:
                try:
                    cur_anns = anns_category2instances[category_id]
                    cur_mask = None
                    for ann in cur_anns:
                        if cur_mask is None:
                            cur_mask = coco_api.annToMask(ann)
                        else:
                            cur_mask = cur_mask | coco_api.annToMask(ann)
                    assert cur_mask is not None
                    masks.append(cur_mask)
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        elif ds in ['objects365']:
            masks = torch.rand(len(bboxes_labels), *original_size)
            label = torch.ones(original_size) * self.ignore_label
            masks_valid = [0]*len(bboxes_labels)
        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)
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
