## Training Data

The training data is comprised of the following parts:

1. General segmentation and detection datasets: [COCO-2017]([http://images.cocodataset.org/zips/train2017.zip](https://cocodataset.org/#download)), [COCO-Stuff]([http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip](https://github.com/nightrome/cocostuff)), [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup), and [Objects364-V2](https://www.objects365.org/overview.html)

2. Referring segmentation datasets: [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 

3. Mixed grounding datasets: Follow the instructions [here](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md) to prepare the GQA images, Flickr30K images, and pre-processed annotations.

4. Visual Question Answering dataset: The llava-80k instruction data can be downloaded [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_80k.json) and our possible location QA data can be downloaded [here](https://huggingface.co/datasets/craigwu/vsm_vqa_data). The image data needed for this part is COCO-2017.

After downloading all the data listed above, they should be organized as follows
```
├── dataset
│
│   ├── coco2017
|   |   ├── annotations
│   │   └── train2017
|   |
│   ├── cocostuff
|   |   ├── annotations
│   │   └── train2017
|   |
│   ├── vsm_vqa_data
│   │   ├── llava_instruct_80k.json
│   │   └── possible_locations_conv_86k.json
|   |
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
|   |
|   ├── MixedGrounding
|   |   ├── flickr30k-images
|   |   ├── GQA
|   |   |    └── images
|   |   ├── final_flickr_separateGT_train.json
|   |   └── final_mixed_train.json
|   |
|   ├── object365
|   |       ├── images
|   |       └── zhiyuan_objv2_train.json
│   └── vlpart
│       └──  paco
               └── annotations
```

## Pre-trained weights

Our visual search model is based on `LLaVA-Lightning-7B-v1-1` model merged from `liuhaotian/LLaVA-Lightning-7B-delta-v1-1`. Please follow the instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to download and merge the weight.

## Preprocess Data

After downloading and organizing data as described above, enter the `VisualSearch` folder and run

```
python preprocess_data.py --data_dir DATASET_FOLDER
```
## Training
To launch training, run
```
python train.py --version PATH_TO_LLAVA_WEIGHT --dataset_dir DATASET_FOLDER
```

After training, convert the deepspeed weights to full model weight:
```
cd ./runs/vsm/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

Merge the weight with the base model under the `VisualSearch` folder:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLAVA_WEIGHT" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```
