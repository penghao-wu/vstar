# *V*\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs

### [Paper]() | [Project Page]()


![Teaser](assets/teaser.png)


## Contents:
1. [Getting Started](#start)
4. [Demo](#demo)
5. [Benchmark](#benchmark)
6. [Evaluation](#evaluation)
7. [Training](#training)
10. [License](#license)
11. [Citation](#citation)

## Getting Started <a name="start"></a>

### Installation
```
conda create -n vstar python=3.10 -y
conda activate vstar
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Pre-trained Model

The VQA LLM can be downloaded [here](https://huggingface.co/craigwu/seal_vqa_7b).  
The visual search model can be downloaded [here](https://huggingface.co/craigwu/seal_vsm_7b).

### Training Dataset

The alignment stage of the VQA LLM uses the 558K subset of the LAION-CC-SBU dataset used by LLaVA which can be downloaded [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

The instruction tuning stage requires several instruction tuning subsets which can be found [here](https://huggingface.co/datasets/craigwu/seal_vqa_data).

## Demo <a name="demo"></a>
You can launch a local Gradio demo after the installation by running `python app.py`. Note that the pre-trained model weights will be automatically downloaded if you have not downloaded them before.

You are expected to see the web page below:

## Benchmark <a name="benchmark"></a>
Our *V*\*Bench is available [here](https://huggingface.co/datasets/craigwu/vstar_bench). 
The benchmark contains folders for different subtasks. Within each folder is a list of image files and annotation JSON files. The image and annotations files are paired according to the filename. The format of the annotation files is:
```javascript
{
  "target_object": [] // A list of target object names
  ,
  "bbox": [] // A list of target object coordinates in <x,y,w,h>
  ,
  "question": "",
  "options": [] // A list of options, the first one is the correct option by default
}
```
## Evaluation <a name="evaluation"></a>

## Training <a name="training"></a>

## License <a name="license"></a>

## Citation <a name="citation"></a>
