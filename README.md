# FIQ: Fundamental Question Generation with the Integration of Question Embeddings for Video Question Answering
This is the official repository of the paper FIQ: Fundamental Question Generation with the Integration of Question Embeddings for Video Question Answering, which was published in IEEE International Conference on Systems, Man, and Cybernetics (SMC) 2025

[[paper]](https://arxiv.org/abs/2507.12816)
![figure](fiq_figure.png)

## Dataset Preparation
We use the [SUTD-TrafficQA](https://sutdcv.github.io/SUTD-TrafficQA/#/) dataset as the main source.  
Please download the dataset from the link above and place the extracted folder inside the `/data` directory.  
After setup, your project structure should look like this:

```
FIQ/
├── data/
│ └── [SUTD-TrafficQA files here]
├── model/
├── ...
```

Additionally, download the feature files, model checkpoints, and the Q&A JSON dataset from [this Google Drive link](https://drive.google.com/drive/folders/1u4bk0CUn17Y67lxlVML9EQst78mpTT6Q).  
Once downloaded, create a subdirectory named `/sutd-traffic/` inside the `/data` folder, and place **all downloaded files** into that directory:
```
FIQ/
├── data/
│ ├── sutd-traffic/
│ │ ├── sutd-traffic_appearance_feat_clip_image.h5
│ │ ├── output_file_train.jsonl
│ │ ├── output_file_test.jsonl
│ │ └── final_SUTD_qa_without_blank_allU.jsonl
├── model/
├── ...
```

## Installation
We use the docker environment for this experiment. After the preparation of the dataset, please run the following command below:

```
docker run -itd \
  -v ./FIQ:/workspace/FIQ \
  -v ./compressed_videos/:/workspace/FIQ/data/raw_videos \
  -v ./annotations/archived/R3_all.jsonl:/workspace/FIQ/data/annotation_file/R3_all.jsonl \
  --gpus all \
  --name tem-adapter-remove\
  pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
```
After running the command above, please run commands below inside a docker container.

```
pip install ffmpeg scikit-video ftfy regex tqdm timm jsonlines decord line_profiler einops wandb
apt-get update
apt-get install git
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```
After that, please replace the clip.py code in the path "/opt/conda/lib/python3.11/site-packages/clip/clip.py" with this [code](clip_code/clip.py). 

## Preprocessing
Please set an available cuda device number.
```
CUDA_VISIBLE_DEVICES=0 python preprocess/preprocess_features.py --dataset sutd-traffic --model clip_image 
```

## Q&A Dataset Generation
We use [VideChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) to extract the video description. Based on this, we design the prompt to generate Q&A pairs for GPT-4o-mini by following [paper](https://arxiv.org/pdf/2205.01883). Please run the command below to generate Q&A pairs:
```
python gpt_QG/qg_sutd.py
```
## Training
To train the model, please run the following command:
```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/sutd-traffic_transition.yml
```

## Evaluation
To evaluate the model, please execute the following command structure:
```
CUDA_VISIBLE_DEVICES=0 python validate.py --cfg configs/sutd-traffic_transition.yml
```
## Citation  
if you find our work is helpful, please consider cite this paper:
```
@inproceedings{fiq,
  title={FIQ: Fundamental Question Generation with the Integration of Question Embeddings for Video Question Answering},
  author={{Juyoung Oh and Ho-Joong Kim and Seong-Whan Lee},
  booktitle={arXiv preprint arXiv:2507.12816},
  year={2025}
}
```
## Acknowledgement
Our methods are developed based on [Tem-adapter](https://github.com/XLiu443/Tem-adapter). Thank authors for releasing the code and the pretrained models.
