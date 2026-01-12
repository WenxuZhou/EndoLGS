# EndoLGS
This repository contains the official implementation for the paper "Open-Vocabulary Endoscopic Scene Understanding via 4D Language Gaussian Splatting". Below are the detailed setup and usage instructions.

## Environment Setup
We recommend using `conda` to create a dedicated environment:
```bash
# Create conda environment
conda create -n endolgs python=3.10
conda activate endolgs
```

```bash
# Install PyTorch with CUDA 11.8
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Install submodules
pip install -e submodules/4d-langsplat-rasterization
pip install -e submodules/simple-knn
```

## Training
Extract cross-modal features with CLIP
```bash
python generate_clip_features.py \
    --dataset_path <dataset_path> \
    --precompute_seg <path_to_semantic_masks>
```

Train autoencoder
```bash
python autoencoder/train.py \
    --dataset_path <dataset_path> \
    --model_name <autoencoder_save_path>
```

Obtain compressed features
```bash
python autoencoder/test.py \
    --dataset_path <dataset_path> \
    --model_name <autoencoder_save_path>
```

To train the model, run the following command
```bash
python train.py \
    -s <dataset_path> \
    --expname <experiment_name> \
    --configs <parameter_configs>
```

## Others
Visualize rendering results
```bash
python render.py \
    --model_path <reconstruction_result_path> \
    --skip_train \
    --skip_video \
    --configs <parameter_configs>
```

Test text query results
```bash
python eval_lang.py \
    --model_path <reconstruction_result_path> \
    --model_name <output_filename> \
    --ae_ckpt_path <autoencoder_weights_path>
```

Evaluate visual quality
```bash
python metrics.py \
    --model_path <reconstruction_result_path>
```

Convert images to video (use `ffmpeg` to convert image sequences to video)
```bash
ffmpeg -framerate 24 -i <path_to_render_result>/%05d.png -c:v libx264 -pix_fmt yuv420p <output_video_path>
```

Visualize segmentation masks
```bash
python seg_vis.py \
    --mask_dir <path_to_mask_dir> \
    --image_dir <path_to_image_dir>
```
