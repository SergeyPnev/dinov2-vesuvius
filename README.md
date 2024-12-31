# DINOv2 for Vesuvius Ink Detection
The main hypothesis was to pre-train DINO-v2 on several channel images, because original backbone uses only 3, but we
have much more (here we use 16). I pre-trained model during 1000000 steps with batch size 82 in self-supervised mode. For pre-training I used data from scrolls 1, 2, 3, 4, 5.
The model converged on step 93749. For segmentation fine-tune I used frozen ViT encoder and simple Convolution Head for decoding. 

- DINOv2 pre-training code in file [dinov2_train.py](./dinov2/train/train_vc.py) 
- And ViT finetune code in [finetune_vit.py](finetune_vit.py)
- Dataset code is in [vc_dataset.py](./dinov2/data/datasets/vc_dataset.py)

## Predictions
You can find predictions and pre-trained DINOv2 and ViT models at [google drive](https://drive.google.com/drive/folders/1XwW-Cu09d6JAZR2q7eZPThzPLHnF8liP?usp=sharing)
In images folder predictions from baseline_190924_model_0093749_s1 means that pre-trained DINOv2 model was fine-tuned on segments and labels from scroll 1. Same for s5.
Predictions on s4 were made with model fine-tuned on scrolls 1 and 5.

## Installation
The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

```shell
pip install -r requirements.txt -r requirements-extras.txt
```

## Data preparation
For pre-training you can choose 2 options: 
1. Use dataset class in [Vesuvius-Grandprize-Winner](https://github.com/younader/Vesuvius-Grandprize-Winner/tree/main) style. In order to use this you should just uncomment bottom VC dataset class in [dataset](./dinov2/data/datasets/vc_dataset.py) and specify which fragment you want to use for pre-train.  
2. Preprocess data with [preprocess](./preprocess.py) file. It crops and resave specified segments in images with shape CxHxW = 16x224x224.

## Training

### Fast setup: training DINOv2 ViT-L/16 on Vesuvius Segments
1. Download backbone weights from [weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) and put it in [~/.cache/torch/hub/checkpoints](~/.cache/torch/hub/checkpoints)
2. Run following code:
```shell
CUDA_VISIBLE_DEVICES=gpu_id PYTHONPATH=. python dinov2/run/train/train_vc.py \
    --config-file dinov2/configs/train/vitl16_short.yaml 
```

## Fine-tune
In order to fine-tune you need to change CFG in [finetune_vit.py](finetune_vit.py):
1. Specify valid_id 
2. root
3. comp_dataset_path
4. outputs_path

Also you need to change paths to dino baseline:
```shell
checkpoint = torch.load("~/dinov2/baseline_190924/model_0093749.rank_0.pth")
```


Fine-tuning command:
```shell
CUDA_VISIBLE_DEVICES=gpu_id python finetune_vit.py
```

## Inference
Same changes as in fine-tune and you should additionally choose segments and 
change path to saved ViT backbone in CFG:
```shell
vit_checkpoint_path = f"{model_dir}/baseline_190924_model_0093749_s1s5.pth"
```

And run:
```shell
CUDA_VISIBLE_DEVICES=gpu_id python inference.py
```

## License

DINOv2 code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing DINOv2

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

```
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```
