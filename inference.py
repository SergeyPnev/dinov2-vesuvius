import os
import gc
import json
import wandb
import argparse
from datetime import datetime

import cv2
import math
import random
import PIL.Image
import threading
import numpy as np
import pandas as pd
import tifffile as tiff
import scipy.stats as st
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from timesformer_pytorch import TimeSformer

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
# from warmup_scheduler import GradualWarmupScheduler

from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch

join = os.path.join


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="dinov2/configs/train/vitl16_short.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


class CFG:
    # ============== comp exp name =============
    comp_name = "vesuvius"
    comp_dir_path = "./"
    comp_folder_name = "./"
    exp_name = "vit_dinov2_s1_161224"
    root = "/raid/cian/user/sergei.pnev/VC/VC_224/"
    comp_dataset_path = f"/raid/cian/user/sergei.pnev/VC/"
    segments_json_path = "segments.json"

    # ============== model cfg =============
    in_chans = 16  #

    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 4
    pred_shape = (224, 224)
    train_batch_size = 128  # 32
    valid_batch_size = train_batch_size * 2

    num_workers = 16

    # ============== set dataset path =============
    save_dir = f"./{exp_name}/"
    model_dir = "/raid/cian/user/sergei.pnev/results/VC/iter7/vit_dinov2_s1_161224"
    dino_checkpoint_path = f"{model_dir}/model_0093749.rank_0.pth"

    # weights for ViT pretrained on segments from Scroll 1 and Scroll 5
    vit_checkpoint_path = f"{model_dir}/baseline_190924_model_0093749_s1s5.pth"

#     # weights for ViT pretrained on segments from Scroll 1
#     vit_checkpoint_path = f"{model_dir}/baseline_190924_model_0093749_s1.pth"
#
#     # weights for ViT pretrained on segments from Scroll 5
#     vit_checkpoint_path = f"{model_dir}/baseline_190924_model_0093749_s5.pth"

    # ============== augmentation =============
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(5, p=1)])


def read_image_mask(fragment_id, start_idx=17, end_idx=43, CFG=CFG, fold="train_scrolls"):
    images = []
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in idxs:
        if os.path.exists(
            CFG.comp_dataset_path + f"{fold}/{fragment_id}/layers/{i:02}.tif"
        ):
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"{fold}/{fragment_id}/layers/{i:02}.tif",
                0,
            )
        elif os.path.exists(
                CFG.comp_dataset_path + f"{fold}/{fragment_id}/layers/{i:02}.png"
        ):
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"{fold}/{fragment_id}/layers/{i:02}.png",
                0,
            )
        else:
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"{fold}/{fragment_id}/layers/{i:02}.jpg",
                0,
            )
        print()
        pad0 = CFG.tile_size - image.shape[0] % CFG.tile_size
        pad1 = CFG.tile_size - image.shape[1] % CFG.tile_size
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 200)
        images.append(image)
    images = np.stack(images, axis=2)
    if any(
        id_ in fragment_id
        for id_ in [
            "20230701020044",
            "verso",
            "20230901184804",
            "20230901234823",
            "20230531193658",
            "20231007101615",
            "20231005123333",
            "20231011144857",
            "20230522215721",
            "20230919113918",
            "20230625171244",
            "20231022170900",
            "20231012173610",
            "20231016151000",
        ]
    ):
        images = images[:, :, ::-1]
    fragment_mask = cv2.imread(
        CFG.comp_dataset_path + f"{fold}/{fragment_id}/{fragment_id}_mask.png",
        0,
    )
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    return images, fragment_mask, (pad0, pad1)


def get_img_splits(fragment_id, CFG=CFG):
    images = []
    xyxys = []

    image, fragment_mask, (pad0, pad1) = read_image_mask(fragment_id, CFG=CFG)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])

    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys), (image.shape[0],image.shape[1]), fragment_mask, (pad0, pad1)


class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image']
        return image, xy


class Block(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, hidden_dim, kernel_size=(3, 3),  padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3),  padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

# simple conv Decoder
class ConvHead(nn.Module):
    def __init__(self, embedding_size=768, hidden_dim=512, num_classes=1):
        super(ConvHead, self).__init__()
        print(hidden_dim / 2)
        self.block_1 = Block(embedding_size, hidden_dim)
        self.block_2 = Block(hidden_dim, 256)
        self.block_3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 64, kernel_size=(3, 3),  padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size=(3, 3),  padding=(1, 1)),
        )
        self.up = nn.Upsample(scale_factor=2),

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        return x_3


class RegressionPLModel(pl.LightningModule):
    def __init__(self, cfg, pred_shape, size=256, with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode="binary")
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(
            x, y
        ) + 0.5 * self.loss_func2(x, y)

        self.backbone = None
        self.head = ConvHead()

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward_backbone(self, x, B, out_indices=[1, 2, 3]):
        with torch.no_grad():
            x = self.backbone.forward_features(x, out_indices=out_indices)["x_norm_patchtokens"]
            x = x.permute(0, 2, 1)
            x = x.reshape((B, 768, 16, 16))
        return x

    def forward(self, x, out_indices=[1, 2, 3]):
        B = x.shape[0]
        x = self.forward_backbone(x, B, out_indices=[])
        x = self.head(x)
        return x

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         outputs = self.forward(x, out_indices=[])
#         loss1 = self.loss_func(outputs, y)
#         if torch.isnan(loss1):
#             print("Loss nan encountered")
#         self.log(
#             "train/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True
#         )
#         return {"loss": loss1}


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel=gkern(CFG.size,1)
    kernel=kernel/kernel.max()

    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with autocast():
                y_preds = model(images)

        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(
                F.interpolate(
                    y_preds[i].unsqueeze(0).float(),
                    (224, 224),
                    mode='bilinear').squeeze(0).squeeze(0).numpy(),
                kernel
            )
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    return mask_pred

def build_model_and_load_checkpoint(dino_cfg, CFG):
    # DINO part
    model = RegressionPLModel(cfg=CFG, pred_shape=CFG.pred_shape, size=CFG.size)
    m = SSLMetaArch(dino_cfg)
    checkpoint = torch.load(CFG.dino_checkpoint_path)
    weights = checkpoint["model"]
    m.load_state_dict(weights)
    backbone = m.teacher.backbone.cuda()
    model.backbone = backbone

    # ViT part
    checkpoint = torch.load(CFG.vit_checkpoint_path, map_location="cpu")
    model = model.cpu()
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    return model


def main(CFG, segments):
    # load dino congig
    args = get_args_parser(add_help=True).parse_args()
    dino_cfg = setup(args)

    # setup save_dir
    save_dir = CFG.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # define pl model -> build DINO model -> load DINO backbone to pl model ->
    # -> load ViT weights -> update ViT weights in pl model
    # kinda tricky, should change it
    model = build_model_and_load_checkpoint(dino_cfg, CFG)

    # load segment -> predict it -> save logits and probs -> empty cache
    for fragment_id in segments:
        start = datetime.now()
        test_loader, test_xyxz, test_shape, fragment_mask, (pad0, pad1) = get_img_splits(fragment_id, CFG=CFG)

        device = "cuda"
        mask_pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
        np.nan_to_num(mask_pred, nan=0)
        probs = F.sigmoid(torch.tensor(mask_pred)).numpy()

        cv2.imwrite(f"{save_dir}/{fragment_id}_mask.png", mask_pred * 255)
        cv2.imwrite(f"{save_dir}/{fragment_id}_prob.png", probs * 255)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Fragment {fragment_id} is processed during {datetime.now() - start}\n")


if __name__ == "__main__":
    scroll = "SCROLL_1"
    with open(CFG.segments_json_path, "r") as f:
        segments = json.load(f)[scroll]

    main(CFG, segments)