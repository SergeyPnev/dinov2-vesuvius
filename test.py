import argparse
import math
import os
from functools import partial

import torch
import torch.nn as nn
from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch
# from dinov2.eval.segmentation.models.decode_heads import BNHead

import wandb
import random
import cv2
import PIL.Image
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import segmentation_models_pytorch as smp

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from timesformer_pytorch import TimeSformer

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from warmup_scheduler import GradualWarmupScheduler
PIL.Image.MAX_IMAGE_PIXELS = 933120000


class ConvHead(nn.Module):
    def __init__(self, embedding_size=768, num_classes=1):
        super(ConvHead, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 256, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 64, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        # x = torch.sigmoid(x)
        return x


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
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
    comp_dataset_path = f"/raid/cian/user/sergei.pnev/VC/"
    exp_name = "vit_baseline"
    # ============== model cfg =============
    in_chans = 16  #
    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 4
    train_batch_size = 32  # 32
    valid_batch_size = train_batch_size * 2

    scheduler = "GradualWarmupSchedulerV2"
    epochs = 20  # 30
    warmup_factor = 10
    lr = 1e-4 / warmup_factor
    # ============== fold =============
    valid_id = None
    # ============== fixed =============

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 100

    num_workers = 16

    seed = 0

    # ============== set dataset path =============
    print("set dataset path")

    outputs_path = f"/raid/cian/user/sergei.pnev/results/VC/iter4/{exp_name}/"
    model_dir = outputs_path + f"{comp_name}-models/"
    # ============== augmentation =============
    train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(
            rotate_limit=360, shift_limit=0.15, scale_limit=0.15, p=0.75
        ),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
        A.CoarseDropout(
            max_holes=2,
            max_width=int(size * 0.2),
            max_height=int(size * 0.2),
            mask_fill_value=0,
            p=0.5,
        ),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(5, p=1)])


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    for dir in [cfg.model_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode="train"):
    set_seed(cfg.seed)
    if mode == "train":
        make_dirs(cfg)


cfg_init(CFG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image_mask(fragment_id, start_idx=17, end_idx=43, CFG=CFG):
    # fragment_id_ = fragment_id.split("_")[0]
    images = []
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)
    # idxs = range(start_idx, end_idx)

    for i in idxs:
        if os.path.exists(
                CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif"
        ):
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"train_scrolls/{fragment_id}/layers/{i:02}.tif",
                0,
                )
        elif os.path.exists(
                CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.png"
        ):
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"train_scrolls/{fragment_id}/layers/{i:02}.png",
                0,
                )
        else:
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"train_scrolls/{fragment_id}/layers/{i:02}.jpg",
                0,
                )
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
    if fragment_id in ['20231106155351',
                       '20231022170901',
                       '20231210121321',
                       '20230929220926',
                       '20231031143852',
                       '20230820203112',
                       '20230702185753',
                       '20231221180251',
                       '20231005123336']:
        mask = cv2.imread(
            CFG.comp_dataset_path
            + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels_refined.png",
            0,
            )
    else:
        if any(id_ in fragment_id for id_ in ["20231022170901", "20231022170900"]):
            mask = cv2.imread(
                CFG.comp_dataset_path
                + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.tiff", 0
            )
        else:
            mask = cv2.imread(
                CFG.comp_dataset_path
                + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0
            )
    fragment_mask = cv2.imread(
        CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_mask.png",
        0,
        )
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype("float32")
    mask /= 255
    return images, mask, fragment_mask


def worker_function(fragment_id, CFG):
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    # if not os.path.exists(f"train_scrolls/{fragment_id}"):
    #     fragment_id = fragment_id + "_superseded"
    # print("reading ", fragment_id)
    # try:
    image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)
    # except:
    #     print("aborted reading fragment", fragment_id)
    #     return None
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if not np.any(
                    fragment_mask[a : a + CFG.tile_size, b : b + CFG.tile_size] == 0
            ):
                if (fragment_id == CFG.valid_id) or (
                        not np.all(
                            mask[a : a + CFG.tile_size, b : b + CFG.tile_size] < 0.05
                        )
                ):
                    for yi in range(0, CFG.tile_size, CFG.size):
                        for xi in range(0, CFG.tile_size, CFG.size):
                            y1 = a + yi
                            x1 = b + xi
                            y2 = y1 + CFG.size
                            x2 = x1 + CFG.size
                            if fragment_id != CFG.valid_id:
                                train_images.append(image[y1:y2, x1:x2])
                                train_masks.append(mask[y1:y2, x1:x2, None])
                                assert image[y1:y2, x1:x2].shape == (
                                    CFG.size,
                                    CFG.size,
                                    CFG.in_chans,
                                )
                            if fragment_id == CFG.valid_id:
                                if (y1, y2, x1, x2) not in windows_dict:
                                    valid_images.append(image[y1:y2, x1:x2])
                                    valid_masks.append(mask[y1:y2, x1:x2, None])
                                    valid_xyxys.append([x1, y1, x2, y2])
                                    assert image[y1:y2, x1:x2].shape == (
                                        CFG.size,
                                        CFG.size,
                                        CFG.in_chans,
                                    )
                                    windows_dict[(y1, y2, x1, x2)] = "1"

    print("finished reading fragment", fragment_id)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_train_valid_dataset(
        fragment_ids=[
            "20231210121321",
            "20231022170901",
            "20231106155351",
            "20231005123336",
            "20230820203112",
            "20230826170124",
            "20230702185753",
            "20230522215721",
            "20230531193658",
            "20230903193206",
            "20230902141231",
            "20231007101615",
            "20230929220926",
            "Frag5",
            # "Frag5-right",
            "20231016151000",
            "20231012184423",
            "20231031143850",
        ]
):
    threads = []
    results = [None] * len(fragment_ids)

    # Function to run in each thread
    def thread_target(idx, fragment_id):
        results[idx] = worker_function(fragment_id, CFG)

    # Create and start threads
    for idx, fragment_id in enumerate(fragment_ids):
        thread = threading.Thread(target=thread_target, args=(idx, fragment_id))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    print("Aggregating results")
    for r in results:
        if r is None:
            continue
        train_images += r[0]
        train_masks += r[1]
        valid_images += r[2]
        valid_masks += r[3]
        valid_xyxys += r[4]

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_transforms(data, cfg):
    if data == "train":
        aug = A.Compose(cfg.train_aug_list)
    elif data == "valid":
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels

        self.transform = transform
        self.xyxys = xyxys
        self.rotate = CFG.rotate

    def __len__(self):
        return len(self.images)

    def fourth_augment(self, image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(8, 16)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[
            ..., crop_indices
        ]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy = self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data["image"].unsqueeze(0)
                label = data["mask"]
                label = F.interpolate(
                    label.unsqueeze(0), (128, 128)
                ).squeeze(0)
            return image, label, xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            # 3d rotate
            # image = image.transpose(2, 1, 0)  # (c,w,h)
            # image = self.rotate(image=image)["image"]
            # image = image.transpose(0, 2, 1)  # (c,h,w)
            # image = self.rotate(image=image)["image"]
            # image = image.transpose(0, 2, 1)  # (c,w,h)
            # image = image.transpose(2, 1, 0)  # (h,w,c)
            #
            # image = self.fourth_augment(image)

            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data["image"].unsqueeze(0)
                label = data["mask"]
                label = F.interpolate(
                    label.unsqueeze(0), (128, 128)
                ).squeeze(0)
            return image, label


class ConvHead(nn.Module):
    def __init__(self, embedding_size=768, num_classes=1):
        super(ConvHead, self).__init__()
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_size, 256, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 64, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, num_classes, (3, 3), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.segmentation_conv(x)
        # x = torch.sigmoid(x)
        return x


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

        # self.backbone = TimeSformer(
        #     dim=512,
        #     image_size=64,
        #     patch_size=16,
        #     num_frames=26,
        #     num_classes=16,
        #     channels=1,
        #     depth=8,
        #     heads=6,
        #     dim_head=64,
        #     attn_dropout=0.1,
        #     ff_dropout=0.1,
        # )
        self.head = ConvHead()

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward_backbone(self, x, B):
        x = backbone.forward_features(x)
        x = x['x_norm_patchtokens']
        x = x.permute(0, 2, 1)
        x = x.reshape((B, 768, 16, 16))
        return x

    def forward(self, x):
        B = x.shape[0]
        x = self.forward_backbone(x, B)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log(
            "train/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True
        )
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x, y, xyxys = batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to("cpu")
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += (
                F.interpolate(
                    y_preds[i].unsqueeze(0).float(), size=(224, 224), mode="bilinear"
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )
            self.mask_count[y1:y2, x1:x2] += np.ones(
                (self.hparams.size, self.hparams.size)
            )

        self.log(
            "val/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True
        )
        return {"loss": loss1}

    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(
            self.mask_pred,
            self.mask_count,
            out=np.zeros_like(self.mask_pred),
            where=self.mask_count != 0,
        )
        wandb_logger.log_image(
            key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"]
        )

        # reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)

        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer], [scheduler]


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6
    )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine
    )

    return scheduler


def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

args = get_args_parser(add_help=True).parse_args()
cfg = setup(args)
torch.set_float32_matmul_precision("medium")
# add all of the validation segments into the array to run multiple validation folds
fragments = ["20231210121321"]
# fragments = ["Frag5-right"]

for fid in fragments:
    CFG.valid_id = fid
    fragment_id = CFG.valid_id
    run_slug = (
        f"training_scrolls_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels"
    )

    valid_mask_gt = cv2.imread(
        CFG.comp_dataset_path
        + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png",
        0,
        )

    pred_shape = valid_mask_gt.shape
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = (
        get_train_valid_dataset()
    )
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        # transform=get_transforms(data="train", cfg=CFG),
        transform=get_transforms(data="valid", cfg=CFG),
    )
    valid_dataset = CustomDataset(
        valid_images,
        CFG,
        xyxys=valid_xyxys,
        labels=valid_masks,
        transform=get_transforms(data="valid", cfg=CFG),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    wandb_logger = WandbLogger(
        project="vesivus", name=run_slug + f"vit_baseline"
    )

    model = RegressionPLModel(cfg=CFG, pred_shape=pred_shape, size=CFG.size)
    m = SSLMetaArch(cfg)
    # # model.prepare_for_distributed_training()
    checkpoint = torch.load("/raid/cian/user/sergei.pnev/results/VC/dinov2/baseline_180924/model_final.rank_0.pth")
    weights = checkpoint["model"]
    m.load_state_dict(weights)
    # backbone = model.teacher.backbone.module
    backbone = m.teacher.backbone.cuda()
    model.backbone = backbone

    wandb_logger.watch(model, log="all", log_freq=100)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=-1,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        # strategy="ddp_find_unused_parameters_true",
        callbacks=[
            ModelCheckpoint(
                filename=f"vit_baseline{fid}_fr" + "{epoch}",
                dirpath=CFG.model_dir,
                monitor="train/total_loss",
                mode="min",
                save_top_k=CFG.epochs,
            ),
        ],
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    torch.save(model.state_dict(), CFG.outputs_path + "/vit_baseline.pth")

    wandb.finish()

# if __name__ == "__main__":
#     args = get_args_parser(add_help=True).parse_args()
#     main(args)