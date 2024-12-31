import os
import cv2
import random
import threading
import numpy as np
import tifffile as tiff
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
join = os.path.join

class CFG:
    comp_dataset_path = "/raid/cian/user/sergei.pnev/data/VC_HRDA/VC_224/"
#     comp_dataset_path = "/data/sergei.pnev/hidden_layer/VC_224/img/"
    size = 224
    tile_size = 224
    stride = tile_size // 2
    in_chans = 16
    valid_id = ""


def read_image_mask(fragment_id, start_idx=24, end_idx=40, CFG=CFG):
    # fragment_id_ = fragment_id.split("_")[0]
    images = []
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in idxs:
        if os.path.exists(
                CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif"
        ):
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"train_scrolls/{fragment_id}/layers/{i:02}.tif",
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

    fragment_mask = cv2.imread(
        CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_mask.png",
        0,
        )
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    return images, None, fragment_mask


def worker_function(fragment_id, CFG):
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if not np.any(
                    fragment_mask[a : a + CFG.tile_size, b : b + CFG.tile_size] == 0
            ):
                for yi in range(0, CFG.tile_size, CFG.size):
                    for xi in range(0, CFG.tile_size, CFG.size):
                        y1 = a + yi
                        x1 = b + xi
                        y2 = y1 + CFG.size
                        x2 = x1 + CFG.size
                        if fragment_id != CFG.valid_id:
                            train_images.append(image[y1:y2, x1:x2])
                            assert image[y1:y2, x1:x2].shape == (
                                CFG.size,
                                CFG.size,
                                CFG.in_chans,
                            )

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


class VCDataset(Dataset):
    def __init__(self, cfg=CFG, transform=None):
        self.cfg = cfg
        self.in_channels = cfg.in_chans

        self.paths = glob(join(cfg.comp_dataset_path, "*.tif"))
        self.transform = transform
        self.rotate = A.Compose([A.Rotate(8, p=1)])

    def __len__(self):
        return len(self.paths)

    def phase_aug(self, img):
        img_fft = np.fft.fft2(img, axes=(0, 1))

        img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)

        img_phase = np.array([[[50000] * self.in_channels]]) * (np.e ** (1j * img_pha))
        img_phase = np.real(np.fft.ifft2(img_phase, axes=(0, 1)))
        # img_phase = np.clip(img_phase, 0, 255)
        img_phase = np.uint8(np.clip(img_phase, 0, 255))
        return img_phase

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
        # image = self.phase_aug(self.images[idx])
        image = tiff.imread(self.paths[idx])
#         print("image: ", image.shape)
        # 3d rotate
        image = image.transpose(2, 1, 0)  # (c,w,h)
        image = self.rotate(image=image)["image"]
        image = image.transpose(0, 2, 1)  # (c,h,w)
        image = self.rotate(image=image)["image"]
        image = image.transpose(0, 2, 1)  # (c,w,h)
        image = image.transpose(2, 1, 0)  # (h,w,c)
        image = self.fourth_augment(image)

#         print("image after fourth aug: ", image.shape)
        if self.transform:
            data = self.transform(image=image)
        return data, 0


class VCDataset(Dataset):
    def __init__(self, cfg=CFG, transform=None, target_transform=None):
        self.cfg = cfg
        self.in_channels = 16

        self.paths = glob(join(cfg.comp_dataset_path, "*.tif"))
        self.transform = transform
        self.rotate = A.Compose([A.Rotate(8, p=1)])

    def __len__(self):
        return len(self.paths)

    def phase_aug(self, img):
        img_fft = np.fft.fft2(img, axes=(0, 1))

        img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)

        img_phase = np.array([[[50000] * self.in_channels]]) * (np.e ** (1j * img_pha))
        img_phase = np.real(np.fft.ifft2(img_phase, axes=(0, 1)))
        # img_phase = np.clip(img_phase, 0, 255)
        img_phase = np.uint8(np.clip(img_phase, 0, 255))
        return img_phase

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
        # image = self.phase_aug(self.images[idx])
        image = tiff.imread(self.paths[idx])
#         print("image: ", image.shape)
        # 3d rotate
        image = image.transpose(2, 1, 0)  # (c,w,h)
        image = self.rotate(image=image)["image"]
        image = image.transpose(0, 2, 1)  # (c,h,w)
        image = self.rotate(image=image)["image"]
        image = image.transpose(0, 2, 1)  # (c,w,h)
        image = image.transpose(2, 1, 0)  # (h,w,c)
        image = self.fourth_augment(image)

#         print("image after fourth aug: ", image.shape)
        if self.transform:
            data = self.transform(image=image)
        return data, 0


# Use this code if you want to prepare images inside Dataset class

# class VCDataset(Dataset):
#     def __init__(self, cfg=CFG, transform=None, target_transform=None):
#         self.cfg = cfg
#         self.in_channels = 16
#
#         # Specify segments here
#         self.images, _, _, _, _ = get_train_valid_dataset(
#             fragment_ids =
#                 [
# # val scrolls
#                     "20231121133215",
#                     "20240304141531",
#                     "20240304144031",
#                     "20240304161941",
# # train scrolls
#                     "20231210121321",
#                     "20231022170901",
#                     "20231106155351",
#                     "20231005123336",
#                     "20230820203112",
#                     "20230826170124",
#                     "20230702185753",
#                     "20230522215721",
#                     "20230531193658",
#                     "20230903193206",
#                     "20230902141231",
#                     "20231007101615",
#                     "20230929220926",
#                     "20231016151000",
#                     "20231012184423",
#                     "20231031143850",
#                 ]
#         )
#         self.transform = transform
#         self.rotate = 0
#
#     def __len__(self):
#         return len(self.images)
#
#     def phase_aug(self, img):
#         img_fft = np.fft.fft2(img, axes=(0, 1))
#
#         img_abs, img_pha = np.abs(img_fft), np.angle(img_fft)
#
#         img_phase = np.array([[[50000] * self.in_channels]]) * (np.e ** (1j * img_pha))
#         img_phase = np.real(np.fft.ifft2(img_phase, axes=(0, 1)))
#         # img_phase = np.clip(img_phase, 0, 255)
#         img_phase = np.uint8(np.clip(img_phase, 0, 255))
#         return img_phase
#
#     def fourth_augment(self, image):
#         image_tmp = np.zeros_like(image)
#         cropping_num = random.randint(8, 16)
#
#         start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
#         crop_indices = np.arange(start_idx, start_idx + cropping_num)
#
#         start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)
#
#         tmp = np.arange(start_paste_idx, cropping_num)
#         np.random.shuffle(tmp)
#
#         cutout_idx = random.randint(0, 2)
#         temporal_random_cutout_idx = tmp[:cutout_idx]
#
#         image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[
#             ..., crop_indices
#         ]
#
#         if random.random() > 0.4:
#             image_tmp[..., temporal_random_cutout_idx] = 0
#         image = image_tmp
#         return image
#
#     def __getitem__(self, idx):
#         # image = self.phase_aug(self.images[idx])
#         image = self.images[idx]
#         # 3d rotate
#         # image = image.transpose(2, 1, 0)  # (c,w,h)
#         # image = self.rotate(image=image)["image"]
#         # image = image.transpose(0, 2, 1)  # (c,h,w)
#         # image = self.rotate(image=image)["image"]
#         # image = image.transpose(0, 2, 1)  # (c,w,h)
#         # image = image.transpose(2, 1, 0)  # (h,w,c)
#
#         # image = self.fourth_augment(image)
#
#         if self.transform:
#             data = self.transform(image=image)
#         return data, 0
