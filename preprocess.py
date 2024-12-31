import datetime
import os
import gc
import sys
import cv2
import glob
# import h5py
import shutil
import imageio
import tifffile
import threading
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tif
from datetime import datetime

Image.MAX_IMAGE_PIXELS = 933120000
join = os.path.join

class CFG:
    comp_dataset_path = "/home/sergeipnev/work/VC/data/"
    tile_size = 224
    stride = 224 // 4
    size = 224
    in_chans = 16
    valid_id = ""


def read_image_mask(fragment_id, CFG=CFG, fold="scroll_5"):
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
        else:
            image = cv2.imread(
                CFG.comp_dataset_path
                + f"{fold}/{fragment_id}/layers/{i:02}.jpg",
                0,
                )

        pad0 = CFG.tile_size - image.shape[0] % CFG.tile_size
        pad1 = CFG.tile_size - image.shape[1] % CFG.tile_size
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = np.clip(image, 0, 200)
        images.append(image)
    images = np.stack(images, axis=2)
    if os.path.exists(
        CFG.comp_dataset_path + f"{fold}/{fragment_id}/{fragment_id}_mask.png"
    ):
        fragment_mask = cv2.imread(
            CFG.comp_dataset_path + f"{fold}/{fragment_id}/{fragment_id}_mask.png",
            0,
            )
    else:
        fragment_mask = cv2.imread(
            CFG.comp_dataset_path + f"{fold}/{fragment_id}/{fragment_id}_flat_mask.png",
            0,
            )
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = cv2.imread(
        CFG.comp_dataset_path + f"{fold}/{fragment_id}/{fragment_id}_inklabels.png",
        0,
    )
    mask = mask / 255
    # mask = None
    return images, fragment_mask, mask


def worker_function(fragment_id, CFG):
    train_images = []
    train_masks = []
    train_xyxys = []

    image, fragment_mask, mask = read_image_mask(fragment_id, CFG=CFG)

    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))

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
                        train_xyxys.append((y1, y2, x1, x2))
                        if fragment_id != CFG.valid_id:
                            train_images.append(image[y1:y2, x1:x2])
                            train_masks.append(mask[y1:y2, x1:x2])
                            assert image[y1:y2, x1:x2].shape == (
                                CFG.size,
                                CFG.size,
                                CFG.in_chans,
                            )

    print("finished reading fragment", fragment_id)
    return train_images, train_masks, train_xyxys


def get_scroll1_dataset(
        fragment_ids=[],
        CFG=None
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

    # print(len(results[0]))
    train_images = []
    train_masks = []
    train_xyxys = []
    print("Aggregating results")
    for r in results:
        if r is None:
            continue
        train_images += r[0]
        train_masks += r[1]
        train_xyxys += r[2]

    return train_images, train_masks, train_xyxys


def preprocess(fragment_ids, ROOT, CFG=CFG):
    start = datetime.now()
    for idx, fragment in enumerate(fragment_ids):
        print(f"Start fragment: {fragment}")
        images_list = []
        masks_list = []
        coords_list = []

        images, masks, xyxys = get_scroll1_dataset(fragment_ids=[fragment], CFG=CFG)
        save_path = ROOT

#         images = np.array(images).transpose((0, 3, 1, 2))
        for i, img in enumerate(images):
            x1, y1, x2, y2 = xyxys[i]
            image_name = f"{fragment}_{x1}_{x2}_{y1}_{y2}.tif"
            mask_name = f"{fragment}_{x1}_{x2}_{y1}_{y2}.png"
            # tif.imwrite(join(save_path, "images", image_name), img, compression="zstd")
            cv2.imwrite(join(save_path, "masks", mask_name), masks[i])

#             np.savez_compressed(join(save_path, image_name), images=img)
            images_list.append(image_name)
            coords_list.append(xyxys[i])
#         with h5py.File(f"{ROOT}/{fragment}.h5", "w") as hf:
#             hf.create_dataset("images", data=images, compression="gzip", compression_opts=9)

        del images, xyxys
        gc.collect()

        finish = datetime.now()
        print(f"{idx}/{len(fragment_ids)}: fragment {fragment} took {finish - start} \n")

        d = {
            "images": images_list,
            "xyxys": coords_list
        }
        df = pd.DataFrame(d)
        df.to_csv(join(ROOT, f"{fragment}.csv"))


def resave_as_jpg(folder_path):
    download_dir_layers = f"{folder_path}/layers"
    for i in glob.glob(os.path.join(download_dir_layers, "*.tif")):
        save_name = i.split(".")[0] + ".jpg"
        img = cv2.imread(i, 0)
        cv2.imwrite(save_name, img)
        os.remove(i)

if __name__ == "__main__":
    ROOT = "/home/sergeipnev/work/VC/data"

    fragment_ids = [i for i in os.listdir(CFG.comp_dataset_path + "scroll_5")]
    preprocess(fragment_ids, ROOT, CFG)




