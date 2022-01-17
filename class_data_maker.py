import argparse
from torchvision import transforms
from network.dataloader import (
    CSVDataset,
    collater,
    Resizer,
    AspectRatioBasedSampler,
    Augmenter,
    Crop,
    crop_collater,
    LabelFlip,
)

import matplotlib.patches as patches
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from PIL import Image

assert torch.__version__.split(".")[0] == "1"


def main(args=None):
    print(torch.__version__)
    parser = argparse.ArgumentParser(
        description="Simple training script for training a RetinaNet network."
    )
    parser.add_argument(
        "--csv_train", help="Path to file containing training annotations (see readme)"
    )
    parser.add_argument(
        "--csv_classes", help="Path to file containing class list (see readme)"
    )
    parser.add_argument(
        "--csv_val",
        help="Path to file containing validation annotations (optional, see readme)",
    )
    parser.add_argument("--debug", help="Batch size", type=bool, default=False)
    parser = parser.parse_args(args)

    """
    Data loaders
    """
    data_labels = {0: "buoy",
                   1: "boat",
                   2: "sailboat(D)",
                   3: "sailboat(U)",
                   4: "ferry",
                   5: "large_commercial_vessel",
                   6: "small_medium_fishing_boat",
                   7: "leisure_craft",
                   8: "buoy_green",
                   9: "buoy_red",
                   10: "harbour",
                   11: "human"}

    dataset_train = CSVDataset(
        train_file=parser.csv_train,
        class_list=parser.csv_classes,
    )
    debug = parser.debug
    path = r'Q:/classification/newData'
    if parser.csv_val is None:
        dataset_val = None
        print("No validation annotations provided.")
    else:
        dataset_val = CSVDataset(
            train_file=parser.csv_val,
            class_list=parser.csv_classes
        )
    i = 0
    for x in tqdm(dataset_train):
        if x.annot.shape[0] == 0:
            continue
        for j in range(x.annot.shape[0]):
            x1, y1, x2, y2, a = x.annot[j]
            w = x2 - x1
            h = y2 - y1
            image = Image.fromarray(np.uint8(x.img * 255))
            crop = image.crop(x.annot[j][:-1])
            import os
            class_path = os.path.join(path, data_labels[a])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            fn = os.path.join(class_path, f'{i}.jpg')
            crop.save(fn)
            i += 1

            if debug:
                plt.imshow(crop)
                plt.show()
                plt.imshow(x.img)
                ax = plt.gca()
                rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none', alpha=0.1)
                ax.add_patch(rect)
                plt.show()
                print('tt')


if __name__ == "__main__":
    main()
