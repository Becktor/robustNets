import argparse
import os
import time

import cv2
import numpy as np
import torch

from network import retinanet
from torch.utils.data import DataLoader
from torchvision import transforms

from network.dataloader import (
    CSVDataset,
    collater,
    Resizer,
    AspectRatioBasedSampler,
    UnNormalizer,
    Normalizer,
)

assert torch.__version__.split(".")[0] == "1"


# print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Validation script for RetinaNet network."
    )
    parser.add_argument(
        "--csv_classes", help="Path to file containing class list (see readme)"
    )
    parser.add_argument(
        "--csv_val",
        help="Path to file containing validation annotations (optional, see readme)",
    )
    parser.add_argument("--model", help="Path to model (.pt) file.")

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(
        train_file=parser.csv_val,
        class_list=parser.csv_classes,
        transform=transforms.Compose([Normalizer(), Resizer()]),
    )

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(
        dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val
    )
    retinanet = retinanet.resnet50(dataset_val.num_classes())

    use_gpu = True
    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()
    checkpoint = torch.load(parser.model)
    retinanet.load_state_dict(checkpoint["state_dict"])

    retinanet.eval()
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(
            image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2
        )
        cv2.putText(
            image,
            caption,
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
        )

    for idx, data in enumerate(dataloader_val):
        cimg = data["img"]
        s = cimg.shape
        noisy_data = cimg + (0.01 ** 0.5) * torch.randn(s[0], s[1], s[2], s[3])
        nd = [cimg, noisy_data]
        for ix, d in enumerate(nd):
            with torch.no_grad():
                st = time.time()
                scores, classification, transformed_anchors = retinanet(d)
                print("Elapsed time: {}".format(time.time() - st))
                idxs = np.where(scores.cpu() > 0.5)
                img = np.array(255 * unnormalize(d[0, :, :, :])).copy()

                img[img < 0] = 0
                img[img > 255] = 255

                img = np.transpose(img, (1, 2, 0))

                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    save = True
                    print(idx, x1, x2, y1, y2)
                    label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                    draw_caption(img, (x1, y1, x2, y2), label_name)
                    img = cv2.rectangle(
                        img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2
                    )
                    print(label_name)

                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                cv2.imwrite(
                    r"C:\Users\Jobe\Documents\git\pytorch-network\{}_img_".format(idx)
                    + str(ix)
                    + ".png",
                    img,
                )


if __name__ == "__main__":
    main()
