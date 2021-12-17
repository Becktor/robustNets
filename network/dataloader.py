from __future__ import print_function, division

import csv
import os
import random
import sys
import imgaug.augmenters as iaa
import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import torch
from PIL import Image
from future.utils import raise_from
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils import to_var


class Batch:
    """CSV batch"""

    def __init__(self, samples):
        self._samples = samples
        self.i = 0

    def as_batch(self):
        imgs = [s.img for s in self._samples]
        annots = [s.annot for s in self._samples]
        names = [s.name for s in self._samples]
        idx = [s.index for s in self._samples]
        crop_ids = [s.crop_id for s in self._samples]

        torch_img = to_var(torch.stack(imgs))
        torch_annots = to_var(torch.stack(annots))

        return torch_img, torch_annots, names, idx, crop_ids

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self._samples):
            elem = self._samples[self.i]
            self.i += 1
            return elem
        else:
            raise StopIteration()

    @property
    def samples(self):
        return self._samples


class Sample:
    """CSV sample"""

    def __init__(self, img, annot, name, index, crop_id=-1, parent=None):
        self._img = img
        self._annot = annot
        self._name = name
        self._index = index
        self._parent = parent
        self._alt = False
        self._crop_id = crop_id

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self._img = value

    @property
    def annot(self):
        return self._annot

    @annot.setter
    def annot(self, value):
        self._annot = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def alt(self):
        if self._parent is None:
            return self._alt
        else:
            return self._parent.alt

    @img.setter
    def alt(self, value):
        self._alt = value

    @property
    def crop_id(self):
        return self._crop_id

    @crop_id.setter
    def crop_id(self, value):
        self._crop_id = value


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, use_path=True, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=","))
        except ValueError as e:
            raise_from(
                ValueError("invalid CSV class file: {}: {}".format(self.class_list, e)),
                None,
            )

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(
                    csv.reader(file, delimiter=","), self.classes
                )
        except ValueError as e:
            raise_from(
                ValueError(
                    "invalid CSV annotations file: {}: {}".format(self.train_file, e)
                ),
                None,
            )
        self.image_names = list(self.image_data.keys())

        if not use_path:
            for x, image_name in enumerate(self.image_names):
                dir_path = os.path.dirname(train_file)
                filename = os.path.split(image_name)[-1]
                img_path = os.path.join(
                    os.path.join(dir_path, "2019_04_12_data"), filename
                )
                img_data = self.image_data[image_name]
                del self.image_data[image_name]
                self.image_names[x] = img_path
                self.image_data[img_path] = img_data

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, "rb")
        else:
            return open(path, "r", newline="")

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(
                    ValueError(
                        "line {}: format should be 'class_name,class_id'".format(line)
                    ),
                    None,
                )
            class_id = self._parse(
                class_id, int, "line {}: malformed class ID: {{}}".format(line)
            )

            if class_name in result:
                raise ValueError(
                    "line {}: duplicate class name: '{}'".format(line, class_name)
                )
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        name = self.image_names[idx]
        annot = self.load_annotations(idx, img.shape)

        sample = Sample(img, annot, name, idx)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def img_anno(self, idx):
        img = self.load_image(idx)
        name = self.image_names[idx]
        annot = self.load_annotations(idx, img.shape)

        sample = Sample(img, annot, name, idx)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        tst = self.image_names[image_index]

        img = skimage.io.imread(tst)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index, image_shape):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = int(image_shape[1] * a["x1"])
            x2 = int(image_shape[1] * a["x2"])
            y1 = int(image_shape[0] * a["y1"])
            y2 = int(image_shape[0] * a["y2"])

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a["class"])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(
                    ValueError(
                        "line {}: format should be 'img_file,x1,y1,x2,y2,class_name' or 'img_file,,,,,'".format(
                            line
                        )
                    ),
                    None,
                )

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ("", "", "", "", ""):
                continue

            x1 = self._parse(x1, float, "line {}: malformed x1: {{}}".format(line))
            y1 = self._parse(y1, float, "line {}: malformed y1: {{}}".format(line))
            x2 = self._parse(x2, float, "line {}: malformed x2: {{}}".format(line))
            y2 = self._parse(y2, float, "line {}: malformed y2: {{}}".format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError(
                    "line {}: x2 ({}) must be higher than x1 ({})".format(line, x2, x1)
                )
            if y2 <= y1:
                raise ValueError(
                    "line {}: y2 ({}) must be higher than y1 ({})".format(line, y2, y1)
                )

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError(
                    "line {}: unknown class name: '{}' (classes: {})".format(
                        line, class_name, classes
                    )
                )

            result[img_file].append(
                {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "class": class_name}
            )
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s.img for s in data]
    annots = [s.annot for s in data]
    names = [s.name for s in data]
    index = [s.index for s in data]
    crop_ids = [s.crop_id for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]

    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, : int(img.shape[0]), : int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, : annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    collated_samples = []
    for i, img in enumerate(padded_imgs):
        collated_samples.append(
            Sample(img, annot_padded[i], names[i], index[i], crop_ids[i])
        )

    batch = Batch(collated_samples)
    return batch


def crop_collater(data):
    imgs = [s.img for s in data]
    annots = [s.annot for s in data]
    names = [s.name for s in data]
    index = [s.index for s in data]
    batch_size = len(imgs)
    collated_samples = []
    too_small = []
    max_num_annots = max(a.shape[0] for a in annots)
    for batch_elem in range(batch_size):
        new_crop = []
        image = imgs[batch_elem]
        annotation = annots[batch_elem]
        if image.shape[0] > 1080:
            scale = 1080 / image.shape[0]
            image = skimage.transform.resize(sm1, (1080, int(image.shape[1] * scale)))
            annotation[:4] *= scale

        rows, cols, cns = image.shape
        mid = int(rows / 20) * 11
        height = int(rows / 3)
        height += 32 - height % 32
        n = 3
        width = height

        x_increment = cols // n
        y_sp = int(mid - height / 2)

        # Create half bbx
        cr = cols - rows
        sm1 = image[:, :rows, :]
        sm2 = image[:, cr:, :]

        cropped_imgs = {0: (sm1, (0, 0, rows, rows)), 1: (sm2, (cr, 0, rows, rows))}

        # Create small bbx
        for x in range(n):
            x_sp = x * x_increment
            cropped_imgs[x + 2] = (
                image[y_sp : y_sp + height :, x_sp : x_sp + width, :],
                (x_sp, y_sp, width, height),
            )

        sample_crops = {}
        scale = height / rows
        for an in annotation:
            x1, y1, x2, y2, lbl = an
            for key in cropped_imgs:
                img, sp = cropped_imgs[key]
                if sp[0] < x1 < sp[0] + sp[2] and sp[1] < y1 < sp[1] + sp[3]:
                    n_x1 = x1 - sp[0]
                    n_y1 = y1 - sp[1]
                    n_x2 = n_x1 + sp[2] if x2 > sp[0] + sp[2] else x2 - sp[0]
                    n_y2 = n_y1 + sp[3] if y2 > sp[1] + sp[3] else y2 - sp[1]

                    anno = np.array([n_x1, n_y1, n_x2, n_y2, lbl])
                    if (key == 0) or (key == 1):
                        anno[:4] *= scale
                    area = (anno[2] - anno[0]) * (anno[3] - anno[1])
                    if area >= 48:
                        sample_crops.setdefault(key, []).append(anno)
                    else:
                        too_small.append((names[batch_elem], area))
        sm1 = skimage.transform.resize(sm1, (int(height), int(round(width))))
        sm2 = skimage.transform.resize(sm2, (int(height), int(round(width))))

        i, s = cropped_imgs[0]
        cropped_imgs[0] = (sm1, s)
        cropped_imgs[1] = (sm2, s)

        for key in cropped_imgs:
            img, sp = cropped_imgs[key]
            try:
                image = torch.FloatTensor(img.copy())
            except Exception as e:
                print(e)
            new_anno = []
            if key in sample_crops:
                an = np.stack(sample_crops[key])
                annot = torch.FloatTensor(an)
                new_anno = annot
            new_crop.append((image, new_anno))

        torch_imgs = torch.zeros(len(new_crop), width, height, 3)
        torch_annots = torch.ones((len(new_crop), max_num_annots, 5)) * -1

        for i in range(len(new_crop)):
            img, an = new_crop[i]
            torch_imgs[i, : int(img.shape[0]), : int(img.shape[1]), :] = img

            if max_num_annots > 0:
                if len(an) > 0:
                    torch_annots[i, : an.shape[0], :] = an

        torch_imgs = torch_imgs.permute(0, 3, 1, 2)
        for i in range(torch_imgs.shape[0]):
            img = torch_imgs[i]
            an = torch_annots[i]
            nm = names[batch_elem]
            id = index[batch_elem]
            d = data[batch_elem]

            collated_samples.append(Sample(img, an, nm, id, d))
    batch = Batch(collated_samples)
    return batch


class Gaussian(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, sample):
        image = sample.img
        annots = sample.annot
        if self.noise_level > 0:
            image = (
                image
                + torch.empty(image.shape).normal_(mean=0, std=self.noise_level).numpy()
            )
            image = np.clip(0, 1, image)
        # temp_img = (image * 255).astype(np.uint8)
        # Image.fromarray(temp_img).save("noisy.png")
        sample.img = image.astype(np.float32)
        sample.annot = annots
        return sample


class GaussianBlur(object):
    def __init__(self, level):
        self.level = level
        self.blur = iaa.Sequential(iaa.blur.GaussianBlur(level))

    def __call__(self, sample, val=False):
        image = sample.img
        annots = sample.annot
        if self.level <= 0:
            return sample
        uint8_image = (image * 255).astype(np.uint8)
        # Image.fromarray(uint8_image).save("Blur_inp.png")
        aug_image = self.blur(image=uint8_image)
        # Image.fromarray(aug_image).save("Blur_aug.png")
        image = aug_image.astype(np.float) / 255
        image = np.clip(0, 1, image)
        sample.img = image.astype(np.float32)
        sample.annot = annots
        return sample


class SAP(object):
    """salt & pepper to input"""

    def __init__(self, percentage, channel_wise=False):
        self.per = percentage
        self.sap = iaa.Sequential(iaa.SaltAndPepper(percentage), channel_wise)

    def __call__(self, sample, val=False):
        image = sample.img
        annots = sample.annot
        if self.per <= 0:
            return {"img": image, "annot": annots}
        uint8_image = (image * 255).astype(np.uint8)
        # Image.fromarray(uint8_image).save("SP_inp.png")
        aug_image = self.sap(image=uint8_image)
        Image.fromarray(aug_image).save("SP_aug.png")
        image = aug_image.astype(np.float) / 255
        image = np.clip(0, 1, image)
        sample.img = image.astype(np.float32)
        sample.annot = annots
        return sample


class AddWeather(object):
    """Apply weather to input"""

    def __init__(self, aug, weight):
        self.aug = aug
        max_weight = 3
        self.weight = weight
        if max_weight < weight:
            self.weight = 0
        elif weight < 0:
            self.weight = -1
        else:
            self.weight = 3 - weight

    def __call__(self, sample, val=False):
        image = sample.img
        annots = sample.annot
        if self.weight < 0:
            return {"img": image, "annot": annots}
        uint8_image = (image * 255).astype(np.uint8)
        # Image.fromarray(uint8_image).save("inp.png")
        aug_image = self.aug(image=uint8_image)
        # Image.fromarray(aug_image).save("aug.png")
        norm_aug_img = aug_image.astype(np.float) / 255
        image = (image * self.weight + norm_aug_img) / (self.weight + 1)
        image = np.clip(0, 1, image)
        # temp_img = (image * 255).astype(np.uint8)
        # Image.fromarray(temp_img).save("noisy.png")
        sample.img = image.astype(np.float32)
        sample.annot = annots
        return sample


class Crop(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, val=False, reweight=False, debug=False):
        self.val = val
        self.debug = debug
        self.flipflop = 0
        self.reweight = reweight
        if self.val:
            random.seed(0)

    def __call__(self, sample, debug=False):
        image = sample.img
        annots = sample.annot
        name = sample.name
        not_added = []
        if image.shape[0] > 1080:
            scale = 1080 / image.shape[0]
            image = skimage.transform.resize(image, (1080, int(image.shape[1] * scale)))
            annots[:, :4] *= scale
        rows, cols, cns = image.shape
        mid = int(rows / 20) * 11
        height = int(rows / 3)
        height += 32 - height % 32
        n = 3
        width = height

        x_increment = cols // n
        y_sp = int(mid - height / 2)

        # Create half bbx
        cr = cols - rows
        sm0 = image[:, :rows, :]
        sm1 = image[:, cr:, :]

        cropped_imgs = {0: (sm0, (0, 0, rows, rows)), 1: (sm1, (cr, 0, rows, rows))}

        # Create small bbx
        for x in range(n):
            x_sp = x * x_increment
            cropped_imgs[x + 2] = (
                image[y_sp : y_sp + height :, x_sp : x_sp + width, :],
                (x_sp, y_sp, width, height),
            )

        sample_crops = {}
        scale = height / rows
        for an in annots:
            x1, y1, x2, y2, lbl = an
            for key in cropped_imgs:
                _, sp = cropped_imgs[key]
                if sp[0] < x1 < sp[0] + sp[2] and sp[1] < y1 < sp[1] + sp[3]:
                    n_x1 = x1 - sp[0]
                    n_y1 = y1 - sp[1]
                    n_x2 = n_x1 + sp[2] if x2 > sp[0] + sp[2] else x2 - sp[0]
                    n_y2 = n_y1 + sp[3] if y2 > sp[1] + sp[3] else y2 - sp[1]

                    anno = [n_x1, n_y1, n_x2, n_y2, lbl]
                    area = (anno[2] - anno[0]) * (anno[3] - anno[1])
                    # if area > 32:
                    sample_crops.setdefault(key, []).append(anno)
                    # else:
                    #     not_added.append((key,area))

        keys = list(cropped_imgs.keys())
        key = random.choice(keys)
        img = cropped_imgs[key][0]
        annotations = annots
        crop_id = -1
        ## if any labels exists select one and save crop id.
        if len(sample_crops) > 0:
            keys = list(sample_crops.keys())
            key = random.choice(keys)
            img = cropped_imgs[key][0]
            annotations = np.array(sample_crops[key])
            crop_id = key
            # if self.reweight:
            #     keys = []
            #     for k in sample_crops:
            #         if self.flipflop == 0 and sample_crops[k][4] == 1:
            #             keys.append(k)
            #         elif self.flipflop == 1:
            #             keys.append(k)
            #     if len(keys) > 0:
            #         key = random.choice(keys)
            #         img = cropped_imgs[key][0]
            #         annotation = np.array(sample_crops[key])
            #     self.flipflop = 0 if self.flipflop == 1 else 1

        if debug:
            import matplotlib.pyplot as plt
            import cv2

            for key in sample_crops:
                img = cropped_imgs[key][0]
                annotations = np.array(sample_crops[key])
                tmp_img = img
                for v in annotations:
                    if (int(v[2]) - int(v[0])) * (int(v[3]) - int(v[1])) < 16:
                        tmp_img = cv2.rectangle(
                            img,
                            (int(v[0]), int(v[1])),
                            (int(v[2]), int(v[3])),
                            color=(0, 0, 1),
                            thickness=2,
                        )
                plt.imshow(tmp_img)
                plt.show()
        sample.img = img
        sample.crop_id = crop_id
        sample.annot = annotations
        sample.name = name
        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=384, max_side=512):
        image = sample.img
        annots = sample.annot
        name = sample.name
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (int(round(rows * scale)), int(round((cols * scale))))
        )
        rows, cols, cns = image.shape
        pad_w, pad_h = (0, 0)
        if rows % 32 != 0:
            pad_w = 32 - rows % 32
        if cols % 32 != 0:
            pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale
        resized_annots = np.ones_like(annots) * -1
        for x in range(len(annots)):
            area = (annots[x, 2] - annots[x, 0]) * (annots[x, 3] - annots[x, 1])
            if area > 16:
                resized_annots[x] = annots[x]

        sample.img = torch.from_numpy(new_image)
        sample.annot = torch.from_numpy(resized_annots)
        sample.name = name

        return sample


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image = sample.img
            annots = sample.annot
            name = sample.name
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample.img = image
            sample.annot = annots
            sample.name = name

        return sample


class LabelFlip(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, mod=2):
        np.random.seed(0)
        self._mod = mod
        self._alt = {}

    def __call__(self, sample, flip_x=0.5):
        image = sample.img
        annots = sample.annot
        idx = sample.index
        if self._mod == 0 or idx in self._alt:
            return sample
        elif idx % self._mod == 0:
            f_annots = np.ones_like(annots) * -1
            for x in range(len(annots)):
                if np.random.rand() < flip_x:
                    f_annots[x] = annots[x].copy()
                    f_annots[x, 4] = 0 if f_annots[x, 4] == 1 else 1
                else:
                    continue
            sample.img = image
            sample.annot = f_annots
        return sample

    def alt(self, key, value):
        self._alt[key] = value


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        sample.img = (sample.img.astype(np.float32) - self.mean) / self.std
        return sample


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))

        # divide into groups, one group = one batch
        return [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]
