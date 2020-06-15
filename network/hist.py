import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv

from torchvision import transforms

from network.csv_eval import *
import torch

from network.dataloader import CSVDataset, Normalizer, Resizer


def iou(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float((boxAArea + boxBArea - interArea) + 0.001)
    return iou


class Hist:
    def __init__(self, pred, anno, iou_thresh=0.3, conf_thresh=0.7):
        self.iouThresh = iou_thresh
        self.confThresh = conf_thresh
        self.pred = pred
        self.anno = anno
        self.TP = {}
        self.FN = {}
        self.FP = {}

    @staticmethod
    def area(box):
        return np.abs(box[0] - box[2]) * np.abs(box[1] - box[3])

    def eval(self, ground_truth, pred_box, pred_label):
        FP = np.ones(len(pred_box))
        fp = []
        tp = []
        fn = []
        for gt_label, gt in enumerate(ground_truth):
            for gIdx, g in enumerate(gt):
                bestIou = 0
                bestIdx = -1
                for bIdx, b in enumerate(pred_box):
                    if b[4] < self.confThresh:
                        FP[bIdx] = 0
                        continue
                    if g.size == 0:
                        continue
                    bbIou = iou(g[0:4], b[0:4])

                    if bbIou > bestIou:
                        bestIdx = bIdx
                        bestIou = bbIou
                if g.size == 0:
                    continue

                TPfound = False

                if bestIou > self.iouThresh and gt_label == pred_label:
                    tp.append(self.area(g))
                    TPfound = True
                    FP[bestIdx] = 0

                if not TPfound and gt_label == pred_label:
                    fn.append(self.area(g))

        for bIdx, b in enumerate(pred_box):
            if FP[bIdx] == 1:
                fp.append(self.area(b))

        if pred_label in self.TP:
            self.TP[pred_label] = self.TP[pred_label] + tp
        else:
            self.TP[pred_label] = tp
        if pred_label in self.FP:
            self.FP[pred_label] = self.FP[pred_label] + fp
        else:
            self.FP[pred_label] = fp
        if pred_label in self.FN:
            self.FN[pred_label] = self.FN[pred_label] + fn
        else:
            self.FN[pred_label] = fn

    def run(self):
        for g, box in zip(self.anno, self.pred):
            self.eval(g, box[0], 0)
            self.eval(g, box[1], 1)

    def show_hist(self, xrange=3000, lbl=-1, name=''):
        if lbl < 0:
            tp = self.TP[0] + self.TP[1]
            fp = self.FP[0] + self.FP[1]
            fn = self.FN[0] + self.FN[1]
        else:
            tp = self.TP[lbl]
            fp = self.FP[lbl]
            fn = self.FN[lbl]

        self.run()
        # Create subplots
        fig, ax1 = plt.subplots()
        # Find amount of True Positives and False Negatives
        sumDetections = len(tp) + len(fn)
        # Create scaling weights such that cumulative TP + FN sums to 1
        w = np.empty(np.array(tp).shape[0])
        w.fill(1 / (sumDetections + 1e-8))

        # make histogram for TP
        ax1.hist(np.array(tp), bins=100, weights=w, density=0, cumulative=0, alpha=0.5,
                 label='Detected',
                 range=(0, xrange))

        # Create FN scaling weights
        w = np.empty(np.array(fn).shape[0])
        w.fill(1 / (sumDetections + 1e-8))

        # create histogram for FN
        ax1.hist(np.array(fn), bins=100, weights=w, density=0, cumulative=0, alpha=0.5,
                 label='Not Detected', range=(0, xrange))

        # combine the x-axis of the two histograms
        ax2 = ax1.twinx()

        # extract bins + numbers for stepped recall curve

        n1, b1 = np.histogram(np.array(tp), bins=1000, range=(0, xrange))
        n2, b2 = np.histogram(np.array(fn), bins=1000, range=(0, xrange))

        n1_old = 0
        n2_old = 0
        x_new = [0]
        y_new = [0]
        allowPrint = True
        for b, in1, in2 in zip(b1[:-1], n1, n2):
            n1_old += in1
            n2_old += in2
            r = n1_old / (n1_old + n2_old + 1e-6)
            if (n1_old + n2_old) > 5:
                x_new.append(b)
                y_new.append(r)
            if r >= 0.5 and (n1_old + n2_old) > 10 and allowPrint:
                print('Recall of 0.5 achieved at a minimum pixel area of ', b)
                allowPrint = False

        for x, y in zip(x_new, y_new):
            if y > 0.9 * np.max(y_new):
                print('90% of maximum recall achieved at pixel area of ', x)
                break
        ax2.plot(x_new, y_new, linewidth=2, c='lime', label='Recall')

        # extract bins + numbers for stepped precision curve

        n1, b1 = np.histogram(np.array(tp), bins=1000, range=(0, xrange))
        n3, b3 = np.histogram(np.array(fp), bins=1000, range=(0, xrange))

        n1_old = 0
        n2_old = 0
        x_new = [0]
        y_new = [0]
        for b, in1, in2 in zip(b1[:-1], n1, n3):
            n1_old += in1
            n2_old += in2
            r = n1_old / (n1_old + n2_old + 1e-6)
            if (n1_old + n2_old) > 5:
                x_new.append(b)
                y_new.append(r)

        ax2.plot(x_new, y_new, linewidth=2, c='g', label='Precision')

        # make axis limits + legends
        ax2.set_ylim([0, 1.1])
        ax1.set_ylim([0, 0.09])
        ax2.set_xlim([0, xrange])
        ax1.set_xlim([0, xrange])

        ax2.legend(loc='lower left', bbox_to_anchor=(0.55, 1.0), ncol=2,
                   borderaxespad=0, frameon=False)
        ax1.legend(loc='lower right', bbox_to_anchor=(0.55, 1.0), ncol=2,
                   borderaxespad=0, frameon=False)

        ax1.set_title(name + ' Detection Histogram', loc='center', y=1.07)
        ax1.set_ylabel('Density')
        ax2.set_ylabel('Value')
        ax1.set_xlabel('Object Area [Pixels]')
        ax1.minorticks_on()
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.5)
        fig.tight_layout()
        plt.show()
        fig.savefig('../figures/' + name + '_hist.png')


def main(args=None):
    parser = argparse.ArgumentParser(description='Program for showing detection histogram')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument("-t", "--title", help="figure title name", type=str, default='Test')
    parser.add_argument("-iou", "--iouThresh", help="Evaluation iou threshold", type=float, default=0.3)
    parser.add_argument("-conf", "--confThresh", help="Evaluation confidence threshold", type=float, default=0.3)
    parser = parser.parse_args(args)

    model = torch.load(parser.model)
    model_name = os.path.basename(os.path.splitext(parser.model)[0])
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model.eval()
    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    noise = 0.2
    all_detections = get_detections(dataset_val, model, score_threshold=0.3,
                                    max_detections=100, noise_level=noise)
    all_annotations = get_annotations(dataset_val)

    hist_class = Hist(all_detections, all_annotations, parser.iouThresh, parser.confThresh)
    hist_class.run()
    hist_class.show_hist(3000, name=model_name+'_BB_' + str(noise))

    hist_class.show_hist(3000, lbl=0, name=model_name+'_Buoy_' + str(noise))
    hist_class.show_hist(3000, lbl=1, name=model_name+'_Boat_' + str(noise))


if __name__ == "__main__":
    main()
