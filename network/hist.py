import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv


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


class BoxGenerator:
    def __init__(self, path):
        self.path = path
        self.images = self.image_gen()
        self.file = open(path)
        self.ground_truth_dir = next(csv.reader(self.file, delimiter=','))
        self.ground_truth_file = open(*self.ground_truth_dir)
        self.init_gen()

    def init_gen(self):
        self.file.seek(0)
        self.ground_truth_file.seek(0)
        self.box_gen = self.generator(self.file)
        next(self.box_gen)
        self.truth_gen = self.generator(self.ground_truth_file)
        self.img = self.image_gen()

    def destroy(self):
        self.file.close()
        self.ground_truth_file.close()

    def image_gen(self):
        with open(self.path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            oldRow = None
            for i, row in enumerate(csv_reader):
                if len(row) == 1:
                    continue

                if row[0] == oldRow:
                    continue

                else:
                    oldRow = row[0]
                    yield row[0]

    @staticmethod
    def generator(csv_file):
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_lines = list(csv_reader)
        csv_file.seek(0)
        csv_reader = csv.reader(csv_file, delimiter=',')
        oldRow = None
        boxes = []
        for i, row in enumerate(csv_reader):
            if len(row) == 1:
                yield row
                continue

            if row[0] == oldRow:
                continue

            else:
                currentBoxes = [x for x in csv_lines if x[0] == row[0]]
                boxes.clear()
                for line in currentBoxes:
                    if (line[5] == 'Buoy') or (line[5] == 'buoy'):
                        label = 0
                    else:
                        label = 1
                    if line[5] != '':
                        if len(line) == 7:
                            boxes.append(
                                [int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4])),
                                 label, float(line[6])])

                        else:
                            boxes.append(
                                [int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4])),
                                 label])

                oldRow = row[0]
                yield boxes


class Hist:
    def __init__(self, path, iou_thresh=0.3, conf_thresh=0.7):
        self.path = path
        self.gen = BoxGenerator(path)
        self.gen.init_gen()
        self.iouThresh = iou_thresh
        self.confThresh = conf_thresh
        self.TP = list()
        self.FN = list()
        self.FP = list()

    @staticmethod
    def area(box):
        return np.abs(box[0] - box[2]) * np.abs(box[1] - box[3])

    def eval(self, ground_truth, pred_box):
        FP = np.ones(len(pred_box))
        for gIdx, g in enumerate(ground_truth):
            bestIou = 0
            bestIdx = -1
            for bIdx, b in enumerate(pred_box):
                if b[5] < self.confThresh:
                    FP[bIdx] = 0
                    continue

                bbIou = iou(g[0:4], b[0:4])

                if bbIou > bestIou:
                    bestIdx = bIdx
                    bestIou = bbIou
            TPfound = False

            if bestIdx != -1:
                if bestIou > self.iouThresh and g[4] == pred_box[bestIdx][4]:
                    self.TP.append(self.area(g))
                    TPfound = True
                    FP[bestIdx] = 0

            if not TPfound:
                self.FN.append(self.area(g))

        for bIdx, b in enumerate(pred_box):
            if FP[bIdx] == 1:
                self.FP.append(self.area(b))

    def run(self):
        for g, box in zip(self.gen.truth_gen, self.gen.box_gen):
            self.eval(g, box)

    def show_hist(self, xrange=3000, name=''):

        self.run()
        # Create subplots
        fig, ax1 = plt.subplots()
        # Find amount of True Positives and False Negatives
        sumDetections = len(self.TP) + len(self.FN)
        # Create scaling weights such that cumulative TP + FN sums to 1
        w = np.empty(np.array(self.TP).shape[0])
        w.fill(1 / (sumDetections + 1e-8))

        # make histogram for TP
        ax1.hist(np.array(self.TP), bins=100, weights=w, density=0, cumulative=0, alpha=0.5,
                 label='Detected',
                 range=(0, xrange))

        # Create FN scaling weights
        w = np.empty(np.array(self.FN).shape[0])
        w.fill(1 / (sumDetections + 1e-8))

        # create histogram for FN
        ax1.hist(np.array(self.FN), bins=100, weights=w, density=0, cumulative=0, alpha=0.5,
                 label='Not Detected', range=(0, xrange))

        # combine the x-axis of the two histograms
        ax2 = ax1.twinx()

        # extract bins + numbers for stepped recall curve

        n1, b1 = np.histogram(np.array(self.TP), bins=1000, range=(0, xrange))
        n2, b2 = np.histogram(np.array(self.FN), bins=1000, range=(0, xrange))

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

        n1, b1 = np.histogram(np.array(self.TP), bins=1000, range=(0, xrange))
        n3, b3 = np.histogram(np.array(self.FP), bins=1000, range=(0, xrange))

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
        fig.savefig('figures/' + 'hist.png')


def main():
    parser = argparse.ArgumentParser(description='Program for showing detection histogram')
    parser.add_argument("-p", "--path", help="file path to results from model", type=str)
    parser.add_argument("-t", "--title", help="figure title name", type=str, default='')
    parser.add_argument("-iou", "--iouThresh", help="Evaluation iou threshold", type=float, default=0.3)
    parser.add_argument("-conf", "--confThresh", help="Evaluation confidence threshold", type=float, default=0.7)
    args = parser.parse_args()

    hist_class = Hist(args.path, args.iouThresh, args.confThresh)

    hist_class.show_hist(3000, args.title)


if __name__ == "__main__":
    main()
