import os
from os import walk, getcwd
from PIL import Image
import csv

IMAGE_SIZE = (1920, 1080)
ship_buoy = {'buoy': 1,
             'Buoy': 1,
             'Motorboat': 0,
             'Sailboat (D)': 0,
             'Ferry': 0,
             'Sailboat (U)': 0,
             'Ship': 0,
             'boat': 0,
             '': -1}


def box_to_pbox(box):
    pbox = [0, 0, 0, 0]
    pbox[0] = box[0] / IMAGE_SIZE[0]
    pbox[1] = box[1] / IMAGE_SIZE[1]
    pbox[2] = box[2] / IMAGE_SIZE[0]
    pbox[3] = box[3] / IMAGE_SIZE[1]
    return pbox


def x1x2y1y2xyhw(box):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x, y, w, h)


def convert(csv):
    start = 34
    name = csv[0][:-4]
    bbox = [float(i) for i in csv[1:5]]
    bbox = box_to_pbox(bbox)
    x, y, h, w = x1x2y1y2xyhw(bbox)

    label = ship_buoy[csv[5]]
    return name, [label, x, y, h, w]


def main(fn):
    with open(fn, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:
            if lines[2] == '':
                continue
            name, tmp = convert(lines)
            l, x, y, h, w = tmp
            print("name: {} -- content {}".format(name, tmp))
            pn = 't12'
            yolo = "{} {:06f} {:06f} {:06f} {:06f}\n".format(l, x, y, h, w)
            if not os.path.exists(pn):
                os.makedirs(pn)
            n = os.path.join(pn, name + '.txt')
            file = open(n, "a")
            file.write(yolo)
            file.close()


if __name__ == '__main__':
    fn = r'Z:\Shippinglab\mmdet\rgb_train.csv'
    fn2 = r'Z:\Shippinglab\mmdet\weight_set.csv'
    fn = r'C:\Users\Jobe\Documents\git\robustNets\20200612-TucoPlatformB.csv'
    main(fn)
