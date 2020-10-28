import os
from os import walk, getcwd
from PIL import Image
import csv

ship_buoy = {'buoy': 1,
             'Buoy': 1,
             'Motorboat': 0,
             'Sailboat (D)': 0,
             'Ferry': 0,
             'Sailboat (U)': 0,
             'boat': 0,
             '': -1}


def x1x2y1y2xyhw(box):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x, y, w, h)


def convert(csv):
    name = csv[0][34:-4]
    fb = [float(i) for i in csv[1:5]]
    x, y, h, w = x1x2y1y2xyhw(fb)
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

            yolo = "{} {:06f} {:06f} {:06f} {:06f}\n".format(l, x, y, h, w)
            if not os.path.exists('yolo2'):
                os.makedirs('yolo2')
            n = os.path.join('yolo2', name + '.txt')
            file = open(n, "a")
            file.write(yolo)
            file.close()


if __name__ == '__main__':
    fn = r'Z:\Shippinglab\mmdet\rgb_train.csv'
    fn2 = r'Z:\Shippinglab\mmdet\weight_set.csv'
    main(fn)
