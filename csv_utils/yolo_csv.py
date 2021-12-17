import os
from os import walk, getcwd
from PIL import Image
import csv
import os

IMAGE_SIZE = (1920, 1080)
ship_buoy = {1: "buoy", 0: "boat", 2: "harbour"}


def x1x2y1y2_xywh(box):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x, y, w, h)


def xyhw_x1x2y1y2(xywh):
    x1 = xywh[0] - xywh[2] / 2.0
    x2 = xywh[0] + xywh[2] / 2.0
    y1 = xywh[1] - xywh[3] / 2.0
    y2 = xywh[1] + xywh[3] / 2.0
    return (x1, y1, x2, y2)


def convert(data):
    data = data.splitlines()
    ret = []
    for d in data:
        d = d.split(" ")
        d = [float(i) for i in d]
        x1, y1, x2, y2 = xyhw_x1x2y1y2(d[1:5])
        label = ship_buoy[d[0]]
        ret.append([x1, y1, x2, y2, label])
    return ret


def main(fn):
    converted = []
    for filename in os.listdir(fn):
        with open(os.path.join(fn, filename), "r") as f:
            data = f.read()
            if data == "":
                name = filename[:-4] + ".bmp"
                converted.append([name, "", "", "", "", ""])
            else:
                name = filename[:-4] + ".bmp"
                tmp = convert(data)

                for t in tmp:
                    if t[4] == "harbour":
                        continue
                    converted.append([name, t[0], t[1], t[2], t[3], t[4]])

    # name of csv file
    filename = "csvs/raymarine_20200611-2020_12_12_10_49_19.csv"
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # writing to csv file
    with open(filename, "w+", newline="") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(converted)


if __name__ == "__main__":
    fn = r"/home/jobe/Downloads/csvs/aalborg_02_07_2019-2021_06_02_09_15_19/obj_train_data/jobe_gbar/aalborg_02_07_2019/converted/color"
    fn = r"/home/jobe/Downloads/csvs/faaborg-2021_04_24_09_07_20/obj_train_data/NAS/20201215-TucoPlatform/Disk 2"
    fn = r"/home/jobe/Downloads/csvs/raymarine 20200612-2020_12_06_10_42_26/obj_train_data/jobe_gbar/20200612"
    fn = r"/home/jobe/Downloads/csvs/raymarine_20200611-2020_12_12_10_49_19/obj_train_data/jobe_gbar/20200611"
    main(fn)
