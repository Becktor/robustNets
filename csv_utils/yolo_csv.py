import os
from os import walk, getcwd
from PIL import Image
import csv
import os
from pathlib import Path

IMAGE_SIZE = (1920, 1080)
ship_buoy = {0: "buoy", 1: "boat", 2: "harbour"}
sl = {0: 'ship', 1: 'large_commercial_vessel', 2: 'buoy', 3: 'sailboat_sail_up', 4: 'small_medium_fishing_boat',
      5: 'leisure_craft',
      6: 'sailboat_sail_down', 7: 'buoy_green', 8: 'buoy_red', 9: 'harbour', 10: 'human'}


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
        label = sl[d[0]]
        ret.append([x1, y1, x2, y2, label])
    return ret


def main(fn):
    converted = []
    fp = Path(fn)
    for filename in os.listdir(fn):
        with open(os.path.join(fn, filename), "r") as f:
            labels = f.readlines()
            for data in labels:
                pp = Path(filename)
                name = pp.stem+".png"
                path = os.path.join(fp.parent, "images")
                fname = os.path.join(path, name)
                if data == "":
                    converted.append([fname, "", "", "", "", ""])
                else:
                    tmp = convert(data)
                    for t in tmp:
                        converted.append([fname, t[0], t[1], t[2], t[3], t[4]])

    # name of csv file
    csvfilename = r"Q:\newData\dataset.csv"
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # writing to csv file
    with open(csvfilename, "a", newline="") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(converted)


if __name__ == "__main__":
    folders = [r"Q:\newData\JSON2YOLO\videoplayback\labels",
               r"Q:\newData\JSON2YOLO\videoplayback1\labels",
               r"Q:\newData\JSON2YOLO\videoplayback2\labels",
               r"Q:\newData\JSON2YOLO\videoplayback3\labels",
               r"Q:\newData\JSON2YOLO\Limfjorden_rundt_2015\labels",
               r"Q:\newData\JSON2YOLO\Palby_Fyn_Cup_2013_Aebeloe\labels",
               r"Q:\newData\JSON2YOLO\Palby_Fyn_Cup_2013_Galsklint\labels",
               r"Q:\newData\JSON2YOLO\Traeskibe_Limfjorden_2015\labels",
               r"Q:\newData\JSON2YOLO\Havkajak_Nibe_Aalborg\labels",
               r"Q:\newData\JSON2YOLO\handheld0\labels",
               r"Q:\newData\JSON2YOLO\handheld1\labels",
               r"Q:\newData\JSON2YOLO\handheld2\labels"]
    for fn in folders:
        main(fn)
