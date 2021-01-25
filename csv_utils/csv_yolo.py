import os
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
    name = os.path.basename(csv[0])[:-4]
    if csv[2] == '':
        return name, [-1,-1,-1,-1,-1]

    bbox = [float(i) for i in csv[1:5]]
    #bbox = box_to_pbox(bbox)
    x, y, h, w = x1x2y1y2xyhw(bbox)

    label = ship_buoy[csv[5]]
    return name, [label, x, y, h, w]


def main(fn):
    with open(fn, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:
            name, tmp = convert(lines)
            l, x, y, h, w = tmp
            if lines[2] == '':
                continue
            else:
                yolo = "{} {:06f} {:06f} {:06f} {:06f}\n".format(l, x, y, h, w)
            print("name: {} -- content {}".format(name, tmp))
            pn = '2019_04_12_yolo'
            if not os.path.exists(pn):
                os.makedirs(pn)
            n = os.path.join(pn, name + '.txt')
            file2 = open(n, "a")
            file2.write(yolo)
            file2.close()


if __name__ == '__main__':
    #fn = r'Z:\Shippinglab\mmdet\rgb_train.csv'
    #fn2 = r'Z:\Shippinglab\mmdet\weight_set.csv'
    fn = r'Z:\Shippinglab\mmdet\2019_04_12.csv'
    main(fn)
