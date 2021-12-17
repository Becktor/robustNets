import os
from os import walk, getcwd
from PIL import Image
import csv
import random


def main(fn):
    for csv_pair in csvs_path:
        csv_path = csv_pair
        path = csvs_path[csv_path]

        with open(csv_path, mode="r") as file:
            # reading the CSV file
            csvFile = csv.reader(file)
            # displaying the contents of the CSV file
            cntr = 0
            pn = "joined_jbibe.csv"
            for lines in csvFile:
                name = os.path.split(lines[0])[-1]
                name = os.path.join(path, name)

                if lines[2] == "":
                    continue
                    nl = "{},,,,,\n".format(name)
                else:
                    nl = "{},{:06f},{:06f},{:06f},{:06f},{}\n".format(
                        name,
                        float(lines[1]),
                        float(lines[2]),
                        float(lines[3]),
                        float(lines[4]),
                        lines[5],
                    )
                # print(nl)
                file = open(pn, "a+")
                file.write(nl)
                file.close()
                cntr += 1
            print(cntr)


if __name__ == "__main__":
    # csvs_path = {r'Z:\Shippinglab\Raymarine20200611.csv': r'\Shippinglab\20200611',
    #             r'Z:\Shippinglab\Raymarine20200612.csv': r'Z:\Shippinglab\20200612',
    #             r'Z:\Shippinglab\mmdet\rgb_train.csv': r'Z:\Shippinglab\mmdet\2019_04_12_data'}

    csvs_path = {
        r"Z:\Shippinglab\Raymarine20200611.csv": r"/home/jbibe/sftp/transfer/20200611",
        r"Z:\Shippinglab\Raymarine20200612.csv": r"/home/jbibe/sftp/transfer/20200612",
        r"Z:\Shippinglab\mmdet\rgb_train.csv": r"/home/jbibe/sftp/transfer/mmdet/2019_04_12_data",
    }

    main(csvs_path)
