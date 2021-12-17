import re

f = open(r"csvs/full_sing.csv")
vl = f.readlines()
print(len(vl))
unique = {}
buoy_c = 0
boat_c = 0
for c in vl:
    v = c[29:].split(".")[0]

    boat = re.search(r"boat", c)
    buoy = re.search(r"buoy", c)
    kay = re.search(r"kayak", c)
    if buoy or kay:
        buoy_c += 1

    if v not in unique:
        if boat:
            unique[v] = ["boat"]
        elif buoy or kay:
            unique[v] = ["buoy"]

        else:
            print("hmm")
    elif v in unique:
        if boat:
            unique[v].append("boat")
        if buoy or kay:
            unique[v].append("buoy")
    else:
        print("yo")


b_b = 0
ann = 0
bu = 0
bu_a = 0
bo_a = 0

boBu, bo = [], []
for c in unique.keys():
    if "buoy" in unique[c]:
        boBu.append(c)
    else:
        bo.append(c)

import random

random.seed(0)
boBu_smpls = random.sample(boBu, int(len(boBu) / 15))

rw_sp = []
val_sp = []
for x in boBu_smpls:
    if random.random() < 0.05:
        rw_sp.append(x)
    else:
        val_sp.append(x)
bo_smpls = random.sample(bo, int(len(bo) / 25))

cntr = 0
for x in bo_smpls:
    if random.random() < 0.05:
        rw_sp.append(x)
    else:
        val_sp.append(x)

print(len(rw_sp))
print(len(val_sp))


val_set = []
train_set = []
rw_set = []

for x in vl:
    check = x[29:].split(".")[0]
    if check in val_sp:
        val_set.append(x)
    elif check in rw_sp:
        rw_set.append(x)
    else:
        train_set.append(x)

with open(r"csvs\train_set_sing.csv", "w") as f:
    for item in train_set:
        f.write("{}".format(item))
with open(r"csvs\val_set_sing.csv", "w") as f:
    for item in val_set:
        f.write("{}".format(item))
with open(r"csvs\rw_set_sing.csv", "w") as f:
    for item in rw_set:
        f.write("{}".format(item))
