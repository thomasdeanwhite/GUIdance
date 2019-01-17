import config as cfg
from yolo import Yolo
from cv2 import imread, resize
import numpy as np
import tensorflow as tf
import sys
import gc
import math
import random
import os
import pickle
import re
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

totals = []
yolo = Yolo()

def normalise_point(point, val):
    i = point*val
    return i, math.floor(i)

def normalise_label(label):
    px, cx = normalise_point(max(0, min(1, label[0])), cfg.grid_shape[0])
    py, cy = normalise_point(max(0, min(1, label[1])), cfg.grid_shape[1])
    return [
               px,
               py,
               max(0, min(cfg.grid_shape[0], label[2]*cfg.grid_shape[0])),
               max(0, min(cfg.grid_shape[1], label[3]*cfg.grid_shape[1])),
               label[4]
           ], (cx, cy)

def load_files(raw_files):
    raw_files = [f.replace("/data/acp15tdw", "/home/thomas/experiments") for f in raw_files]
    label_files = [f.replace("/images/", "/labels/") for f in raw_files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    pickle_files = [f.replace("/images/", "/pickle/") for f in raw_files]
    pickle_files = [f.replace(".png", ".pickle") for f in pickle_files]

    areas = []
    images = []
    labels = []
    object_detection = []

    for i in range(len(raw_files)):
        f = raw_files[i]
        f_l = label_files[i]
        d = imread(f, 0)

        if not os.path.isfile(f_l) or not os.path.isfile(f) or d is None:
            continue

        image = np.int16(d)

        height, width = image.shape

        areas.append(height*width)

        images.append(image)

        # read in format [c, x, y, width, height]
        # store in format [c], [x, y, width, height]
        with open(f_l, "r") as l:
            obj_detect = [[0 for i in
                           range(cfg.grid_shape[0])]for i in
                          range(cfg.grid_shape[1])]
            imglabs = [[[0 for i in
                         range(5)]for i in
                        range(cfg.grid_shape[1])] for i in
                       range(cfg.grid_shape[0])]

            for line in l:
                elements = line.split(" ")
                #print(elements[1:3])
                normalised_label, centre = normalise_label([float(elements[1]), float(elements[2]),
                                                            float(elements[3]), float(elements[4]), 1])
                x = max(0, min(int(centre[0]), cfg.grid_shape[0]-1))
                y = max(0, min(int(centre[1]), cfg.grid_shape[1]-1))
                imglabs[y][x] = normalised_label
                obj_detect[y][x] = int(elements[0])
                #obj_detect[y][x][int(elements[0])] = 1

            object_detection.append(obj_detect)
            labels.append(imglabs)

    return images, labels, object_detection, areas

def proc_subset(imgs, labels, classes, dataset, areas):
    density = np.array([[[0 for i in
                          range(len(yolo.names)+1)]for i in
                         range(cfg.grid_shape[1])] for i in
                        range(cfg.grid_shape[0])])

    pixels = np.array([0 for i in
                       range(30)])
    quantities = []

    lim = imgs.shape[0]

    last_prog = 0


    for i in range(lim):

        c_prog = i/float(lim)
        if c_prog >= last_prog:
            print(last_prog * 100, "%")
            last_prog += 0.1

        widget_q = 0

        class_count = np.array([0 for i in
                                range(len(yolo.names))])

        displacement = 255/30

        widget_area = 0
        image_area = areas[i]

        img = imgs[i]

        for j in range(30):
            ub = (j+1)*displacement
            lb = j*displacement
            quantity = np.sum((img < ub).astype(np.int32) *
                              (img > lb).astype(np.int32)).astype(np.float32)
            pixels[j] += quantity


        for x in range(cfg.grid_shape[0]):
            for y in range(cfg.grid_shape[1]):
                if (labels[i,x,y,4] == 1):
                    c = classes[i,x,y]
                    density[y,x,0] = density[x,y,0] + 1
                    density[y,x,c+1] = density[x,y,c+1] + 1
                    class_count[c] += 1

                    widget_q += 1

                    with open("label_dims.csv", "a") as file:
                        area = str(labels[i,x,y,2]/cfg.grid_shape[0] * labels[i,x,y,3]/cfg.grid_shape[1])
                        file.write(str(i) + "," + yolo.names[c] + ",width," + str(labels[i,x,y,2]/cfg.grid_shape[0]) + "," + dataset + "\n")
                        file.write(str(i) + "," + yolo.names[c] + ",height," + str(labels[i,x,y,3]/cfg.grid_shape[1]) + "," + dataset + "\n")
                        file.write(str(i) + "," + yolo.names[c] + ",area," + area + "," + dataset + "\n")
                        widget_area += image_area*labels[i,x,y,2]/cfg.grid_shape[0]*labels[i,x,y,3]/cfg.grid_shape[0]

        quantities.append(widget_q)

        with open("white_space.csv", "a") as file:
            file.write(str(i) + "," + str(image_area) + "," + str(np.sum(class_count)) + "," + str(widget_area) + "," + dataset + "\n")

        for c in range(len(class_count)):
            with open("class_count.csv", "a") as file:
                file.write(str(i) + "," + yolo.names[c] + "," + str(class_count[c]/np.sum(class_count)) + "," + dataset + "\n")

        return density, pixels, quantities

def print_info(imgs, labels, classes, dataset, areas):

    density, pixels, quantities = proc_subset(imgs, labels, classes, dataset, areas)

    return density, pixels, quantities

def run_dataset(files, dataset):

    density = np.array([[[0 for i in
                          range(len(yolo.names)+1)]for i in
                         range(cfg.grid_shape[1])] for i in
                        range(cfg.grid_shape[0])])

    pixels = np.array([0 for i in
                       range(30)])
    quantities = []

    max_run = 500

    iterations = 1 + int(len(files) / max_run)

    print(dataset, "set ---")

    for i in range(iterations):
        f_s = files[i*max_run:max((i+1)*max_run, len(files))]

        v_imgs, v_labels, v_obj_detection, areas = load_files(
            f_s)

        v_imgs = np.array(v_imgs)

        v_labels = np.array(v_labels)

        v_obj_detection = np.array(v_obj_detection)



        density_p, pixels_p, quantities_p = print_info(v_imgs, v_labels, v_obj_detection, dataset, areas)

        density += density_p
        pixels += pixels_p
        quantities += quantities_p

        del(v_imgs, v_labels, v_obj_detection)

    print("img_hist:", pixels)

    for j in range(30):
        with open("img_hist.csv", "a") as file:
            file.write(str(j) + "," + str(pixels[j]) + "," + dataset + "\n")

    for x in range(cfg.grid_shape[0]):
        for y in range(cfg.grid_shape[1]):
            for c in range(len(yolo.names)):
                with open("label_heat.csv", "a") as file:
                    dens = (density[x, y, c] + 1) / (np.amax(density[..., c]) + 1)

                    if (density[x, y, c] == 0):
                        dens = 0

                    cl = "total"
                    if (c > 0):
                        cl = yolo.names[c-1]
                    file.write(str(x) + "," + str(y) + "," + str(dens) + "," + cl + "," + dataset + "\n")



    areas = np.array(areas)

    med_area = np.median(areas)

    lower_quat_area = np.quantile(areas, 0.25)
    upper_quat_area = np.quantile(areas, 0.75)

    quantities = np.array(quantities)

    med_quantities = np.median(quantities)

    lower_quat_quantities = np.quantile(quantities, 0.25)
    upper_quat_quantities = np.quantile(quantities, 0.75)

    print("AREAS:", "BIG:", upper_quat_area, "MED:", med_area, "SMALL:", lower_quat_area)
    print("QUANTITIES:", "BIG:", upper_quat_quantities, "MED:", med_quantities, "SMALL:", lower_quat_quantities)

if __name__ == '__main__':
    training_file = cfg.data_dir + "/../backup/data/train.txt"

    valid_images = []

    real_images = []

    pattern = re.compile(".*\/([0-9]+).*")

    with open(training_file, "r") as tfile:
        for l in tfile:

            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())



    valid_file = cfg.data_dir + "/../backup/data/validate.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())


    valid_file = cfg.data_dir + "/../backup/data/test.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])

            if file_num <= 243:
                real_images.append(l.strip())

    real_images = [f.replace("/data/acp15tdw", "/data/acp15tdw/backup") for f in real_images]


    with open("img_hist.csv", "w") as file:
        file.write("pixel_value,quantity,dataset" + "\n")

    with open("label_heat.csv", "w") as file:
        file.write("x,y,density,class,dataset" + "\n")

    with open("class_count.csv", "w") as file:
        file.write("img,class,count,dataset" + "\n")

    with open("label_dims.csv", "w") as file:
        file.write("img,class,dimension,value,dataset" + "\n")

    with open("white_space.csv", "w+") as file:
        file.write("img,area,widget_count,widget_area,dataset\n")

    valid_file = cfg.data_dir + "/train.txt"

    with open(valid_file, "r") as tfile:
        for l in tfile:
            file_num = int(pattern.findall(l)[-1])
            valid_images.append(l.strip())

    valid_images = valid_images[:1000]

    valid_images = [f.replace("/home/thomas/work/GuiImages/public", "/data/acp15tdw/data") for f in valid_images]


    run_dataset(valid_images, "synthetic")


    run_dataset(real_images, "real")

    gc.collect()
