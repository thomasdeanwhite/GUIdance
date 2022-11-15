import config as cfg
import pickle
import os
from cv2 import imread, resize
import numpy as np
import math
import random
import cv2

debug = False
use_pickle = False

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
        #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def convert_coords(x, y, w, h, aspect):
    if aspect > 1: # width is bigger than height
        h = h * aspect
        y = 0.5 + ((y - 0.5)*aspect)
    elif aspect < 1:
        aspect = 1 / aspect
        w = w * aspect
        x = 0.5 + ((x - 0.5)*aspect)

    return x, y, w, h

def edge_detection(img):
    return cv2.Canny(img.astype(np.uint8), 0, 255)

def disable_transformation():
    cfg.brightness_probability = 0
    cfg.contrast_probability = 0
    cfg.invert_probability = 0
    cfg.edge_detection_probability = 0

def normalise_point(point, val):
    i = point*val
    return i, math.floor(i)

def normalise_label(label):
    px, cx = normalise_point(max(0, min(0.999, label[0])), cfg.grid_shape[0])
    py, cy = normalise_point(max(0, min(0.999, label[1])), cfg.grid_shape[1])
    return [
               px,
               py,
               max(0.05, min(cfg.grid_shape[0], label[2]*cfg.grid_shape[0])),
               max(0.05, min(cfg.grid_shape[1], label[3]*cfg.grid_shape[1])),
               label[4]
           ], (cx, cy)

def pad_image(img):
    image = np.int16(img)

    height, width = image.shape

    if width == 0:
        return img

    aspect = height/width

    padding_x = 0
    padding_y = 0

    if aspect > 1: # portrait
        padding_x = round((height-width)/2)
        elements = np.transpose(np.random.rand(padding_x, image.shape[0])*255)
        image = np.append(elements, image, 1)
        elements = np.transpose(np.random.rand(padding_x, image.shape[0])*255)
        image = np.append(image, elements, 1)
    elif aspect < 1: #landscape
        padding_y = round((width-height)/2)
        #for i in range(padding_y):
        elements = np.random.rand(padding_y, image.shape[1])*255
        image = np.append(elements, image, 0)
        elements = np.random.rand(padding_y, image.shape[1])*255
        image = np.append(image, elements, 0)

    image = np.uint8(resize(image, (cfg.width, cfg.height)))
    image = np.reshape(image, [cfg.width, cfg.height, 1])

    if debug:
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image

def load_raw_image(file):
    img = imread(file)
    return img

def load_image(file):
    img = imread(file, 0)
    image = np.int16(pad_image(img))

    # contrast = 0.5
    #
    # contrast_diff = np.multiply(image - np.median(image), contrast).astype(np.uint8)
    # image = np.add(image, contrast_diff)

    image = np.uint8(resize(image, (cfg.width, cfg.height)))

    image = np.reshape(image, [cfg.width, cfg.height, 1])

    return image


def load_files(raw_files):
    raw_files = [f.replace("/data/acp15tdw", "/home/thomas/experiments") for f in raw_files]
    label_files = [f.replace("/images/", "/labels/") for f in raw_files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    pickle_files = [f.replace("/images/", "/pickle/") for f in raw_files]
    pickle_files = [f.replace(".png", ".pickle") for f in pickle_files]

    images = []
    labels = []
    object_detection = []

    for i in range(len(raw_files)):
        pickle_f = pickle_files[i]


        pickled_data = []

        if os.path.isfile(pickle_f) and use_pickle:
            pickled_data = pickle.load(open(pickle_f, "rb"))
            images.append(pickled_data[0])
            labels.append(pickled_data[1])
            object_detection.append(pickled_data[2])
        else:
            f = raw_files[i]
            f_l = label_files[i]
            if not os.path.isfile(f_l) or not os.path.isfile(f) or f is None:
                continue

            img = imread(f, 0)
            if img is None:
                continue
            image = np.int16(img)

            height, width = image.shape

            if height < 15 or width < 15:
                continue

            aspect = height/width

            padding_x = 0
            padding_y = 0

            if aspect > 1: # portrait
                padding_x = round((height-width)/2)
                elements = np.transpose(np.random.rand(padding_x, image.shape[0])*255)
                image = np.append(elements, image, 1)
                elements = np.transpose(np.random.rand(padding_x, image.shape[0])*255)
                image = np.append(image, elements, 1)
            else: #landscape
                padding_y = round((width-height)/2)
                #for i in range(padding_y):
                elements = np.random.rand(padding_y, image.shape[1])*255
                image = np.append(elements, image, 0)
                elements = np.random.rand(padding_y, image.shape[1])*255
                image = np.append(image, elements, 0)

            image = np.uint8(resize(image, (cfg.width, cfg.height)))
            image = np.reshape(image, [cfg.width, cfg.height, 1])
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

                for x in range(cfg.grid_shape[0]):
                    for y in range(cfg.grid_shape[1]):
                        imglabs[x][y][0] = y
                        imglabs[x][y][1] = x

                for line in l:
                    elements = line.split(" ")
                    #print(elements[1:3])

                    if float(elements[3]) <= 0 or float(elements[4]) <= 0:
                        continue

                    normalised_label, centre = normalise_label([float(elements[1]), float(elements[2]),
                                                                float(elements[3]), float(elements[4]), 1])

                    if padding_x > 0:
                        ratio = 1-(2*padding_x)/height
                        newx = cfg.grid_shape[0] * (0.5 + (normalised_label[0]/cfg.grid_shape[0] - 0.5)*ratio)
                        normalised_label[0] = newx
                        normalised_label[2] = normalised_label[2] * ratio
                    elif padding_y > 0:
                        ratio = 1-(2*padding_y)/width
                        newy = cfg.grid_shape[1] * (0.5 + (normalised_label[1]/cfg.grid_shape[1] - 0.5)*ratio)
                        normalised_label[1] = newy
                        normalised_label[3] = normalised_label[3] * ratio

                    x = max(0, min(int(normalised_label[0]), cfg.grid_shape[0]-1))
                    y = max(0, min(int(normalised_label[1]), cfg.grid_shape[1]-1))
                    add = True
                    if imglabs[y][x][4] == 1:
                        add = False
                        if normalised_label[2] * normalised_label[3] > imglabs[y][x][2] * imglabs[y][x][3]:
                            add = True
                    if add:
                        imglabs[y][x] = normalised_label
                        obj_detect[y][x] = int(elements[0])
                        #obj_detect[y][x][int(elements[0])] = 1

                object_detection.append(obj_detect)
                labels.append(imglabs)

            pickled_data = [image, imglabs, obj_detect]

            pickle.dump(pickled_data, open(pickle_f, "wb"))

    for im in range(len(images)):
        image = images[im]

        labs = labels[im]

        height, width, channels = image.shape

        image = image.astype(np.int16)
        if random.random() < cfg.brightness_probability:



            brightness = int((random.random()*cfg.brightness_var*2)-cfg.brightness_var)
            image = np.add(image, brightness)

            image = np.maximum(0, np.minimum(255, image))

        if random.random() < cfg.contrast_probability:
            contrast = (random.random() * cfg.contrast_var * 2) - cfg.contrast_var

            contrast_diff = np.multiply(image - np.mean(image), contrast).astype(np.int16)
            image = np.maximum(0, np.minimum(255, np.add(image, contrast_diff)))

        if random.random() < cfg.invert_probability:
            image = 255 - image

        if random.random() < cfg.edge_detection_probability:
            image = edge_detection(image)

        if random.random() < cfg.dimension_probability:
            # crop or padd image
            rand_num = random.random()
            if rand_num < 0.3333: #pad

                new_width = int(random.random() * (width) + width)
                new_height = int(random.random() * (height) + height)

                new_x = int(random.random() * (new_width-width))
                new_y = int(random.random() * (new_height-height))

                img = np.zeros([new_height, new_width, channels]).astype(np.uint8) + int(np.median(image))

                img[new_y:new_y+height, new_x:new_x+width] = image

                image = img

                mult_x = width / new_width
                mult_y = height / new_height

                obj_det = np.copy(object_detection[im])
                labs_copy = np.copy(labels[im])

                for lnx in range(len(labs)):
                    for lny in range(len(labs[lnx])):
                        object_detection[im][lnx][lny] = 0
                        labels[im][lnx][lny] = [0, 0, 0, 0, 0]

                for lnx in range(len(labs)):
                    for lny in range(len(labs[lnx])):

                        lbl = labs_copy[lnx][lny]
                        if lbl[4] == 1:
                            lbl[0] = max(0, min(cfg.grid_shape[0]-1, (new_x/new_width + (lbl[0] / cfg.grid_shape[0] * mult_x)) * cfg.grid_shape[0]))
                            lbl[1] = max(0, min(cfg.grid_shape[1]-1, (new_y/new_height + (lbl[1] / cfg.grid_shape[1] * mult_y)) * cfg.grid_shape[1]))
                            lbl[2] = lbl[2] * mult_x
                            lbl[3] = lbl[3] * mult_y

                            object_detection[im][int(lbl[1])][int(lbl[0])] = obj_det[lnx][lny]
                            labels[im][int(lbl[1])][int(lbl[0])] = lbl

            elif rand_num < 0.6666: #crop
                new_width = int((random.random() * width/2)+width*0.75)
                new_height = int((random.random() * height/2)+height*0.75)

                new_x = int(random.random() * width*0.25)
                new_y = int(random.random() * height*0.25)

                img = np.zeros([new_height, new_width, channels]).astype(np.uint8) + int(np.median(image))

                max_x = min(new_x + new_width, width) - new_x
                max_y = min(new_y + new_height, height) - new_y

                img[0:max_y, 0:max_x] = image[new_y:new_y+max_y, new_x:new_x+max_x]

                image = img

                mult_x = width / new_width
                mult_y = height / new_height

                obj_det = np.copy(object_detection[im])
                labs_copy = np.copy(labels[im])

                for lnx in range(len(labs)):
                    for lny in range(len(labs[lnx])):
                        object_detection[im][lnx][lny] = 0
                        labels[im][lnx][lny] = [0, 0, 0, 0, 0]

                for lnx in range(len(labs)):
                    for lny in range(len(labs[lnx])):

                        lbl = labs_copy[lnx][lny]
                        if lbl[4] == 1:
                            lbl[0] = ((lbl[0] / cfg.grid_shape[0] * mult_x)-new_x/new_width) * cfg.grid_shape[0]
                            lbl[1] = ((lbl[1] / cfg.grid_shape[1] * mult_y)-new_y/new_height) * cfg.grid_shape[1]
                            lbl[2] = lbl[2] * mult_x
                            lbl[3] = lbl[3] * mult_y

                            if (lbl[0] >= 0 and lbl[1] >= 0 and lbl[0] < cfg.grid_shape[0] and lbl[1] < cfg.grid_shape[1] ): # check if element is in screenshot
                                object_detection[im][int(lbl[1])][int(lbl[0])] = obj_det[lnx][lny]
                                labels[im][int(lbl[1])][int(lbl[0])] = lbl

            else: # tile
                new_width = width * 2
                new_height = height * 2

                img = np.zeros([new_height, new_width, channels]).astype(np.uint8) + int(np.median(image))

                copy_chance = 0.9

                copies = [random.random() < copy_chance, random.random() < copy_chance, random.random() < copy_chance, random.random() < copy_chance]

                if copies[0]:
                    img[:height, :width] = np.copy(image)

                if copies[1]:
                    img[height:, :width] = np.copy(image)

                if copies[2]:
                    img[:height, width:] = np.copy(image)

                if copies[3]:
                    img[height:, width:] = np.copy(image)

                image = img

                obj_det = np.copy(object_detection[im])
                labs_copy = np.copy(labels[im])

                for lnx in range(len(labs)):
                    for lny in range(len(labs[lnx])):
                        object_detection[im][lnx][lny] = 0
                        labels[im][lnx][lny] = [0, 0, 0, 0, 0]

                disp_x = cfg.grid_shape[0]/2
                disp_y = cfg.grid_shape[1]/2

                for lnx in range(len(labs)):
                    for lny in range(len(labs[lnx])):
                        lbl = labs_copy[lnx][lny] / 2
                        lbl[4] = lbl[4] * 2
                        if lbl[4] == 1:

                            x, y = (lbl[0], lbl[1])

                            if copies[0]:
                                labels[im][int(y)][int(x)] = np.copy(lbl)
                                object_detection[im][int(y)][int(x)] = obj_det[lnx][lny]

                            if copies[1]:
                                lbc2 = np.copy(lbl)
                                lbc2[1] += disp_y
                                labels[im][int(lbc2[1])][int(lbc2[0])] = lbc2
                                object_detection[im][int(lbc2[1])][int(lbc2[0])] = obj_det[lnx][lny]

                            if copies[2]:
                                lbc2 = np.copy(lbl)
                                lbc2[0] += disp_x
                                labels[im][int(lbc2[1])][int(lbc2[0])] = lbc2
                                object_detection[im][int(lbc2[1])][int(lbc2[0])] = obj_det[lnx][lny]

                            if copies[3]:
                                lbc2 = np.copy(lbl)
                                lbc2[0] += disp_x
                                lbc2[1] += disp_y
                                labels[im][int(lbc2[1])][int(lbc2[0])] = lbc2
                                object_detection[im][int(lbc2[1])][int(lbc2[0])] = obj_det[lnx][lny]

        image = image.astype(np.uint8)

        image = cv2.resize(image, (cfg.height, cfg.width))

        image = np.reshape(image, [cfg.width, cfg.height, 1])

        images[im] = image

        if debug:
            height, width, chan = image.shape

            for lnx in range(len(labs)):
                for lny in range(len(labs[lnx])):

                    lbl = labs[lnx][lny]

                    col = 0

                    if (lbl[4] > 0 and object_detection[im][lnx][lny] > 0):
                        col = 255

                    lbl[0] = lbl[0] / cfg.grid_shape[0]
                    lbl[1] = lbl[1] / cfg.grid_shape[1]
                    lbl[2] = lbl[2] / cfg.grid_shape[0]
                    lbl[3] = lbl[3] / cfg.grid_shape[1]
                    x1, y1 = (int(width * (lbl[0] - lbl[2]/2)),
                              int(height * (lbl[1] - lbl[3]/2)))
                    x2, y2 = (int(width * (lbl[0] + lbl[2]/2)),
                              int(height * (lbl[1] + lbl[3]/2)))
                    cv2.rectangle(image,
                                  (x1, y1),
                                  (x2, y2),
                                  127, 3, 4)

                    cv2.rectangle(image,
                                  (int((x1+x2)/2-1), int((y1+y2)/2-1)),
                                  (int((x1+x2)/2+1), int((y1+y2)/2+1)),
                                  127, 3, 4)

                    cv2.rectangle(image,
                                  (int(lny/cfg.grid_shape[0]*width), int(lnx/cfg.grid_shape[0]*height)),
                                  (int((1+lny)/cfg.grid_shape[1]*width), int((1+lnx)/cfg.grid_shape[1]*height)),
                                  col, 1, 4)

            cv2.imshow('image',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return images, labels, object_detection



def load_raw_labels(raw_files):
    raw_files = [f.replace("/data/acp15tdw", "/home/thomas/experiments") for f in raw_files]
    label_files = [f.replace("/images/", "/labels/") for f in raw_files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    labels = []

    for i in range(len(raw_files)):

        f = raw_files[i]
        f_l = label_files[i]
        if not os.path.isfile(f_l) or not os.path.isfile(f) or f is None:
            continue


        # read in format [c, x, y, width, height]
        # store in format [c], [x, y, width, height]

        with open(f_l, "r") as l:
            for line in l:
                elements = line.split(" ")
                #print(elements[1:3])

                if float(elements[3]) <= 0 or float(elements[4]) <= 0:
                    continue

                normalised_label, centre = normalise_label([float(elements[1]), float(elements[2]),
                                                            float(elements[3]), float(elements[4]), 1])
                labels.append(normalised_label)


    return labels



def load_file_raw(raw_file):
    raw_file = raw_file.replace("/data/acp15tdw", "/home/thomas/experiments")
    label_file = raw_file.replace("/images/", "/labels/")
    label_file = label_file.replace(".png", ".txt")
    labels = []
    object_detection = []

    f = raw_file
    f_l = label_file
    if not os.path.isfile(f_l) or not os.path.isfile(f) or f is None:
        return [], [], []

    img = imread(f, 1)
    if img is None:
        return [], [], []
    image = np.int16(img)

    height, width = image.shape[:2]

    if height < 15 or width < 15:
        return [], [], []

    images = image.astype(np.uint8)



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

        for x in range(cfg.grid_shape[0]):
            for y in range(cfg.grid_shape[1]):
                imglabs[x][y][0] = y
                imglabs[x][y][1] = x

        for line in l:
            elements = line.split(" ")
            #print(elements[1:3])

            if float(elements[3]) <= 0 or float(elements[4]) <= 0:
                continue

            normalised_label, centre = normalise_label([float(elements[1]), float(elements[2]),
                                                        float(elements[3]), float(elements[4]), 1])

            x = max(0, min(int(normalised_label[0]), cfg.grid_shape[0]-1))
            y = max(0, min(int(normalised_label[1]), cfg.grid_shape[1]-1))
            imglabs[y][x] = normalised_label
            obj_detect[y][x] = int(elements[0])

        object_detection = obj_detect
        labels = imglabs

    return images, labels, object_detection


def load_files_split(raw_files, scale):
    raw_files = [f.replace("/data/acp15tdw", "/home/thomas/experiments") for f in raw_files]
    label_files = [f.replace("/images/", "/labels/") for f in raw_files]
    label_files = [f.replace(".png", ".txt") for f in label_files]

    pickle_files = [f.replace("/images/", "/pickle/") for f in raw_files]
    pickle_files = [f.replace(".png", ".pickle") for f in pickle_files]

    images = []
    labels = []
    object_detection = []

    img_sizes = []

    for i in range(len(raw_files)):

        f = raw_files[i]
        f_l = label_files[i]
        if not os.path.isfile(f_l) or not os.path.isfile(f) or f is None:
            continue

        img = imread(f, 0)
        if img is None:
            continue
        image = np.int16(img)

        height, width = image.shape

        img_sizes.append([width, height])

        if height < 15 or width < 15:
            continue

        aspect = height/width

        padding_x = 0
        padding_y = 0

        if aspect > 1: # portrait
            padding_x = round((height-width)/2)
            for i in range(padding_x):
                elements = np.transpose(np.expand_dims((np.random.rand(image.shape[0])*255), 0))
                image = np.append(elements, image, 1)
            for i in range(padding_x):
                elements = np.transpose(np.expand_dims((np.random.rand(image.shape[0])*255), 0))
                image = np.append(image, elements, 1)
        else: #landscape
            padding_y = round((width-height)/2)
            for i in range(padding_y):
                elements = np.transpose(np.expand_dims((np.random.rand(image.shape[1])*255), 1))
                image = np.append(elements, image, 0)
            for i in range(padding_y):
                elements = np.transpose(np.expand_dims((np.random.rand(image.shape[1])*255), 1))
                image = np.append(image, elements, 0)

        image = np.uint8(resize(image, (cfg.width*scale, cfg.height*scale)))
        image = np.reshape(image, [cfg.width*scale, cfg.height*scale, 1])
        images.append(image)



        # read in format [c, x, y, width, height]
        # store in format [c], [x, y, width, height]

        with open(f_l, "r") as l:
            obj_detect = [[0 for i in
                           range(cfg.grid_shape[0]*scale)]for i in
                          range(cfg.grid_shape[1]*scale)]
            imglabs = [[[0 for i in
                         range(5)]for i in
                        range(cfg.grid_shape[1]*scale)] for i in
                       range(cfg.grid_shape[0]*scale)]

            for x in range(cfg.grid_shape[0]*scale):
                for y in range(cfg.grid_shape[1]*scale):
                    imglabs[x][y][0] = y
                    imglabs[x][y][1] = x

            for line in l:
                elements = line.split(" ")
                #print(elements[1:3])

                if float(elements[3]) <= 0 or float(elements[4]) <= 0:
                    continue

                normalised_label, centre = normalise_label([float(elements[1]), float(elements[2]),
                                                            float(elements[3]), float(elements[4]), 1])

                for icl in range(4):
                    normalised_label[icl] = normalised_label[icl] * scale

                if padding_x > 0:
                    ratio = 1-(2*padding_x)/height
                    newx = scale * cfg.grid_shape[0] * (0.5 + (normalised_label[0]/(cfg.grid_shape[0]*scale) - 0.5)*ratio)
                    normalised_label[0] = newx
                    normalised_label[2] = normalised_label[2] * ratio
                elif padding_y > 0:
                    ratio = 1-(2*padding_y)/width
                    newy = scale * cfg.grid_shape[1] * (0.5 + (normalised_label[1]/(cfg.grid_shape[1]*scale) - 0.5)*ratio)
                    normalised_label[1] = newy
                    normalised_label[3] = normalised_label[3] * ratio

                x = max(0, min(int(normalised_label[0]), scale*cfg.grid_shape[0]-1))
                y = max(0, min(int(normalised_label[1]), scale*cfg.grid_shape[1]-1))
                imglabs[y][x] = normalised_label
                obj_detect[y][x] = int(elements[0])
                #obj_detect[y][x][int(elements[0])] = 1

            object_detection.append(obj_detect)
            labels.append(imglabs)

    img_length = len(images)

    images, labels, object_detection = (np.array(images), np.array(labels), np.array(object_detection))

    raw_data = load_files(raw_files)

    s_images, s_labels, s_object_detection = ([], [], [])

    for im in range(img_length):

        if (img_sizes[im][0] < cfg.width and img_sizes[im][1] < cfg.height):#don't split
            s_images.append(raw_data[0][im])
            s_labels.append(raw_data[1][im])
            s_object_detection.append(raw_data[2][im])
            continue

        image = images[im]

        labs = labels[im]

        obj = object_detection[im]

        for ic in range(scale):
            for jc in range(scale):
                xl, xu = (cfg.grid_shape[0]*ic,cfg.grid_shape[0]*(ic+1))
                yl, yu = (cfg.grid_shape[1]*jc,cfg.grid_shape[1]*(jc+1))
                sub_img = image[32*xl:32*xu, 32*yl:32*yu]
                sub_labs = labs[xl:xu, yl:yu]
                sub_labs[..., 0] = (sub_labs[..., 0] - (yl))
                sub_labs[..., 1] = (sub_labs[..., 1] - (xl))

                sub_obj = obj[xl:xu, yl:yu]

                s_images.append(sub_img)
                s_labels.append(sub_labs)
                s_object_detection.append(sub_obj)

    for im in range(len(s_images)):

        image = s_images[im]

        labs = s_labels[im]

        if debug:
            height, width, chan = image.shape

            for lnx in range(len(labs)):
                for lny in range(len(labs[lnx])):

                    lbl = labs[lnx][lny]

                    col = 0

                    thickness = 1

                    if lbl[4] > 0:
                        col = 255
                        thickness = 4

                    lbl[0] = lbl[0] / cfg.grid_shape[0]
                    lbl[1] = lbl[1] / cfg.grid_shape[1]
                    lbl[2] = lbl[2] / cfg.grid_shape[0]
                    lbl[3] = lbl[3] / cfg.grid_shape[1]
                    x1, y1 = (int(width * (lbl[0] - lbl[2]/2)),
                              int(height * (lbl[1] - lbl[3]/2)))
                    x2, y2 = (int(width * (lbl[0] + lbl[2]/2)),
                              int(height * (lbl[1] + lbl[3]/2)))
                    cv2.rectangle(image,
                                  (x1, y1),
                                  (x2, y2),
                                  127, 3, 4)

                    cv2.rectangle(image,
                                  (int((x1+x2)/2-1), int((y1+y2)/2-1)),
                                  (int((x1+x2)/2+1), int((y1+y2)/2+1)),
                                  127, 3, 4)

                    cv2.rectangle(image,
                                  (int(lny/cfg.grid_shape[0]*width), int(lnx/cfg.grid_shape[0]*height)),
                                  (int((1+lny)/cfg.grid_shape[1]*width), int((1+lnx)/cfg.grid_shape[1]*height)),
                                  col, thickness, 4)

            cv2.imshow('image',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return s_images, s_labels, s_object_detection