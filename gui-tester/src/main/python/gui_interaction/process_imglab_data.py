import config as cfg
import os
import xmltodict
from yolo import Yolo
from shutil import copy

if __name__ == '__main__':

    class_mapping = {  # map annotated class to actual class names
        "textfield": "text_field",
        "textbox": "text_field",
        "button": "button",
        "combobox": "combo_box",
        "tree": "tree",
        "list": "list",
        "scrollbar": "scroll_bar",
        "menuitem": "menu_item",
        "menu": "menu",
        "togglebutton": "toggle_button",
        "tabs": "tabs",
        "slider": "slider",
        "menuww": "menu",

    }

    yolo = Yolo()

    real_folder = cfg.data_dir + "/../mac"
    unproc_folder = real_folder + "/unproc"
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(unproc_folder + "/images")) for f in fn]

    print(files)

    label_files = [(unproc_folder + "/labels/" + x[x.rfind("/") + 1:-4] + ".xml") for x in files]

    for i in range(len(files)):
        img_exists =  os.path.isfile(files[i])
        labs_exists = os.path.isfile(label_files[i])

        if img_exists and labs_exists:

            os.system("cp \"" + files[i] + "\" " + real_folder + "/data/images/" + str(i) + ".png")

            with open(real_folder + "/data/labels/" + str(i) + ".txt", "w+") as label_output_file:
                with open(label_files[i], "r") as fo:
                    xml = xmltodict.parse(fo.read())['annotation']

                    size = xml['size']

                    width, height = (int(size['width']), int(size['height']))

                    objects = xml['object']

                    if not type(objects) == list: # only 1 widget
                        objects = [objects]

                    for o in range(len(objects)):
                        obj = objects[o]
                        #print(obj)
                        clazz = obj['name']

                        if not clazz in class_mapping:
                            print("Class",clazz,"not supported!")
                            continue

                        class_index = yolo.names.index(class_mapping[clazz])

                        box = obj['bndbox']
                        x1, y1 = (int(box['xmin']), int(box['ymin']))
                        x2, y2 = (int(box['xmax']), int(box['ymax']))

                        x, y = ((x1+x2)/2, (y1+y2)/2)
                        w, h = (x2-x1, y2-y1)

                        line = str(class_index) + " " + str(x/width) + " " + str(y/height) + " " + str(w/width) + " " + str(h/height) + "\n"

                        label_output_file.write(line)

    print("Finished Exporting Files")
