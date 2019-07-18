import config as cfg
import sys
from data_loader import load_image, load_raw_image, disable_transformation, convert_coords
import json

if __name__ == '__main__':

    disable_transformation()

    raw_img = load_raw_image(sys.argv[1]).tolist()

    print(json.dumps(raw_img).replace("]],", "]],\n"))
