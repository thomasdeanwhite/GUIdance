

# ------------------------
# YOLO variables      |
# ------------------------

anchors = [0.9817857382941845, 9.398278193098788,
           5.52115919597547, 6.985948423888699,
           0.9090057538352261, 1.94398118062752,
           2.4060115242354794, 0.7252024765267888,
           9.922079927099324, 1.5033104111199442]

width = 416
height = 416

threshold = 0.1

cudnn_on_gpu = False

grid_shape = [13, 13]

data_dir = "/home/thomas/work/gui_image_identification/public"


# ------------------------
# input files            |
# ------------------------
names_file = "data.names"
train_file = "train.txt"
validate_file = "validate.txt"
test_file = "test.txt"

images_dir = "images"
labels_dir = "labels"

# ------------------------
# training variables     |
# ------------------------
learning_rate_start = 0.0001
learning_rate_min = 0.00001
learning_rate_decay = 0.95
momentum = 0.9
object_detection_threshold = 0.5

#maximum training epochs
epochs = 100000

batch_size = 32

var_sd = 0.001

# ------------------------
# output locations       |
# ------------------------

weights_dir = "weights"

output_dir = "/home/thomas/work/gui_image_identification/public/output"

results_dir = "results"
log_file = "loss.log"

output_dir = "/home/thomas/work/gui_image_identification/public/output"

results_dir = "results"
log_file = "loss.log"


# ------------------------
# training weights       |
# ------------------------

coord_weight = 1.0
obj_weight = 5.0
noobj_weight = 1.0
class_weight = 1.0
