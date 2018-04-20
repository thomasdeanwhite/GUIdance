

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
learning_rate_start = 1.0
learning_rate_min = 0.005
learning_rate_decay = 0.9995

#maximum training epochs
epochs = 5000000

batch_size = 10000

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

