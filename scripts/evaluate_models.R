library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(RColorBrewer)
library(ggthemes)

args = commandArgs(TRUE)

load_data <- function(directory){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv("validation.csv")
  return(data)
}

setwd('/home/thomas/work/GUIdance')

data <- load_data('/home/thomas/work/GUIdance')

data$iou_threshold = factor(data$iou_threshold)

#data = data[!data$class == "menu_item",]

#data$class = "Widget"

average_precision = data[FALSE,]

classes = unique(data$class)
datasets = unique(data$dataset)
iou_thresholds = unique(data$iou_threshold)

# for (i in 0:11){
#   limit = i * 0.1
#   for (c in classes){
#     for (d in datasets){
#       for (i in iou_thresholds){
#         class_rows = data %>% filter(class == c, dataset == d, iou_threshold == i)
#         if (nrow(class_rows) > 0){
#           row = head(class_rows, 1)
#           
#           row$recall = limit
#           
#           above_rows = class_rows %>% filter(recall >= limit)
#           
#           if (nrow(above_rows) > 0){
#             row$precision = max(above_rows %>% select(precision))
#           } else {
#             row$precision = 0
#           }
#           
#           average_precision = bind_rows(average_precision, row)
#         }
#       }
#     }
#   }
# }

p = data  %>% 
  #filter(correct==1) %>%
  ggplot(aes(x=recall, y=precision, color=dataset, lty=iou_threshold)) +
  geom_line() +
  #geom_smooth(method="lm", se=F) +
  #scale_y_log10() +
  labs(x="Recall",
       y="Precision",
       title="Average Precision Plots") +
  #scale_x_discrete() +
  #scale_y_log10() +
  facet_wrap(~class) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_colour_colorblind()
  #scale_fill_brewer(palette="Set1") + 
  #scale_color_brewer(palette="BrBG")

# for (d in datasets){
#   mAP = 0
#   for (i in iou_thresholds){
#     class_precision = 0
#     
#     for (c in classes){
#       
#       set_data = average_precision %>% filter(class == c, dataset == d, iou_threshold == i)
#       
#       if (nrow(set_data) > 0){
#         class_precision = class_precision + mean(set_data$precision)
#       }
#     }
#     
#     class_precision = class_precision / length(classes)
#     
#     print(paste("AP", i, "(", d, "):", class_precision))
#     
#     mAP = mAP + class_precision
#   }
#   
#   mAP = mAP / length(iou_thresholds)
#   
#   print(paste("mAP(", d, "):", mAP))
#   
# }

print(p)
