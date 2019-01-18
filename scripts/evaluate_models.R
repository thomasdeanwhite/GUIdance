library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(RColorBrewer)
library(ggthemes)

args = commandArgs(TRUE)

load_data <- function(directory, file){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv(file)
  return(data)
}

setwd('/home/thomas/work/GUIdance')

data <- load_data('/home/thomas/work/GUIdance', 'errors.csv')

p = data  %>% 
  #filter(correct==1) %>%
  ggplot(aes(x=error_type, y=percentage, fill=error_type)) +
  geom_bar(stat="sum") +
  #geom_smooth(method="lm", se=F) +
  #scale_y_log10() +
  labs(x="Error Type",
       y="Quantity",
       title="Error Type Plot") +
  #scale_x_discrete() +
  #scale_y_log10() +
  facet_wrap(~class) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_colour_colorblind()
#scale_fill_brewer(palette="Set1") + 
#scale_color_brewer(palette="BrBG")

ggsave("error.png", p, height=5, width=5, dpi=150)


data <- load_data('/home/thomas/work/GUIdance', 'validation.csv')

#data$iou_threshold = data$iou_threshold * 2 + 0.5

data$iou_threshold = factor(data$iou_threshold)

#data = data[!data$class == "menu_item",]

#data = data[!data$dataset == "real",]

data$class = "Widget"

average_precision = data[FALSE,]

classes = unique(data$class)
datasets = unique(data$dataset)
iou_thresholds = unique(data$iou_threshold)

# p = data  %>%
#   #filter(iou_threshold == 0.5) %>%
#   #filter(correct==1) %>%
#   ggplot(aes(x=dataset, y=precision, fill=dataset, )) +
#   geom_boxplot() +
#   #geom_bar(position = "dodge", stat = "summary", fun.y = "mean") +
#   #geom_smooth(method="lm", se=F) +
#   #scale_y_log10() +
#   labs(x="Dataset",
#        y="Precision",
#        title="") +
#   #scale_x_discrete() +
#   #scale_y_log10() +
#   #facet_wrap(~class) +
#   theme_minimal() +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
#   scale_fill_grey(start=0.6, end=1.0)
#   #scale_fill_brewer(palette="Set1") +
#   #scale_color_brewer(palette="BrBG")

t_data = data %>% gather(Sensitivity, value, precision:recall)

t_data$size_cat = factor(data$size_cat, levels=c('xs', 's', 'l', 'xl'))

t_data$busy_cat = factor(data$busy_cat, levels=c('desolate', 'few', 'many', 'crowded'))

p = t_data  %>%
  #filter(iou_threshold == 0.5) %>%
  #filter(correct==1) %>%
  #group_by(size_cat) %>%
  ggplot(aes(x=Sensitivity, y=value, fill=dataset)) +
  geom_boxplot() +
  #geom_bar(position = "dodge", stat = "summary", fun.y="sum") +
  #geom_smooth(method="lm", se=F) +
  #scale_y_log10() +
  labs(x="Image Size",
       y="",
       title="") +
  #scale_x_discrete() +
  #scale_y_log10() +
  facet_wrap(~class) +
  theme_minimal() +
  #facet_wrap(busy_cat~size_cat) +
  #theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_grey(start=0.3, end=0.6)
#scale_fill_brewer(palette="Set1") +
#scale_color_brewer(palette="BrBG")

print(p)

for (i in 0:11){
  limit = i * 0.1
  for (c in classes){
    for (d in datasets){
      for (i in iou_thresholds){
        class_rows = data %>% filter(class == c, dataset == d, iou_threshold == i)
        if (nrow(class_rows) > 0){
          row = head(class_rows, 1)
          
          row$recall = limit
          
          above_rows = class_rows %>% filter(recall >= limit)
          
          if (nrow(above_rows) > 0){
            row$precision = max(above_rows %>% select(precision))
          } else {
            row$precision = 0
          }
          
          average_precision = bind_rows(average_precision, row)
        }
      }
    }
  }
}

for (d in datasets){
  mAP = 0
  for (i in iou_thresholds){
    class_precision = 0

    for (c in classes){

      set_data = average_precision %>% filter(class == c, dataset == d, iou_threshold == i)

      if (nrow(set_data) > 0){
        class_precision = class_precision + mean(set_data$precision)
      }
    }

    class_precision = class_precision / length(classes)

    print(paste("AP", i, "(", d, "):", class_precision))

    mAP = mAP + class_precision
  }

  mAP = mAP / length(iou_thresholds)

  print(paste("mAP(", d, "):", mAP))

}

# 
# data <- load_data('/home/thomas/work/GUIdance', 'confusion.csv')
# 
# #data = data[!data$dataset == "real",]
# 
# data = data %>% group_by(actual_class) %>% mutate(sum = sum(quantity)+1)
# 
# data$percent = data$quantity/data$sum
# 
# #data$quantity = data$quantity/max(data$quantity)
# 
# p = data %>% ggplot(aes(predicted_class, actual_class)) +
#   geom_tile(aes(fill=percent)) +
#   labs(x="Predicted Class",
#        y="Actual Class",
#        title="Confusion Matrix for Class Prediction") +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
#   scale_fill_gradient(low = "#0000FF", high = "#FF0000", na.value = "#00FF00") +
#   facet_wrap(~dataset)
# 
# ggsave("confusion.png", p, height=5, width=5, dpi=150)

