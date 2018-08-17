library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(RColorBrewer)


args = commandArgs(TRUE)

load_data <- function(directory){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv("validation.csv")
  return(data)
}

setwd('/home/thomas/work/GUIdance')

data <- load_data('/home/thomas/work/GUIdance')

p = data %>%
  ggplot(aes(x=var, y=val, color=dataset)) +
  geom_boxplot() +
  #geom_smooth(method="lm", se=F) +
  #scale_y_log10() +
  labs(x="IoU Threshold",
       y="",
       title=paste("")) +
  #scale_x_discrete() +
  #scale_y_log10() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_brewer(palette="Set1") + 
  scale_color_brewer(palette="Dark2")

spread_data = data %>% spread(var, val)

correlation = spread_data %>%
  ggplot(aes(x=recall, y=precision)) +
  geom_smooth() +
  #geom_smooth(method="lm", se=F) +
  #scale_y_log10() +
  labs(x="Recall",
       y="Precision",
       title="Precision against Recall")+
  #scale_x_discrete() +
  #scale_y_log10() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_brewer(palette="Set1") + 
  scale_color_brewer(palette="Dark2")

synthetic_data = spread_data %>%
  filter(dataset == "synthetic")

real_data = spread_data %>%
  filter(dataset == "real")

print(paste("recall", wilcox.test(synthetic_data$recall, real_data$recall, exact=FALSE)$p.value))
print(paste("precision", wilcox.test(synthetic_data$precision, real_data$precision, exact = FALSE)$p.value))
print(paste("mAP", wilcox.test(synthetic_data$mAP, real_data$mAP, exact = FALSE)$p.value))
print(paste("average_iou", wilcox.test(synthetic_data$average_iou, real_data$average_iou, exact = FALSE)$p.value))

print(mean(real_data$actual_boxes))
print(mean(synthetic_data$actual_boxes))

print(p)
