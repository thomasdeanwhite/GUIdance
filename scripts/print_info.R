library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library("RColorBrewer")
args = commandArgs(TRUE)

load_data <- function(directory, filename){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv(filename)

  return(data)
}

setwd('/home/thomas/work/GUIdance')

data <- load_data('/home/thomas/work/GUIdance', "img_hist.csv")

data$pixel_value = data$pixel_value * 8.5

p = data %>%
  ggplot(aes(pixel_value, quantity)) +
  #geom_boxplot() + 
  geom_bar(color="grey40", alpha=0.2, stat="identity") +
  #geom_smooth(method="lm", se=F) +
  labs(x="Pixel Value",
       y="Quantity",
       title=paste("Pixel Histogram for dataset images")) +
  #scale_x_discrete() +
  #scale_y_log10() +
  theme_minimal()
  #facet_wrap(~dataset, scales = "free")
  #theme(axis.text.x = element_text(angle = 45, hjust = 1))+

ggsave("hist.png", p, height=4, width=6, dpi=150)

data <- load_data('/home/thomas/work/GUIdance', "label_heat.csv")

p = data %>% ggplot(aes(x,y)) +
  geom_raster(aes(fill=density)) +
  facet_wrap(class~dataset, scales="free") +
  scale_fill_gradient(low = "#0000FF", high = "#FF0000", na.value = "#00FF00") +
  scale_y_reverse()

ggsave("heatmap.png", p, height=10, width=10, dpi=150)

data <- load_data('/home/thomas/work/GUIdance', "class_count.csv")

p = data %>% ggplot(aes(dataset, count)) +
  geom_boxplot() +
  facet_wrap(~class, scales="free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylab("Proportion")

ggsave("class_count.png", p, height=10, width=10, dpi=150)

data <- load_data('/home/thomas/work/GUIdance', "label_dims.csv")

p = data %>% filter(dimension != "area") %>% 
  ggplot(aes(dataset, value, fill=dimension)) +
  geom_boxplot() +
  facet_wrap(~class, scales="free")
  
ggsave("label_dims.png", p, height=10, width=10, dpi=150)


print(p)
