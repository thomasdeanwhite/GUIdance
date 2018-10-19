library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library("RColorBrewer")
library(ggthemes)
args = commandArgs(TRUE)

load_data <- function(directory, filename){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv(filename)
  setwd(wd)

  return(data)
}

setwd('/home/thomas/work/GUIdance/pdfs')

data <- load_data('/home/thomas/work/GUIdance', "img_hist.csv")

data$pixel_value = data$pixel_value * 8.5

p_data = data

p = data %>%
  ggplot(aes(pixel_value, quantity)) +
  #geom_boxplot() + 
  geom_bar(color="grey40", alpha=0.2, stat="identity") +
  #geom_smooth(method="lm", se=F) +
  labs(x="Pixel Value",
       y="Quantity",
       title=paste("Pixel Histogram for dataset images")) +
  #scale_x_discrete() +
  scale_y_log10() +
  facet_wrap(~dataset, scales = "free") +
  theme_minimal()
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
  facet_wrap(~class) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme_minimal() +
  ylab("Proportion") +
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  stat_summary(fun.y=mean, colour="red", geom="text", show.legend = FALSE, 
               vjust=-0.7, aes( label=round(..y.., digits=3)))

ggsave("class_count.png", p, height=10, width=10, dpi=150)

data <- load_data('/home/thomas/work/GUIdance', "label_dims.csv")

p = data %>% filter(dimension != "area") %>% 
  ggplot(aes(interaction(dimension, dataset), value, fill=dimension)) +
  geom_boxplot() +
  theme_minimal() +
  facet_wrap(~class, scales="free") +
  stat_summary(fun.y=mean, colour="darkred", geom="point", shape=18, size=3, show.legend = FALSE) + 
  stat_summary(fun.y=mean, colour="red", geom="text", show.legend = FALSE, 
               vjust=-0.7, aes( label=round(..y.., digits=3))) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  
ggsave("label_dims.png", p, height=10, width=10, dpi=150)

data <- load_data('/home/thomas/work/GUIdance', "white_space.csv")

p = data %>%
  ggplot(aes(area, widget_count, color=dataset)) +
  #geom_boxplot() + 
  #geom_bar(color="grey40", alpha=0.2, stat="identity") +
  geom_point()+
  #geom_smooth(method="lm", se=F) +
  labs(x="Image Area",
       y="Widget Count",
       title="Widget Count against Image Area") +
  #scale_x_discrete() +
  #scale_y_log10() +
  #facet_wrap(~dataset, scales = "free") +
  theme_minimal() +
  scale_color_grey(start=0.4, end=1.0)
#theme(axis.text.x = element_text(angle = 45, hjust = 1))+

ggsave("widget_quant.png", p, height=10, width=10, dpi=150)

p = data %>%
  ggplot(aes(area, widget_area, color=dataset)) +
  #geom_boxplot() + 
  #geom_bar(color="grey40", alpha=0.2, stat="identity") +
  geom_point()+
  #geom_smooth(method="lm", se=F) +
  labs(x="Image Area",
       y="Widget Count",
       title="Widget Count against Image Area") +
  #scale_x_discrete() +
  #scale_y_log10() +
  #facet_wrap(~dataset, scales = "free") +
  theme_minimal() +
  scale_color_grey(start=0, end=0.7)
#theme(axis.text.x = element_text(angle = 45, hjust = 1))+

ggsave("widget_area.png", p, height=5, width=5, dpi=150)

print(p)
