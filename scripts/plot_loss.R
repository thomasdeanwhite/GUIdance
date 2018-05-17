library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library("RColorBrewer")
args = commandArgs(TRUE)

load_data <- function(directory){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv("training.csv")
  return(data)
}

setwd('/home/thomas/work/GuiMimic')

data <- load_data('/home/thomas/work/GuiMimic')

p = data %>%
  ggplot(aes(x=weight, y=Intersection_Over_Union, color=dataset, fill=dataset)) +
  geom_line() +
  #geom_smooth(method="lm", se=F) +
  scale_y_log10() +
  labs(x="Training Iteration",
       y="Intersection Over Union",
       title=paste("IoU for grayscale images with confidence in top", (100*(1-percent)), "quantile")) +
  #scale_x_discrete() +
  #scale_y_log10() +
  theme_minimal() +
  #facet_wrap(~dataset, scales = "free") +
  ylim(0.0, 1.0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_brewer(palette="Set1") + 
  scale_color_brewer(palette="Set1")