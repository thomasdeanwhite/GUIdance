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

setwd('/home/thomas/work/GUIdance')

data <- load_data('/home/thomas/work/GUIdance')

data = data %>% gather("loss", "value", loss, loss_position, loss_dimension, loss_obj, loss_class, precision, recall, mAP)

p = data %>%
  ggplot(aes(x=epoch, y=value, lty=dataset, color=dataset)) +
  geom_line() +
  #geom_smooth(method="lm", se=F) +
  labs(x="Epoch",
       y="",
       title=paste("Loss when training GUI recognition model")) +
  #scale_x_discrete() +
  #scale_y_log10() +
  theme_minimal() +
  facet_wrap(~loss, scales = "free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_brewer(palette="Set1") + 
  scale_color_brewer(palette="Dark2")

print(p)
