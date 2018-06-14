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

data = data %>% gather("loss", "value", loss, loss_position, loss_dimension, loss_obj, loss_class)

p = data %>%
  ggplot(aes(x=epoch, y=value, lty=dataset)) +
  geom_line() +
  #geom_smooth(method="lm", se=F) +
  scale_y_log10() +
  labs(x="Epoch",
       y="log10 - Loss",
       title=paste("Loss when training GUI recognition model")) +
  #scale_x_discrete() +
  #scale_y_log10() +
  theme_minimal() +
  facet_wrap(~loss, scales = "free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_brewer(palette="Set1") + 
  scale_color_brewer(palette="Set1")

print(p)
