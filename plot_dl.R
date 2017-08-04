library(dplyr)
library(ggplot2)

setwd('/home/thomas/work/NuiMimic/NuiMimic/data')

data <- read.csv("training_out.log", header = FALSE)

p <- data %>% 
  ggplot(aes(x=V1, y=V2)) +
  geom_line() + xlab('epoch') + ylab('accuracy')

print(p)

