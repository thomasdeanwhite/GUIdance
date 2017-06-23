library(dplyr)
library(ggplot2)
library(tidyr)

setwd('C:/work/NuiMimic/NuiMimic/')

data <- read.csv("debug.csv", header = TRUE)

#p <- data %>% 
#  group_by(output, value, ignore_dict) %>%
#  ggplot(aes(x=output, y=value, group=output)) +
#  geom_boxplot() +
#  facet_wrap(state~ignore_dict, scales = "free")

p <- data %>% 
  group_by(output, value, state) %>%
  ggplot(aes(x=output, y=value, group=output)) +
  geom_boxplot() +
  facet_wrap(~state, scales = "free")

print(p)

