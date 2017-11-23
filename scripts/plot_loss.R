library("dplyr")
library("ggplot2")
library("reshape2")

setwd('~/work/NuiMimic/NuiMimic/data')

print("Loss plot")
loss_data <- read.csv('encoding_out.log')
mloss = melt(loss_data, id=c("epoch")) %>%
  filter(variable != 'learn_rate')
p = mloss %>%
  ggplot(aes(x=epoch, y=value, color = variable, fill=variable)) +
  geom_line() +
  xlab("Epoch") + 
  ylab("Loss (average mean^2 difference)") +
  scale_y_log10(limits=c(min(mloss$value), max(mloss$value)))  +
  ggtitle("Loss function for GUI data set")

print("Distribution plot - loading")
raw_data <- read.csv('data.csv')
colnames(raw_data) <- c('d')
raw_mean = mean(raw_data$d)
print("Distribution plot - plotting")
p = raw_data %>%
  #sample_n(10000) %>%
  ggplot(aes(x=d)) +
  geom_histogram(stat="bin", binwidth = 0.01) +
  xlab("Processed Pixel Value") +
  geom_vline(xintercept = raw_mean)
  #stat_function(fun=function(x)tanh(x), geom="line", aes(colour="square"))

print(p)