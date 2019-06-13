library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(RColorBrewer)
library(ggthemes)
library(viridis)
library(Hmisc)

args = commandArgs(TRUE)

load_data <- function(directory, file){
  wd = getwd()
  setwd(directory)
  data <- readr::read_csv(file)
  return(data)
}

setwd('/home/thomas/work/GUIdance')

data <- load_data('/home/thomas/work/GUIdance', 'validation-iou.csv')
data2 <- load_data('/home/thomas/work/GUIdance', 'validation-centre.csv')

data = data %>% mutate(Metric="IoU")
data2 = data2 %>% mutate(Metric="Centre")

data = bind_rows(data, data2)

data$class = "Widget"

average_precision = data[FALSE,]

classes = unique(data$class)
datasets = unique(data$dataset)

t_data = data

t_data$Sensitivity = capitalize(t_data$sensitivity)

colscale <- c("Synthetic"="#FFAA99", "Ubuntu"="#99AAFF",
              "Mac"="#FFFFAA")
fillscale <- scale_fill_manual(name="Dataset", values=colscale)

dataset_a = t_data %>% mutate(Dataset=dataset)
dataset_r = dataset_a %>% filter(dataset == "real") %>% mutate(Dataset="Ubuntu")
dataset_s = dataset_a %>% filter(dataset == "synthetic") %>% mutate(Dataset="Synthetic")
dataset_m = dataset_a %>% filter(dataset == "mac") %>% mutate(Dataset="Mac")
t_data = rbind(dataset_r, dataset_s, dataset_m)

t_data = t_data[t_data$Metric == "IoU",]

t_data$Dataset = factor(t_data$Dataset, c("Ubuntu", "Mac", "Synthetic"))

p = t_data  %>% filter(dataset == "real" | dataset == "synthetic") %>%
  ggplot(aes(x=Sensitivity, y=value, fill=Dataset)) +
  geom_boxplot() +
  labs(x="",
       y="",
       title="") +
  #facet_wrap(~Metric) +
  theme_bw() +
  #theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_viridis(discrete=TRUE, option="viridis", begin=0.5) + 
  theme(legend.position="bottom", 
        legend.margin=margin(t=-0.7, r=0, b=0.5, l=0, unit="cm"),
        plot.margin = unit(x = c(0, 0, -0.2, 0), units = "cm"))

print(median((t_data %>% filter(dataset == "real", sensitivity == "precision", Metric == "Centre"))$value))
print(median((t_data %>% filter(dataset == "real", sensitivity == "recall", Metric == "Centre"))$value))

print(median((t_data %>% filter(dataset == "real", sensitivity == "precision", Metric == "IoU"))$value))
print(median((t_data %>% filter(dataset == "real", sensitivity == "recall", Metric == "IoU"))$value))

ggsave("rq1.pdf", p, width = 3, height=2.5)
ggsave("rq1.png", p, width = 3, height=2.5)


p = t_data  %>% filter(dataset == "mac" | dataset == "synthetic") %>%
  #filter(iou_threshold == 0.5) %>%
  #filter(correct==1) %>%
  #group_by(size_cat) %>%
  ggplot(aes(x=Sensitivity, y=value, fill=Dataset)) +
  geom_boxplot() +
  #geom_bar(position = "dodge", stat = "summary", fun.y="sum") +
  #geom_smooth(method="lm", se=F) +
  #scale_y_log10() +
  labs(x="",
       y="",
       title="") +
  #scale_x_discrete() +
  #scale_y_log10
  #facet_wrap(~Metric) +
  theme_bw() +
  #facet_wrap(busy_cat~size_cat) +
  #theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_viridis(discrete=TRUE, option="viridis", begin=0.75) + 
  theme(legend.position="bottom", 
        legend.margin=margin(t=-0.7, r=0, b=0.5, l=0, unit="cm"),
        plot.margin = unit(x = c(0, 0, -0.2, 0), units = "cm"))
#scale_fill_brewer(palette="Set1") +
#scale_color_brewer(palette="BrBG")

print(median((t_data %>% filter(dataset == "mac", sensitivity == "precision", Metric == "Centre"))$value))
print(median((t_data %>% filter(dataset == "mac", sensitivity == "recall", Metric == "Centre"))$value))


print(median((t_data %>% filter(dataset == "mac", sensitivity == "precision", Metric == "IoU"))$value))
print(median((t_data %>% filter(dataset == "mac", sensitivity == "recall", Metric == "IoU"))$value))

ggsave("rq2.pdf", p, width = 3, height=2.5)

print(p)


# data <- load_data('/home/thomas/work/GUIdance', 'confusion.csv')
# 
# #data = data[!data$dataset == "real",]
# 
# #data = data %>% filter(predicted_class != "menu_item", actual_class != "menu_item")
# 
# data = data %>% mutate(dataset = capitalize(dataset))
# 
# data = data %>% group_by(dataset, predicted_class) %>%
#   mutate(percent = (quantity/(sum(quantity)+1)))
# 
# p = data %>% ggplot(aes(predicted_class, actual_class)) +
#   geom_tile(aes(fill=percent)) +
#   labs(x="Predicted Class",
#        y="Actual Class",
#        title="") +
#   theme_bw() +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
#   scale_fill_gradient(low = "#000077", high = "#FF9900", na.value = "#00FF00") +
#   facet_wrap(~dataset)
# 
# ggsave("confusion.pdf", p, height=3, width=8)
# 
# print(p)

