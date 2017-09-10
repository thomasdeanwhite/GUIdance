library('ggplot2')
library('dplyr')
library('reshape2')

setwd('C:/work/NuiMimic/NuiMimic/')

reset <- function(){
  remove(screenshot)
  remove(heatmap)
}

#if (!exists("screenshot")){
  screenshot <- read.csv('screenshot.csv', header = TRUE)
#}

#if (!exists("heatmap")){
  heatmap <- read.csv('heatmap.csv', header = TRUE)

  heatmap <- heatmap[complete.cases(heatmap),]

  heatmap[heatmap$leftClick != 0,]$leftClick <- 1
  heatmap[heatmap$rightClick != 0,]$rightClick <- 1
#}

heatmap_melt <- melt(heatmap, id=c("x", "y"))

averaged_heatmap <- dcast(heatmap_melt, x + y ~ variable, mean)

freq_positions <- heatmap %>% group_by(x, y) %>%
  count(x, y)

max_freq <- (max(freq_positions$n) * 4)

freq_positions$n <- freq_positions$n / max_freq

joined_heatmap <- screenshot %>% left_join(freq_positions) %>%
  left_join(averaged_heatmap)

joined_heatmap$pixel <- joined_heatmap$pixel / 255

joined_heatmap$leftClick <- joined_heatmap$leftClick / 4
joined_heatmap$rightClick <- joined_heatmap$rightClick / 4

joined_heatmap[is.na(joined_heatmap)] <- 0

cols <- c("LEFT_CLICK"="red", "RIGHT_CLICK"="blue", "MOVE"="green")

p <- joined_heatmap %>%
  ggplot(aes(x=x, y=y, fill=pixel)) + geom_tile() +
  geom_point(aes(alpha=n, color="MOVE", stroke=0)) +
  geom_point(aes(alpha=leftClick, color="LEFT_CLICK", stroke=0)) +
  #geom_point(aes(alpha=rightClick,color="RIGHT_CLICK", stroke=0)) +
  scale_alpha_continuous(limits=c(0,1.0)) +
  scale_colour_manual(name="Action",values=cols) +
  scale_y_reverse() + scale_fill_gradient(low = "#000000", high = "#FFFFFF", space = "Lab",
                                          na.value = "#00FFFF", guide = "colourbar")

print(p)