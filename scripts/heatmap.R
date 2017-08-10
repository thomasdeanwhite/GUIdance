library('ggplot2')
library('dplyr')
library('reshape2')

setwd('~/work/NuiMimic/NuiMimic/')

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
  
  heatmap[heatmap$leftClick != 0,] <- 1
  heatmap[heatmap$rightClick != 0,] <- 1
#}

heatmap_melt <- melt(heatmap, id=c("x", "y"))

averaged_heatmap <- dcast(heatmap_melt, x + y ~ variable, mean)

freq_positions <- heatmap %>% group_by(x, y) %>%
  count(x, y)

freq_positions$n <- freq_positions$n / max(freq_positions$n)

joined_heatmap <- screenshot %>% left_join(freq_positions) %>%
  left_join(averaged_heatmap)

joined_heatmap$pixel <- joined_heatmap$pixel / 255

joined_heatmap[is.na(joined_heatmap)] <- 0

p <- joined_heatmap %>%
  ggplot(aes(x=x, y=y, fill=pixel)) + geom_tile() +
  geom_tile(alpha=joined_heatmap$n, fill="green") +
  geom_tile(alpha=joined_heatmap$leftClick, fill="red") +
  geom_tile(alpha=joined_heatmap$rightClick, fill="blue") +
  scale_y_reverse() + scale_fill_gradient(low = "#000000", high = "#FFFFFF", space = "Lab",
                                          na.value = "#FF0000", guide = "colourbar")

print(p)