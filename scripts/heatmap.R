library('ggplot2')
library('dplyr')
library('reshape2')
library('png')
library('grid')

setwd('/home/thomas/work/NuiMimic/NuiMimic/data/')

reset <- function(){
  remove(screenshot)
  remove(heatmap)
}

screenshot <- readPNG('screenshot.png')

glob_screenshot <- screenshot#rasterGrob(screenshot, interpolate=TRUE)

# screen_grey = apply(screenshot, 1, rowMeans)
# screen_grey = as.data.frame(screen_grey, row.names = range(0, 1024))
# 
# screen_grey_tidy = data.frame(x=c(), y=c(), var=c(), val=c())



# for (i in 1:1280){
#   row_screenshot = data.frame(x=c(), y=c(), var=c(), val=c())
#   for (j in 1:1024){
#     row_screenshot = bind_rows(row_screenshot, 
#       data.frame(x=c(i), y=c(j), var=c("Pixel"), val=c(screen_grey[i,j])))
#   }
#   screen_grey_tidy = bind_rows(screen_grey_tidy, row_screenshot)
#   cat(paste("\r", i, "/1280"))
# }

#if (!exists("heatmap")){
  heatmap <- read.csv('screenshot_out.csv', header = TRUE)
#}
  


cols <- c("LeftClick"="red", "RightClick"="blue", "Keyboard"="green", "Shortcut"="green")

p <- heatmap %>%
  ggplot(aes(x=x, y=y, fill=val)) + #geom_tile() +
  #geom_point(aes(alpha=n, color="MOVE", stroke=0)) +
  annotation_raster(glob_screenshot, xmin=1, xmax=1280, ymin=-1024, ymax=0) +
  geom_point(aes(alpha=val, color="LeftClick", stroke=0), size=1, shape=15) +
  #geom_point(aes(alpha=rightClick,color="RIGHT_CLICK", stroke=0)) +
  scale_alpha_continuous(limits=c(0,16.0)) +
  scale_colour_manual(name="Action",values=cols) +
  scale_y_reverse() + scale_fill_gradient(low = "#000000", high = "#FFFFFF", space = "Lab",
                                          na.value = "#00FFFF", guide = "colourbar")

print(p)