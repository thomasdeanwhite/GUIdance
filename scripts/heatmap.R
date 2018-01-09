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

#screenshot <- matrix(rgb(screenshot[,,1],screenshot[,,2],screenshot[,,3], screenshot[,,4] * 0.2), nrow=dim(screenshot)[1])

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
# if (nrow(heatmap[heatmap$val < 0,]) > 0){
#   heatmap[heatmap$val < 0,]$val = 0
# }
#   
# if (nrow(heatmap[heatmap$val > 1,]) > 0){
#   heatmap[heatmap$val > 1,]$val = 1
# }
#cols <- c("LeftClick"="red", "RightClick"="blue", "Keyboard"="green", "Shortcut"="green")

heatmap$val = as.numeric(heatmap$val)

p <- heatmap %>%
  ggplot(aes(x=y, y=x, fill=var, alpha=val)) + #geom_tile() +
  #geom_point(aes(alpha=n, color="MOVE", stroke=0)) +
  #geom_point(aes(alpha=val, color="LeftClick", stroke=0), size=0.2, shape=15) +
  #geom_tile(aes(fill=var)) + #color="LeftClick")) +
  annotation_raster(glob_screenshot, xmin=1, xmax=1280, ymin=-1024, ymax=0) +
  geom_tile(colour="#FFFFFF00", aes(y, x)) +
  #scale_alpha_continuous(range = c(0, 1), guide = "none") +
  #geom_point(aes(alpha=rightClick,color="RIGHT_CLICK", stroke=0)) +
  scale_alpha_continuous(range=c(0.0,1.0)) +
  #scale_color_identity() +
  #scale_fill_gradient(low = "#FFFFFF", high = "#FF0000") +
  # scale_fill_gradient(low = "white", high = "red") +
  #scale_colour_manual(name="Action",values=cols) +

  scale_y_reverse() + 
  #scale_fill_gradient(low = "#000000", high = "#FFFFFF", space = "Lab",
  #                                        na.value = "#00FFFF", guide = "colourbar") +
  coord_fixed() +
  facet_wrap(~var) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

print(p)