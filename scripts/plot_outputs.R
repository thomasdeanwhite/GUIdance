library("dplyr")
library("ggplot2")

setwd('/home/thomas/work/NuiMimic/NuiMimic/default-user/')

outputs = read.csv(file="training_outputs.csv", header = FALSE)

cols = c("variable", "group")

x_mov = data.frame(outputs$V1, 'x_vel')
y_mov = data.frame(outputs$V2, 'y_vel')
lmb = data.frame(outputs$V3, 'lmb')
rmb = data.frame(outputs$V4, 'rmb')

colnames(x_mov) <- cols
colnames(y_mov) <- cols
colnames(lmb) <- cols
colnames(rmb) <- cols

exploded = rbind(x_mov, y_mov, lmb, rmb)

#p = exploded %>%
#  ggplot(aes(x=variable, y=group, fill=group)) + geom_boxplot()

p= exploded %>%
  ggplot(aes(x=variable)) +
  geom_density() +
  facet_wrap(~group, scales="free")


print(p)