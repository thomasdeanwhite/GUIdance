library(devtools)
#install_github("ndphillips/yarrr")
library("yarrr")
library("reshape2")
library("ggplot2")

load_positions <- function(filename, file_pattern, numbers){
  files <- list.files(path=filename, pattern=paste(file_pattern, "*", sep=""), full.names=T, recursive=FALSE)
  first = TRUE
  dataset = NULL
  for (i in 2:length(files)){
    read = TRUE
    fname = files[i]
    if (length(numbers) > 0){
      found = FALSE
      for (j in numbers){
        if(length(grep(paste('-', j, '{1}.csv', sep=''),fname))>0){
          found=TRUE
          break
        }
      }
      if (!found){
        read = FALSE
      }
    }
    if (read){
      print(fname)
      t <- read.table(fname, header=FALSE, fill=TRUE, sep=",")
      t['run'] <- i
      if (first){
        dataset <- t
        first = FALSE
      } else {
        dataset <- rbind(dataset, t)
      }
    }

  }
  return(dataset)
}

plot_table <- function(data, filename){
  data <- data[!is.na(data[2]),]
  dat <- sort(table(data[1]))
  dat <- dat[dat > 0]
  pdf(paste(filename, ".pdf", sep=''), height=8.27, width=11.69)
  plot(dat)
  dev.off();
  
  mi <- min(data[1])
  ma <- max(data[1])
  
  pdf(paste(filename, "-bar.pdf", sep=''), height=8.27, width=11.69)
  pirateplot(formula=V1~weight, 
             data=data, 
             xlab = 'State Weight', ylab='Cluster', main="Clusters seeded for given runs",
             line.fun=median, ylim=c(mi, ma))
  dev.off();
  
}

load_data <- function(filename){
  files <- list.files(path=filename, pattern="*test-results.csv", full.names=T, recursive=FALSE)
  dataset <- read.table(files[1], header=TRUE, sep=",")
  for (i in 2:length(files)){
    t <- read.table(files[i], header=TRUE, sep=",")
    common_cols <- intersect(colnames(dataset), colnames(t))
    dataset <- rbind(
      subset(dataset, select=common_cols),
      subset(t, select=common_cols)
    )
    dataset <- rbind(dataset, t)
  }
  
  dataset['runtime_mins'] = round(dataset['runtime'] / 60000);
  
  dataset['fs'] <- apply(dataset, 1, FUN=function(x){
    if (x['frameSelectionStrategy'] == 'RANDOM'){
      val <- 'RANDOM'
    } else if (x['frameSelectionStrategy'] == 'STATE_DEPENDANT'){
      val <- 'STATE_DEPENDENT'
    } else {
      val <- x['frameSelectionStrategy']
    }
  })
  
  dataset[dataset$fs == 'REPRODUCTION' | dataset$fs == 'USER_PLAYBACK',]$runtime_mins <- dataset[dataset$fs == 'REPRODUCTION' | dataset$fs == 'USER_PLAYBACK',]$runtime_mins + 1
  
  return(dataset)
}

colors <- c('blue', 'red', 'green', 'orange', 'yellow', 'cyan', 'purple', 'pink', 'brown', 'grey')
line_types = c(18, 17, 15, 16, 19, 20, 21, 22, 23, 24)

line_plot <- function(data, name){
  #pdf(paste(name, ".pdf", sep=''))
  #par(mar = c(10, 5, 5, 4) + 0.1)
  
  name <- paste(name, ".pdf", sep="")
  
  types <- (unique(data$frame_selector))
  types <- sort(types)
  counter <- 1
  #data$line_coverage <- log10(data$line_coverage)
  #plot(data$runtime_mins, data$related_line_coverage, type='l', main='Lines', col='white', xlab='Runtime (minutes)', ylab='Lines Covered')
  pd <- position_dodge(0.5)
  
  dataS <- summarySE(data, measurevar="line_coverage", groupvars=c("fs", "gesture_files", "runtime_mins" ))
  
  ggplot(dataS, aes(x=runtime_mins, y=line_coverage, colour=gesture_files))+
    geom_errorbar(aes(ymin=line_coverage-ci, ymax=line_coverage+ci), width=.1, position=pd) +
    geom_line() +
    geom_point()
  
  #ggplot(dataS, aes(x=states_discovered, y=line_coverage, colour=fs))+
  #  geom_errorbar(aes(ymin=line_coverage-ci, ymax=line_coverage+ci), width=.1, position=pd) +
  #  geom_line() +
  #  geom_point()

  ggsave(name, device="pdf")

  
  for (t in types){
    data_type <- data[data$frame_selector == t,]
    d <- data_type#aggregate(data_type, by=list(data_type$runtime_mins), FUN=mean, na.rm=TRUE)
    #lines((d$runtime/60000), d$related_line_coverage, type='l', col=colors[counter], lwd=2)
    #points((d$runtime/60000), d$related_line_coverage, col=colors[counter], cex=1, pch=line_types[counter], lwd=3)
    
    
    counter <- counter + 1
  }
  #legend("bottomright", legend=types, col=colors, pch=line_types, lwd=3)
  #dev.off()
}

var_plot <- function(data, name){
  
  name <- paste(name, ".pdf", sep="")
  
  types <- (unique(data$frame_selector))
  types <- sort(types)
  counter <- 1

  pd <- position_dodge(0.5)
  
  dataS <- summarySE(data, measurevar="statesVisited", groupvars=c("lineCoverage" ))
  
  ggplot(data, aes(x=statesVisited, y=lineCoverage))+
    geom_point() +
    #geom_line()
  
  ggsave(name, device="pdf")
}

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  library(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}

bplot <- function(data, name, title){
  #pdf(paste(name, ".pdf", sep=''), height=8.27, width=11.69)
  #par(mar = c(10, 5, 5, 4) + 0.1, cex.axis=1)
  #boxplot(related_line_coverage~person_method, data=data, main=title, ylab="Line Coverage", las=2, outline=FALSE)
  
  data$v = data$fs
  data$g = data$fs
  data$y = data$branchCoverage
  
  mi <- round(min(data$y)-0.001, digits=3)
  ma <- round(max(data$y)+0.001, digits=3)
  
  data <- data[order(data$v),]
  
  
  pd <- position_dodge(0.5)
  
  ggplot(data, aes(x=v, y=y, colour=g))+
    #geom_errorbar(aes(ymin=y-ci, ymax=y+ci), width=.1, position=pd) +
    geom_boxplot() +
    xlab("Frame Selection Strategy") +
    ylab("Branch Coverage")
  
  name <- paste(name, ".pdf", sep="")
  
  ggsave(name, device="pdf")
  
  
  #boxplot(formula=y~v, 
  #        data=data, 
  #        xlab = 'Generation Strategy', ylab='Line Coverage', main=title, ylim=c(mi, ma))
  #dev.off()
}

bplot_states <- function(data, name, title){
  #pdf(paste(name, ".pdf", sep=''), height=8.27, width=11.69)
  #par(mar = c(10, 5, 5, 4) + 0.1, cex.axis=1)
  #boxplot(related_line_coverage~person_method, data=data, main=title, ylab="Line Coverage", las=2, outline=FALSE)
  
  data$v = data$fs
  data$g = data$fs
  data$y = data$statesVisited
  
  mi <- round(min(data$y)-0.001, digits=3)
  ma <- round(max(data$y)+0.001, digits=3)
  
  data <- data[order(data$v),]
  
  
  pd <- position_dodge(0.5)
  
  ggplot(data, aes(x=v, y=y, colour=g))+
    #geom_errorbar(aes(ymin=y-ci, ymax=y+ci), width=.1, position=pd) +
    geom_boxplot() +
    xlab("Frame Selection Strategy") +
    ylab("States Seen")
  
  name <- paste(name, ".pdf", sep="")
  
  ggsave(name, device="pdf")
}

bplot_filter <- function(data, name, title, filter){
  
  data <- data[data$fs %in% filter,]
  
  bplot(data, paste(name, "-coverage", sep=""), title)
  
  bplot_states(data, paste(name, "-states", sep=""), title)
}

vplot <- function(data, name, title){

  #par(mar = c(10, 5, 5, 4) + 0.1, cex.axis=0.5)
  #boxplot(related_line_coverage~person_method, data=data, main=title, ylab="Line Coverage", las=2, outline=FALSE)
  mi <- round(min(data$line_coverage)-0.001, digits=3)
  ma <- round(max(data$line_coverage)+0.001, digits=3)
  
  data$variable <- paste(data$gesture_files, data$fs, sep="-")
  data$value <- data$line_coverage
  
  data <- data[order(data$variable),]
  data$A <- seq(1, length(data[, 'variable']))
  
  #pirateplot(formula=line_coverage~v,
  #           data=data, 
  #           xlab = 'Frame Selector', ylab='Line Coverage', main=title,
  #           line.fun=median, ylim=c(mi, ma))
  
  p <- ggplot(dat = data, aes(x=A, y=value, group=variable)) +
    stat_boxplot(geom ='errorbar') +
    geom_boxplot(aes(fill=variable)) +
    #geom_violin(aes(fill=variable), alpha=1/3) +
    guides(fill = guide_legend(reverse=TRUE)) +
    coord_flip() +
    theme(axis.text.y=element_blank(), axis.title.y = element_blank()) +
    labs(y = "Line Coverage")
  
  p
  
  ggsave(paste(name, '.pdf', sep=''), device="pdf")
}

make <- function(){
  folders <- list.files(path="", pattern="*/", full.names=T, recursive=FALSE)
  texTable <- "\\begin{figure}\n
  \\begin{table}\n
  \\begin{tabulerx}{0.4\textwidth}{l|rr|rr|rr|rr|rr}\n
  Application & \\multicolumn{2}{c}{Playback} & \\multicolumn{2}{c}{Random} & \\multicolumn{2}{c}{VQ} & \\multicolumn{2}{c}{Single Model} & \\multicolumn{2}{c}{State Dependent} \\\n
  \\hline\n"
  
  data10 <- load_data(folders[1])[FALSE,]
  
  for (i in 1:length(folders)){
    folder <- folders[i]
    data <- load_data(folder)
    d10 <- data[data$runtime_mins == 10,]
    
    
    
    data10 <- rbind(d10)
    #manual <- 
  }
  
  bplot_filter(data10, 'rq1', 'Line Coverage for different Frame Selection Strategies', c('USER_PLAYBACK', 'REPRODUCTION', 'RANDOM', 'VQ'))
  bplot_filter(data10, 'rq2', 'Line Coverage for different Frame Selection Strategies', c('SINGLE_MODEL', 'VQ'))
  bplot_filter(data10, 'rq3', 'Line Coverage for different Frame Selection Strategies', c('SINGLE_MODEL', 'STATE_DEPENDENT'))
  
}
