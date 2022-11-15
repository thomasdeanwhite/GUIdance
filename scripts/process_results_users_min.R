library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(RColorBrewer)
library(ggthemes)
library(xtable)
library(effsize)
library(viridis)

setwd('/home/thomas/experiments/test_experiments')

dir = "output"
random_name = "detection"
detection_name = "user-models"
truth_name = "truth"
#truth_name = "random"

p_random_name = "PREDICTION"
p_detection_name = "WINMODEL"
p_truth_name = "API"
table_label = "chapter4:table:issta-comparison"
table_caption = "Branch Coverage of User-Learned Models, Object Detection and API."

out_name = "issta-user-comparison"

config = 3

limited_config = "user-random"

if (config == 1){
  random_name = "user-random"
  detection_name = "user-models"
  truth_name = "random"
  
  p_random_name = "EVENTRAND"
  p_detection_name = "WINMODEL"
  p_truth_name = "CLICKRAND"
  
  out_name = "issta-random-comparison"
  table_label = "chapter4:table:random-comparison"
}

if (config == 2){
  random_name = "user-models-random"
  detection_name = "user-models"
  truth_name = "user-models-oneout"
  
  p_random_name = "WINMODEL+RAND"
  p_detection_name = "WINMODEL+APP"
  p_truth_name = "WINMODEL-AUT"
  out_name = "user-window-app-comparison"
  table_label = "chapter4:table:window-comparison"
}

if (config == 3){
  random_name = "user-models-app"
  detection_name = "user-models"
  truth_name = "user-models-oneout"
  
  p_random_name = "APPMODEL"
  p_detection_name = "WINMODEL"
  p_truth_name = "WINMODEL-AUT"
  out_name = "user-window-app-oneout-comparison"
  table_label = "chapter4:table:window-app-comparison"
}

table_caption = paste("Branch Coverage of techniques ", p_detection_name, ", ", p_random_name, ", and ", p_truth_name, ".", sep="")

table_caption = paste(table_caption, "\\textbf{Bold} indicates significance. * Indicates Effect Size (*Small, **Medium and ***Large).")

load_file <- function(file){
  filename = paste(getwd(), dir, file, sep="/")
  cat(paste("\r", filename))
  row = readr::read_csv(filename)
  info=strsplit(file, "/")
  app = info[[1]][3]
  if (app == "Java_FireLands"){
    app = "Java_FabledLands"
  }
  row = mutate(row, ITERATION=info[[1]][2])
  row = mutate(row, APPLICATION=app)
  row = mutate(row, FILE=file)
  row = mutate(row, TECHNIQUE=info[[1]][1])
  return(row)
}
test_file = "test.txt"

images_dir = "images"
labels_dir = "labels"


# ------------------------

load_data <- function(){
  if (file.exists("user_data.RData")){
    load("user_data.RData")
  } else {
    pattern="coverage.csv"
    temp = list.files(path = dir, pattern=pattern, recursive = TRUE)
    r_data = bind_rows(lapply(temp, load_file))
    #data = mutate(data, BRANCH_COVERAGE=BRANCH_COVERED/(BRANCH_MISSED+BRANCH_COVERED+1))
    save(r_data, file="user_data.RData")
  }
  return(r_data)
}


process_data <- function(data){
  data = data %>% 
    mutate(BRANCH_COVERED=if_else(BRANCH_COVERED+BRANCH_MISSED==0, METHOD_COVERED, BRANCH_COVERED),
           BRANCH_MISSED=if_else(BRANCH_COVERED+BRANCH_MISSED==0, METHOD_MISSED, BRANCH_MISSED))
  
  #data$APPLICATION = substr(data$APPLICATION, 0, 8)
  
  #data[(data$BRANCH_COVERED + data$BRANCH_MISSED)==0,]$BRANCH_COVERED = 1
  
  #data$BRANCH_COVERAGE=data$BRANCH_COVERED/(data$BRANCH_COVERED+data$BRANCH_MISSED)
  
  p_data = data %>%
    group_by(ITERATION, APPLICATION, TECHNIQUE) %>%
    summarise(BRANCHES_COVERED_TOTAL=sum(BRANCH_COVERED),
              BRANCHES_MISSED_TOTAL=sum(BRANCH_MISSED),
              BRANCHES_TOTAL=sum(BRANCH_COVERED+BRANCH_MISSED),
              LINES_TOTAL=sum(LINE_COVERED+LINE_MISSED),
              BRANCH_COVERAGE=sum(BRANCH_COVERED))
  
  apps = unique(p_data$APPLICATION)
  
  for (a in apps){
    if (nrow(p_data %>% filter(APPLICATION == a, TECHNIQUE==limited_config)) == 0){
      p_data = p_data %>% filter(APPLICATION != a)
    } else {
      mb = max((p_data %>% filter(APPLICATION == a, TECHNIQUE=="truth"))$BRANCHES_TOTAL)
      p_data = p_data %>% mutate(BRANCH_COVERAGE=if_else(APPLICATION == a, BRANCH_COVERAGE/mb, as.double(BRANCH_COVERAGE)),
                               BRANCHES_TOTAL=if_else(APPLICATION == a, mb, BRANCHES_TOTAL))
    }
  }
  
  p_data$TECHNIQUE = factor(p_data$TECHNIQUE, levels=c(random_name, detection_name, truth_name))
  p_data = p_data[complete.cases(p_data),]
  
  p_data = p_data %>% filter(BRANCH_COVERAGE!=0)
  
  p_data = p_data %>% mutate(Technique = TECHNIQUE)
  p_data = p_data %>% mutate(Application = APPLICATION)
  
  p_data_r = p_data %>% filter(TECHNIQUE==random_name) %>% mutate(Technique = p_random_name)
  p_data_d = p_data %>% filter(TECHNIQUE==detection_name) %>% mutate(Technique = p_detection_name)
  p_data_t = p_data %>% filter(TECHNIQUE==truth_name) %>% mutate(Technique = p_truth_name)
  
  p_data = rbind(p_data_r, p_data_d, p_data_t)
  
  p_data$Technique = factor(p_data$Technique, c(p_detection_name, p_random_name, p_truth_name))
  
  # p_data = p_data %>% mutate(BRANCH_COVERAGE=BRANCHES_COVERED_TOTAL/(BRANCHES_COVERED_TOTAL+BRANCHES_MISSED_TOTAL))
  
  return(p_data)
}

get_table_row <- function(){
  # names = c('Application', 'Window Model', 'Prediction', 'Window Model', 'API')
  # if (config == 1){
  #   names = c('Application', 'Window Model', 'ISSTA Random', 'Window Model', 'Random')
  # } else   if (config == 2){
  #   names = c('Application', 'Window Model', 'Random', 'Window Model', 'App Model')
  # } else   if (config == 3){
  #   names = c('Application', 'Window Model', 'Random', 'Window Model', 'Window/Random')
  # }
  names = c('Application', p_detection_name, p_random_name, p_detection_name, p_truth_name)
  
  d = data.frame(matrix(ncol = length(names), nrow = 0), stringsAsFactors=FALSE)
  colnames(d) <- names
  
  
  return(d)
}

r3dp <- function(x, effsize){
  if (is.numeric(x)){
    if (is.nan(x)){
      return("1.000")
    }
    if(x < 0.001){
      return("$<$0.001")
    }
    return(format(round(x, 3), nsmall=3))
  } else {
    return(gsub("_", "-", x))
  }
}
  
bold <- function(x, effsize) {

  if (effsize < 0.5){
    return(r3dp(x))
  }
  
  asize = ""
  if (effsize > 0.56){
    asize = "*"
  }
  
  if (effsize > 0.64){
    asize = "**"
  }
  
  if (effsize > 0.71){
    asize = "***"
  }
  
  paste('{\\textbf{', asize, r3dp(x),'}}', sep ='')
}

get_table <- function(data){
  apps = unique(data$APPLICATION)
  
  table_data = get_table_row()
  
  a_db = 0 
  a_rb = 0
  a_drp= 0
  a_dra= 0
  a_tb= 0
  a_dtp= 0
  a_dta= 0
  
  a12_random = c()
  a12_truth = c()
  i = 0
  
  for (a in apps){
    i = i + 1
    print(a)
    
    style = r3dp
    
    rand = data %>% filter(APPLICATION == a, TECHNIQUE == random_name)
    det = data %>% filter(APPLICATION == a, TECHNIQUE == detection_name)
    tru = data %>% filter(APPLICATION == a, TECHNIQUE == truth_name)
    
    pv = wilcox.test(rand$BRANCH_COVERAGE, det$BRANCH_COVERAGE)$p.value
    pvt = wilcox.test(tru$BRANCH_COVERAGE, det$BRANCH_COVERAGE)$p.value
    style_t = r3dp
    
    style_d = r3dp
    style_dt = r3dp
    
    if (!is.nan(pv) && pv < 0.05){
      style = bold
      style_d = bold
    }
    
    if (!is.nan(pvt) && pvt < 0.05){
      style_t = bold
      style_dt = bold
    }
    
    a_rb = a_rb + mean(rand$BRANCH_COVERAGE)
    a_db = a_db + mean(det$BRANCH_COVERAGE)
    a_tb = a_tb + mean(tru$BRANCH_COVERAGE)
    
    a12_random <- VD.A(rand$BRANCH_COVERAGE, det$BRANCH_COVERAGE)
    a12_truth <- VD.A(tru$BRANCH_COVERAGE, det$BRANCH_COVERAGE)
    
    
    row = get_table_row()
    row[1,] = list(r3dp(a), style_d(median(det$BRANCH_COVERAGE), 1-a12_random$estimate), 
                   style(median(rand$BRANCH_COVERAGE), a12_random$estimate),
                   style_dt(median(det$BRANCH_COVERAGE), 1-a12_truth$estimate),
                   style_t(median(tru$BRANCH_COVERAGE), a12_truth$estimate))
    
    table_data = rbind(table_data, row)
  }
  
  row = get_table_row()
  row[1,] = list("Mean", r3dp(a_db / length(apps)), r3dp(a_rb / length(apps)), r3dp(a_db / length(apps)), r3dp(a_tb / length(apps)))
  
  table_data = rbind(table_data, row)
  
  # table_data = table_data %>% 
  #   mutate_each(funs(if(is.double(.)) as.double(.) else .))
  
  return(xtable(table_data, caption=table_caption, 
         label=table_label, align='ll|rr|rr'))
}

box_plots <- function(data, file){
  # colscale <- c("Random"="#8D8B97", p_detection_name="#B3BEAD",
  #                            p_truth_name="#FFE6CB")
  # fillscale <- scale_fill_manual(name="Technique", values=colscale)
  
  p = data %>% ggplot(aes(x=Technique, y=BRANCH_COVERAGE, fill=Technique)) +
    geom_boxplot() +
    xlab("Application") +
    ylab("Branch Coverage") +
    #fillscale +
    #scale_y_log10() +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none") +
    facet_wrap(~APPLICATION, scales = "free", ncol=5) +
    scale_fill_viridis(discrete=TRUE, option="viridis", begin=0.5)
    
  
  ggsave(file, p, width = 8, height=5)
  
  return(p)
}

print("Loading data...")
r_data = load_data()
print("Processing data...")
data = process_data(r_data)

p = box_plots(data, paste("chapter4-", out_name, ".pdf", sep=""))

#data_rq4 = data %>% filter(grepl("detection", TECHNIQUE) | grepl("truth", TECHNIQUE))

#p = box_plots(data_rq4, "rq4.pdf")

print(p)

table = get_table(data)

hlines <- c(-1, 0, nrow(table)-1, nrow(table))

latex = print(table, sanitize.text.function = function(x) x, include.rownames=FALSE, 
              scalebox = 0.7, booktabs = TRUE, hline.after = hlines)

write(latex, file=paste("chapter4-", out_name, ".tex", sep=""))

