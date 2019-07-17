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

p_random_name = "Pred"
p_detection_name = "WindowM"
p_truth_name = "API"

out_name = "issta-user-comparison"

config = 3

if (config == 1){
  random_name = "random"
  detection_name = "user-models"
  truth_name = "user-random"
  
  p_random_name = "ISSTA-Random"
  p_detection_name = "WindowM"
  p_truth_name = "User Random"
  
  out_name = "issta-random-comparison"
}

if (config == 2){
  random_name = "user-random"
  detection_name = "user-models"
  truth_name = "user-models-oneout"
  
  p_random_name = "User Random"
  p_detection_name = "WindowM"
  p_truth_name = "AppM"
  
  out_name = "user-comparison"
}


if (config == 3){
  random_name = "user-models-app"
  detection_name = "user-models"
  truth_name = "user-random"
  
  p_random_name = "Application"
  p_detection_name = "Window"
  p_truth_name = "Random Events"
  
  out_name = "window-app-comparison"
}


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
  data$TECHNIQUE = factor(data$TECHNIQUE, levels=c(random_name, detection_name, truth_name))
  
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
  
  p_data = p_data[complete.cases(p_data),]
  apps = unique(p_data$APPLICATION)
  
  for (a in apps){
    if (nrow(p_data %>% filter(APPLICATION == a, TECHNIQUE==detection_name)) == 0){
      p_data = p_data %>% filter(APPLICATION != a)
    } else {
      mb = max((p_data %>% filter(APPLICATION == a))$BRANCHES_TOTAL)
      p_data = p_data %>% mutate(BRANCH_COVERAGE=if_else(APPLICATION == a, BRANCH_COVERAGE/mb, as.double(BRANCH_COVERAGE)),
                               BRANCHES_TOTAL=if_else(APPLICATION == a, mb, BRANCHES_TOTAL))
    }
  }
  
  p_data = p_data %>% filter(BRANCH_COVERAGE!=0)
  
  p_data = p_data %>% mutate(Technique = TECHNIQUE)
  p_data = p_data %>% mutate(Application = APPLICATION)
  
  p_data_r = p_data %>% filter(TECHNIQUE==random_name) %>% mutate(Technique = p_random_name)
  p_data_d = p_data %>% filter(TECHNIQUE==detection_name) %>% mutate(Technique = p_detection_name)
  p_data_t = p_data %>% filter(TECHNIQUE==truth_name) %>% mutate(Technique = p_truth_name)
  
  p_data = rbind(p_data_r, p_data_d, p_data_t)
  
  p_data$Technique = factor(p_data$Technique, c(p_detection_name, p_random_name, p_truth_name))
  
  #p_data = p_data %>% mutate(BRANCH_COVERAGE=BRANCHES_COVERED_TOTAL/(BRANCHES_COVERED_TOTAL+BRANCHES_MISSED_TOTAL))
  
  return(p_data)
}

get_table_row <- function(){
  names = c('Application', '\\textbf{Win}dowM Cov.', '\\textbf{Pred}iction Cov.', 'P$_v$(Pred)', '\\^A$_{12}$(Pred)', '\\textbf{API} Cov.', 'P$_v$(Win, API)', '\\^A$_{12}$(Win, API)')
  if (config == 1){
    names = c('Application', '\\textbf{Win}dowM Cov.', '\\textbf{Rand}om Cov.', 'P$_v$(Win, Rand)', '\\^A$_{12}$(Win, Rand)', '\\textbf{U-R}and Cov.', 'P$_v$(Win, U-R)', '\\^A$_{12}$(Win, U-R)')
  } else   if (config == 2){
    names = c('Application', '\\textbf{Win}dowM Cov.', '\\textbf{U-R}and Cov.', 'P$_v$(Win, U-R)', '\\^A$_{12}$(Win, U-R)', '\\textbf{App}M Cov.', 'P$_v$(Win, App)', '\\^A$_{12}$(Win, App)')
  }
  d = data.frame(matrix(ncol = length(names), nrow = 0), stringsAsFactors=FALSE)
  colnames(d) <- names
  
  
  return(d)
}

r3dp <- function(x){
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
  
bold <- function(x) {paste('{\\textbf{',r3dp(x),'}}', sep ='')}

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
    
    if (!is.nan(pv) && pv < 0.05){
      style = bold
    }
    
    if (!is.nan(pvt) && pvt < 0.05){
      style_t = bold
    }
    
    a_rb = a_rb + mean(rand$BRANCH_COVERAGE)
    a_db = a_db + mean(det$BRANCH_COVERAGE)
    a_tb = a_tb + mean(tru$BRANCH_COVERAGE)
    if (is.nan(pv)){
      a_drp = a_drp + 1
    } else {
      a_drp = a_drp + pv
    }
    
    if (is.nan(pvt)){
      a_dtp = a_dtp + 1
    } else {
      a_dtp = a_dtp + pvt
    }
    a_dra = a_dra + VD.A(rand$BRANCH_COVERAGE, det$BRANCH_COVERAGE)$estimate
    a_dta = a_dta + VD.A(tru$BRANCH_COVERAGE, det$BRANCH_COVERAGE)$estimate
    
    a12_random <- c(a12_random, a_dra)
    a12_truth <- c(a12_truth, a_dta)
    
    
    row = get_table_row()
    row[1,] = list(style(a), r3dp(mean(det$BRANCH_COVERAGE)), style(mean(rand$BRANCH_COVERAGE)),
                   style(pv), style(VD.A(rand$BRANCH_COVERAGE, det$BRANCH_COVERAGE)$estimate), style_t(mean(tru$BRANCH_COVERAGE)), style_t(pvt), style_t(VD.A(tru$BRANCH_COVERAGE, det$BRANCH_COVERAGE)$estimate))
    
    table_data = rbind(table_data, row)
  }
  
  row = get_table_row()
  row[1,] = list("Mean", r3dp(a_db / length(apps)), r3dp(a_rb / length(apps)),
                r3dp(a_drp / length(apps)), r3dp(a_dra / length(apps)), r3dp(a_tb / length(apps)), 
                r3dp(a_dtp / length(apps)), r3dp(a_dta / length(apps)))
  
  table_data = rbind(table_data, row)
  
  table_data = table_data %>% 
    mutate_each(funs(if(is.double(.)) as.double(.) else .))
  
  return(xtable(table_data, caption="Branch Coverage of Random against Object Detection. \\textbf{Bold} indicates significance. "))
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

latex = print(table, sanitize.text.function = function(x) x, include.rownames=FALSE)

write(latex, file=paste("chapter4-", out_name, ".tex", sep=""))

