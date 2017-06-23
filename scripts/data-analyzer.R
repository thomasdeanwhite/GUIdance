library(ngram)

load_files <- function(directory){
  dataset <- read.table(directory,
                header=FALSE, sep=",")
  
  return(dataset)
}

k_means <- function(data, centers){
  processed <- kmeans(data, centers, iter.max=100)
  return(processed)
}

write_centers_to_file <- function(centers, filename){
  #centers <- t(cbind(rownames(centers), centers))
  write(centers, file=filename, sep=',', ncolumns=nrow(centers))
}

write_sequences_to_file <- function(seqs, filename){
  write(seqs, file=filename, sep=',', ncolumns=length(seqs))
}

append_sequences_to_file <- function(seqs, filename){
  write(seqs, file=filename, sep=',', ncolumns=length(seqs), append=TRUE)
}

write_clusters_to_file <- function(named_clusters, filename){

  write(named_clusters, file=filename, sep=',', ncolumns=nrow(named_clusters))
}

name_data <- function(names, data){
  data <- t(cbind(names, data))
  return(data)
}

load_list <- function(directory){
  
  dataset <- read.table(directory, sep=",")
  
  dataset <- dataset[!is.na(dataset)]
  return(dataset)
}

replace_data <- function(mapping, data){
  result <- do.call("cbind", lapply(data, FUN=function(name){trim(mapping['means$cluster', which(grepl(name, data))])}))
  return(result)
}

trim <- function (x) gsub("^\\s+|\\s+$", "", x)

count_occurances <- function(previousValue, index, data){
  if (index <= 0){
    return(length(data));
  } else {
    #newData <- data[unlist(lapply(which(grepl(previousValue, data)), FUN=function(i){return(seq(i+1, i+index))}))]
    #l <- list()
    #l[[previousValue]] <- lapply(newData, FUN=function(data){count_occurances(data, index-1, newData)})
    l <- list()
    indices = lapply(which(grepl(previousValue, data)), FUN=function(i){return(seq(i+1, i+index))})
    l[[previousValue]] <- c(lapply(indices,
           FUN=function(list){count_occurances(data[list[[1]]], index-1, data[unlist(list)])}))
    return(l)
  }
}

n_sequences <- function(n, data){
  #lapply(data, FUN=function(d){count_occurances(d, n-1, data)})
  n_string <- concat(data, collapse = ' ')
  ng <- ngram(n_string, n=n)
  return(ng)
}

write_sequence_to_file <- function(ng, length, file_name){
  sink(file_name)
  #write(babble(ng, genlen=length), file=file_name, sep=',', ncolumns=length)
  print(ng, full=TRUE)
  sink()
}

generate_data <- function(file_start, extension, data, clusters, n, sequences, states){
  #is thumb on left hand side? (left handed)
  #removed <- data[!is.na(as.numeric(as.character(data)))]
  #removed_names <- unlist(data[removed,1])
  
  #sequences <- sequences[-c(match(removed_names, sequences))]
  #data <- data[-c(removed),]
  
  raw_data <- data[sapply(data, is.numeric)]
  names = data[1]
  print(paste("K-Means Clustering:", extension))
  means <- cluster::clara(raw_data, clusters, pamLike=TRUE)
  meds <- means$medoids
  meds <- cbind(1:clusters, meds)
  meds <- t(meds)
  write_centers_to_file(meds, paste('processed/', file_start, extension, sep=""))
  named_clusters <- unlist(t(cbind(names, means$clustering)))
  #write_clusters_to_file(named_clusters, paste('processed/', file_start, '-', clusters, '-', n, '.classificationdata', sep=""))
  n_c <- cbind(names, means$cluster)
  print("K-Means complete. Data written to files. Starting cluster replacement.")
  #n_seq <- replace_data(named_clusters, sequences)
  n_match <- match(n_c[,1], sequences)
  n_match <- n_match[!is.na(n_match)]
  n_seq <- replace(sequences, n_match, n_c[,2])
  write_sequences_to_file(n_seq, paste('processed/', file_start, ".raw_sequence", extension, sep=""))
  append_sequences_to_file(sequences, paste('processed/', file_start, ".raw_sequence", extension, sep=""))
  
  rowNum <- nrow(states)
  
  print(paste("- Sequence Writing Complete! Starting State Replacement of", rowNum, "states."))
  
  r_states = read.table(text="",
                        colClasses = c("character", "character"),
                        col.names = c("state", clusters))
  
  output_string <- ""
  
  state_ngrams <- ""

  if (rowNum > 2){
    for (i in 1:rowNum){
      ss <- unlist(strsplit(as.character(states[i,2]), "[,]"))
      ss <- ss[!is.na(ss)]
      ss <- ss[!is.null(ss)]
      n_match <- match(ss, n_c[,1])
      n_match <- n_match[!is.na(n_match)]
      unlistedClusters <- unlist(n_c[n_match,]['means$cluster'])
      sT <- table(unlistedClusters)
      stateT <- paste(states[i, 1], sep="")
      table_out <- capture.output(write.table(sT, "", col.names = FALSE, sep="#", quote = FALSE, row.names = FALSE, eol=","))
      
      output <- stateT
      output_string <- paste(output_string, paste(stateT), sep="\n")
      
      output_string <- paste(output_string, table_out, sep=":")
      
      state_ng <- paste(unlistedClusters)

      state_occurances <- n_sequences(n, state_ng)
      
      sink(paste('processed/', file_start, '.state-ngram', extension, sep=''), append=TRUE)
      print(paste("state:",paste(stateT),"\n", sep=""))
      

      print("cluster:")
      print(state_occurances, full=TRUE)
      print("\n")
      sink()

      remove(output)
      remove(table_out)
      remove(n_match)
      remove(ss)
      remove(sT)
      remove(stateT)
    }
  }
  
  print("- Writing State Replacement")
  
  sink(paste('processed/', file_start, '.state', extension, sep=''))
  print(cat(output_string), quote = FALSE, row.names = FALSE)
  sink()
  
  remove(r_states)
  remove(output_string)
  remove(raw_data)
  
  print("Replacement complete, calculating N-Grams.")
  occurances <- n_sequences(n, n_seq)
  return(occurances)
}

load_states <- function(filename){
  conn <- file(filename, open="r")
  lines <- readLines(conn)
  states = list()
  for (i in 2:length(lines)){
    state <- unlist(strsplit(lines[i], ":", fixed=TRUE))
    hands <- unlist(strsplit(state[[1]][2], "[,]"))
    s <- c(state, hands)
    states <- rbind(states, s)
    states <- states[,1:2]
  }
  close(conn)
  remove(lines)
  remove(conn)
  return(states)
}

process <- function(file_starts, clusters, n, filename){
  handdata <- c()
  first_file_start = head(file_starts, 1)
  print(first_file_start)
  handdata <- load_files(paste(first_file_start, ".pool_joint_positions", sep=""))
  seqdata <- load_list(paste(first_file_start, ".sequence_hand_data", sep=""))
  positiondata <- load_files(paste(first_file_start, ".pool_hand_positions", sep=""))
  rotationdata <- load_files(paste(first_file_start, ".pool_hand_rotations", sep=""))
  gesture_file <- paste(first_file_start, '.sequence_gesture_data', sep="")
  gestures <- readChar(gesture_file, file.info(gesture_file)$size)
  states <- load_states(paste(first_file_start, ".pool_dct", sep=""))
  states_gestures <- load_states(paste(first_file_start, ".pool_dct_gestures", sep=""))
  while (length(file_starts) > 1){
    file_starts = file_starts[2:(length(file_starts))]
    file_start = head(file_starts, 1)
    print(file_start)
    handdata <- rbind(handdata, load_files(paste(file_start, ".pool_joint_positions", sep="")))
    seqdata <- c(seqdata, load_list(paste(file_start, ".sequence_hand_data", sep="")))
    positiondata <- rbind(positiondata, load_files(paste(file_start, ".pool_hand_positions", sep="")))
    rotationdata <- rbind(rotationdata, load_files(paste(file_start, ".pool_hand_rotations", sep="")))
    gesture_file <- paste(file_start, '.sequence_gesture_data', sep="")
    gestures <- paste(gestures, readChar(gesture_file, file.info(gesture_file)$size))
    states <- rbind(load_states(paste(file_start, ".pool_dct", sep="")))
    states_gestures <- rbind(load_states(paste(file_start, ".pool_dct_gestures", sep="")))
  }
  
  ngramhand <- generate_data(filename, '.joint_position_data', handdata, clusters, n, seqdata, states)
  sink(paste('processed/', filename, '.joint_position_ngram', sep=''))
  print(ngramhand, full=TRUE)
  sink()
  
  remove(ngramhand)
  
  ngramposition <- generate_data(filename, '.hand_position_data', positiondata, clusters, n, seqdata, states)
  sink(paste('processed/', filename,'.hand_position_ngram', sep=''))
  print(ngramposition, full=TRUE)
  sink()
  
  remove(ngramposition)
  
  ngramrotation <- generate_data(filename, '.hand_rotation_data', rotationdata, clusters, n, seqdata, states)
  sink(paste('processed/', filename, '.hand_rotation_ngram', sep=''))
  print(ngramrotation, full=TRUE)
  sink()
  
  remove(ngramrotation)

  ngGestures <- ngram(gestures, n=n)
  sink(paste('processed/', filename, '.gesture_type_ngram', sep=''))
  print(ngGestures, full=TRUE)
  sink()

  
  write_sequences_to_file(gestures, paste('processed/', filename, '.raw_sequence.gesture_type_data', sep=''))
  
  output_string <- ""
  for (i in 1:length(states_gestures[,2])){
    ss <- unlist(strsplit(as.character(states_gestures[i,2]), "[,]"))
    ss <- ss[!is.na(ss)]
    ss <- ss[!is.null(ss)]
    sT <- table(unlist(ss))
    stateT <- paste(states_gestures[i, 1], sep="")
    table_out <- capture.output(write.table(sT, "", col.names = FALSE, sep="#", quote = FALSE, row.names = FALSE, eol=","))
    
    output <- stateT
    output_string <- paste(output_string, paste(stateT), sep="\n")
    
    output_string <- paste(output_string, table_out, sep=":")
    
    output <- stateT
    output_string <- paste(output_string, paste(stateT), sep="\n")
    
    output_string <- paste(output_string, table_out, sep=":")
    
    state_ng <- paste(unlistedClusters)
    
    state_occurances <- n_sequences(n, state_ng)
    
    sink(paste('processed/', file_start, '.state-ngram.gesture_data', sep=''), append=TRUE)
    print(paste("state:",paste(stateT),"\n", sep=""))
    
    
    print("cluster:")
    print(state_occurances, full=TRUE)
    print("\n")
    sink()
    
  }
  sink(paste('processed/', filename, '.state.gesture_data', sep=''))
  print(cat(output_string), quote = FALSE, row.names = FALSE)
  sink()
  
  remove(output_string)
  remove(states_gestures)
  
  
}

process_all <- function(file_starts, filename){
    process(file_starts, 100, 2, paste(filename, sep=""))
    rm(list=ls(all=TRUE))
}