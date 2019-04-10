library(tidyverse)

map.booleans <- function(x) {
  as.logical(plyr::mapvalues(x, c('False', 'True'), c(F, T), warn_missing = F))
}

check.predictors <- function(Df){
  # Ensure all we have no 0's or NA's in our main predictor variables
  # Raise error if so.
  all(sapply(Df[,c('cooccurrence.predictions', 'posterior.predictions', 'association.predictions')], 
             function(x) min(x) > 0.0 & is.finite(x)))
}


get.recognition.data <- function(recognitiondata.csvfile){
  # Minimal cleaning and processing of the recognition memory data
  
  Df <- read.csv(recognitiondata.csvfile, 
                 header=T, 
                 stringsAsFactors = F)
  
  Df %<>% select(-X) %>%   # Drop X 
    mutate(expected = map.booleans(expected),  # Convert 'True', 'False' to T, F 
           response = map.booleans(response),
           hit = map.booleans(hit),
           correct = map.booleans(correct)) %>% # Filter out the rows with word = 'dhow'
    filter(word != 'dhow') 

  check.predictors(Df)

  Df %>% mutate(posterior.predictions = log(posterior.predictions),
                association.predictions = log(association.predictions),
                cooccurrence.predictions = log(cooccurrence.predictions))
  
}

get.recall.data <- function(recalldata.csvfile){
  # Minimal cleaning and processing of the recall memory data
  
  Df <- read.csv(recalldata.csvfile, 
                 header=T, 
                 stringsAsFactors = F)
  
  Df %<>% select(-X) %>%   # Drop X 
    mutate(accuracy = map.booleans(accuracy)) # Convert 'True', 'False' to T, F 

  check.predictors(Df)  

  Df %>% mutate(posterior.predictions = log(posterior.predictions),
                association.predictions = log(association.predictions),
                cooccurrence.predictions = log(cooccurrence.predictions))
  
}



