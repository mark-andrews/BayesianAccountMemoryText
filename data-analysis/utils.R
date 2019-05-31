library(tidyverse)
library(assertthat)
library(tools)
library(parallel)
#library(rjags)
#library(coda)

####################################################

logit <- function(p) log(p/(1-p))

ilogit <- function(x) 1/(1+exp(-x))

get.checksum <- function(filename) as.vector(md5sum(filename))

get.randint <- function(size, min=1, max=1e5){
  ceiling(runif(size, min=min, max=max))
}

re.scale <- function(x) as.numeric(scale(x))

get.word2index <- function(Q, rows=T){
  
  if (rows) {
    the.names <- rownames(Q)
  }
  else {
    the.names <- colnames(Q)
  }
  
  word2index <- seq(length(the.names))
  names(word2index) <- the.names
  
  return(word2index)
  
}

# Given a saved brms model file, return the md5sum checksum of the 
# file with just the posterior samples.
stanfile_checksum <- function(inputfilename){
  stan_model <- readRDS(file=inputfilename)
  samples <- brms::posterior_samples(stan_model)
  tmpfilename <- tempfile()
  saveRDS(samples, file=tmpfilename)
  checksum <- unname(tools::md5sum(tmpfilename))
  file.remove(tmpfilename)
  assertthat::assert_that(!file.exists(tmpfilename))
  checksum
}

# In the given directory, get all filenames matching a pattern
# Return file names in list, indexed by the name of the stan/brms
# model that the file refers to.

model_filename_pattern <- 'stan_recognition_model__v[1-6]__(k3|k2_cp|k2_pa|k2_ca|k1_p|k1_c|k1_a|null)__seed_[[:digit:]]+__chains_[[:digit:]]__warmup_[[:digit:]]+__iterations_[[:digit:]]+__samples_[[:digit:]]+\\.Rds'
model_loowaic_filename_pattern <- 'stan_recognition_model__v[1-6]__(k3|k2_cp|k2_pa|k2_ca|k1_p|k1_c|k1_a|null)__seed_[[:digit:]]+__chains_[[:digit:]]__warmup_[[:digit:]]+__iterations_[[:digit:]]+__samples_[[:digit:]]+__loowaic\\.Rds'

get_stanfilenames <- function(cache_directory, filename){
  
  get_modelname <- function(filename){
    filename_split <- stringr::str_split(filename, pattern = '__')[[1]]
    paste('M', filename_split[2], filename_split[3], sep = '_')
  }
  
  stanfilenames <- list.files(cache_directory,
                              full.names = T,
                              pattern = filename)
  
  names(stanfilenames) <- lapply(stanfilenames, get_modelname) %>% unlist()
  
  stanfilenames
}

get_stanmodel_filenames <- function(cache_directory){
  get_stanfilenames(cache_directory, model_filename_pattern)
}

get_stanmodel_loowaic_filenames <- function(cache_directory){
  get_stanfilenames(cache_directory, model_loowaic_filename_pattern)
}


# Return list of all stan/brms models in the given directory.
# Return models in list, indexed by the name of the model
get_stanmodels <- function(cache_directory){
  
  lapply(get_stanfilenames(cache_directory), 
         readRDS)
  
}

## Catch warnings
## Taken from http://stackoverflow.com/a/4947528
withWarnings <- function(expr) {
  myWarnings <- NULL
  wHandler <- function(w) {
    myWarnings <<- c(myWarnings, list(w))
    invokeRestart("muffleWarning")
  }
  val <- withCallingHandlers(expr, warning = wHandler)
  list(value = val, warnings = myWarnings)
} 

#####################################################
# Functions for use with rjags/Jags
#####################################################
make.job.parameters <- function(data, job.names, job.filenames, job.seeds, n.chains){
  
  #assert_that(all(sapply(job.names, function(job.name) job.name %in% names(data))))

  job.parameters <- lapply(seq(length(job.names)),
                           function(i) list('job.seed' = job.seeds[i],
                                            'job.filename' = job.filenames[[job.names[i]]],
                                            'n.chains' = n.chains,
                                            'data' = data[[job.names[i]]]))
  
  names(job.parameters) <- job.names
  
  job.parameters
}

make.jags.seeds <- function(n.chains=3, seed=42){
  set.seed(seed)
  jags.seeds <- get.randint(n.chains, min=101, max=100001)
  lapply(jags.seeds,
         function(k) list(.RNG.name="base::Super-Duper", .RNG.seed=k))
}

model.initialize <- function(n.update, params){
  M <- jags.model(params[['job.filename']],
                  data=params[['data']],
                  inits=make.jags.seeds(n.chains = params[['n.chains']], seed=params[['job.seed']]),
                  n.chains = params[['n.chains']])
  
  update(M, n.update)
  
  M 
}

model.update <- function(M, n.update){
  
  M$recompile()
  update(M, n.update)
  
  M
  
}

model.sample <- function(M, variable.names, n.samples, thin=10){
  
  M$recompile()
  coda.samples(M, variable.names = variable.names, n.iter = n.samples, thin=thin)
  
}

model.dic <- function(M, n.iter, thin=10){
  
  M$recompile()
  dic.samples(M, n.iter = n.iter, thin=thin)
  
}

##################################################################
# Parallel functions for running (the above) rjags/Jags functions
##################################################################

parallel.initialize <- function(the.cluster, job.parameters, n.update){
  parLapply(the.cluster,
            job.parameters,
            function(params) model.initialize(n.update, params))
}

parallel.update <- function(the.models, n.iter, the.cluster){
  parLapply(the.cluster,
            the.models,
            function(the.model) model.update(the.model, n.iter))
}

parallel.sample <- function(the.cluster, the.models, variable.names, n.samples, thin=10){
  parLapply(the.cluster,
            the.models,
            function(the.model) model.sample(the.model, variable.names = variable.names, n.samples=n.samples, thin = thin))
}


parallel.dic <- function(the.cluster, the.models, n.iter, thin=10){
  parLapply(the.cluster,
            the.models,
            function(the.model) model.dic(the.model, n.iter, thin=thin))
}

# ##############################################################
# Functions for reading in the preprocessed experimental data
# ##############################################################

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
  
  rescale_and_log <- function(x) re.scale(log(x))
  
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

  Df %>% mutate(posterior.predictions = rescale_and_log(posterior.predictions),
                association.predictions = rescale_and_log(association.predictions),
                cooccurrence.predictions = rescale_and_log(cooccurrence.predictions)
  )
  
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

# ========================================================================

# Get the predicted probability of the recalled words according to each of 
# our three models, i.e. Topic model, cooccurrence model, association model.
# Check that all the right information is in all three data sets. Specifically,
# check if the rows and columns of the matrices are identical. Rows are text names.
# Columns are the word names.
# Return a list that has a word2index vector and a list of matrices, 
# one for each of the three models.
get.predicted.recall.probabilities <- function(posterior_predictions_csv_filename,
                                               cooccurrence_predictions_csv_filename,
                                               association_predictions_csv_filename) 
{
  
  posterior.predictions.recall <- as.matrix(
    read.csv(file.path(cache_directory, posterior_predictions_csv_filename), 
             check.names = F)
  )
  
  cooccurrence.predictions.recall <- as.matrix(
    read.csv(file.path(cache_directory, cooccurrence_predictions_csv_filename), 
             check.names = F)
  )
  
  association.predictions.recall <- as.matrix(
    read.csv(file.path(cache_directory, association_predictions_csv_filename), 
             check.names = F)
  )
  
  posterior.recalledwords.word2index <- get.word2index(posterior.predictions.recall, row=F)
  posterior.texts.word2index <- get.word2index(posterior.predictions.recall, row=T)
  
  cooccurrence.recalledwords.word2index <- get.word2index(cooccurrence.predictions.recall, row=F)
  cooccurrence.texts.word2index <- get.word2index(cooccurrence.predictions.recall, row=T)
  
  association.recalledwords.word2index <- get.word2index(cooccurrence.predictions.recall, row=F)
  association.texts.word2index <- get.word2index(cooccurrence.predictions.recall, row=T)
  
  assert_that(are_equal(posterior.recalledwords.word2index, cooccurrence.recalledwords.word2index))
  assert_that(are_equal(posterior.texts.word2index, cooccurrence.texts.word2index))
  
  assert_that(are_equal(posterior.recalledwords.word2index, association.recalledwords.word2index))
  assert_that(are_equal(posterior.texts.word2index, association.texts.word2index))
  
  # Return value
  list(word2index = cooccurrence.recalledwords.word2index,
       model.predictions =   list(pp = posterior.predictions.recall,
                                  cc = cooccurrence.predictions.recall,
                                  aa = association.predictions.recall)
  )
}
# ========================================================================


#= 

# Get the recalled word from the memory experiment
# Verify that for every each recalled word, we have
# a predicted probability from each model.
get.recall.dataframe <- function(experiment_recall_results_csv_filename, 
                                 model.predictions){
  
  Df <- read.csv(file.path(cache_directory, 
                           experiment_recall_results_csv_filename)
  )
  
  Df['X'] <- NULL
  
  # Test if the prediction probabilities in Df match those of the matrices
  for (i in seq(dim(Df)[1])){
    
    text.id <- Df[i, 'text']
    word <- Df[i, 'word']
    pp <- Df[i, 'posterior.predictions']
    cc <- Df[i, 'cooccurrence.predictions']
    aa <- Df[i, 'association.predictions']
    
    assert_that(are_equal(model.predictions[['pp']][text.id, word], pp))
    assert_that(are_equal(model.predictions[['cc']][text.id, word], cc))
    assert_that(are_equal(model.predictions[['aa']][text.id, word], aa))
  }
  
  # Return value
  Df[c('subject', 'text', 'word')]
}

# ========================================================================

# Using the recall data data-frame, prepare the data sets to be passed 
# to Jags. Use the word2index vector from any one the model predictions,
# they are the same, to produce the words vector.
get.mcmc.data <- function(Df, word2index, model.predictions){
  
  N <- dim(Df)[1]
  subject <- as.numeric(Df$subject)
  words <- as.vector(sapply(Df$word, function(w) word2index[w]))
  text <- Df$text
  J <- length(levels(Df$subject))
  # We could use pp, aa, or cc here as the matrices
  # are the same size.
  L <- dim(model.predictions[['pp']])[2]
  K <- dim(model.predictions[['pp']])[1]
  
  data <- list(subject = subject,
               text = text,
               y = words,
               N = N,
               J = J,
               L = L,
               K = K,
               psi = NA)
  
  data <- list('pp'=data,
               'cc'=data,
               'aa'=data)
  
  data[['pp']][['psi']] <- log(model.predictions[['pp']])
  data[['cc']][['psi']] <- log(model.predictions[['cc']])
  data[['aa']][['psi']] <- log(model.predictions[['aa']])
  
  data
  
}


