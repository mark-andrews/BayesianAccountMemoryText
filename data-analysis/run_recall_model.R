# ======================================= LOAD PACKAGES ======================================================
library(dplyr)
library(assertthat)
source('utils.R')
library(rjags)

# ================================ GENERAL PARAMETERS =========================================================
cache_directory <- 'cache/'
n_chains <- 3
n_update <- 1000
n_thin <- 10

model_names <- c('k3',
                 'k2_pa',
                 'k2_pc',
                 'k2_ca',
                 'k1_p',
                 'k1_c',
                 'k1_a',
                 'null')

master_seed <- 1234321
set.seed(master_seed)
model_seeds <- floor(runif(length(model_names),
                     min=1e5,
                     max=1e6)
)
names(model_seeds) <- model_names

# ================================ PARSE INPUT ARGUMENTS ======================================================
args = commandArgs(trailingOnly = TRUE)
model <- args[1]         # e.g. k1_p
assert_that(model %in% model_names)

# =========================================== PREPARE DATA ====================================================
model.predictions.list <- get.predicted.recall.probabilities('posterior_predictions_of_recalled_words.csv',
                                                             'cooccurrences_predictions_of_recalled_words.csv',
                                                             'association_predictions_of_recalled_words.csv')

word2index <- model.predictions.list[['word2index']]

model.predictions <- model.predictions.list[['model.predictions']]

rescale_and_log <- function(x) as.numeric(scale(log(x)))

Df <- get.recall.dataframe('experiment_brisbane_recall_memory_tests_results.csv',
                           model.predictions) 

psi_pp <- apply(model.predictions[['pp']], 1, rescale_and_log) %>% t()
psi_cc <- apply(model.predictions[['cc']], 1, rescale_and_log) %>% t()
psi_aa <- apply(model.predictions[['aa']], 1, rescale_and_log) %>% t()

jags_data <- within(list(), {
  
  N <- nrow(Df)
  
  subject <- as.numeric(Df$subject)
  
  y <- as.vector(sapply(Df$word, function(w) word2index[w]))
  
  text <- Df$text
  
  J <- length(levels(Df$subject))
  
  # We could use pp, aa, or cc here as the matrices
  # are the same size.
  L <- dim(model.predictions[['pp']])[2]
  K <- dim(model.predictions[['pp']])[1]
  
})


if (model == 'k1_p') {
  
  jags_data[['psi_a']] <- psi_pp
  
  jags_model <- 'jags/recall_memory_model_k1.jags'
  variables.to.sample <- c('b')
  
} else if (model == 'k1_c') {
  
  jags_data[['psi_a']] <- psi_cc
  
  jags_model <- 'jags/recall_memory_model_k1.jags'
  variables.to.sample <- c('b')
  
} else if (model == 'k1_a') {
  
  jags_data[['psi_a']] <- psi_aa
  
  jags_model <- 'jags/recall_memory_model_k1.jags'
  variables.to.sample <- c('b')
  
} else if (model == 'k2_pa') {
  
  jags_data[['psi_a']] <- psi_pp
  jags_data[['psi_b']] <- psi_aa

  jags_model <- 'jags/recall_memory_model_k2.jags'
  variables.to.sample <- c('b_a', 'b_b')
  
} else if (model == 'k2_pc') {
  
  jags_data[['psi_a']] <- psi_pp
  jags_data[['psi_b']] <- psi_cc
  
  jags_model <- 'jags/recall_memory_model_k2.jags'
  variables.to.sample <- c('b_a', 'b_b')
  
} else if (model == 'k2_ca') {
  
  jags_data[['psi_a']] <- psi_cc
  jags_data[['psi_b']] <- psi_aa
  
  jags_model <- 'jags/recall_memory_model_k2.jags'
  variables.to.sample <- c('b_a', 'b_b')
  
} else if (model == 'k3') {
  
  jags_data[['psi_a']] <- psi_pp
  jags_data[['psi_b']] <- psi_cc
  jags_data[['psi_c']] <- psi_aa
  
  jags_model <- 'jags/recall_memory_model_k3.jags'
  variables.to.sample <- c('b_a', 'b_b', 'b_c')
  
} else if (model == 'null'){
  
  jags_model <- 'jags/recall_memory_model_null.jags'
  variables.to.sample <- c('b')
}

# ==================================== SOME UTILITIES ==========================================================
model_initialize <- function(n_update, chain_seeds){
  

}



# ====================================================================================================

set.seed(model_seeds[[model]])
chain_seeds <- floor(runif(n_chains,
                           min=1e5,
                           max=1e6)
)

M <- jags.model(jags_model,
                data=jags_data,
                inits=lapply(chain_seeds,
                             function(k) list(.RNG.name="base::Super-Duper", .RNG.seed=k)
                ),
                n.chains = 3)

update(M, n_update)
S <- coda.samples(M, variable.names = variables.to.sample, n.iter = n_update, thin=n_thin)
#D <- model.dic(M, n.iter =n_update, thin=n_thin)
D <- dic.samples(M, n.iter = n_update, thin=n_thin)
density_samples <- coda.samples(M, variable.names = c('density'), n.iter = n_update, thin=n_thin)

get_pointwise_elpd <- function(x) log(apply(x, 2, mean))
get_lppd <- function(x) -2 * sum(get_pointwise_elpd(x))
get_pwaic <- function(x) sum(apply(log(x), 2, var))
get_waic <- function(x) get_lppd(x) - get_pwaic(x)
get_waic_se <- function(x) sqrt(ncol(x)*var(get_pointwise_elpd(x)))

density_samples <- do.call(rbind, density_samples)


#####################################

# M <- parallel.initialize(the.cluster, job.parameters, n.iterations)
# S <- parallel.sample(the.cluster, M, variables.to.sample, n.iterations, thin=n.thin)
# D <- parallel.dic(the.cluster, M, n.iterations, thin=n.thin)
# 
list.to.save <- list(data=jags_data,
                     M=M,
                     S=S,
                     D=D,
                     density_samples = density_samples,
                     waic = get_waic(density_samples),
                     waic_se = get_waic_se(density_samples),
                     master.seed=master_seed,
                     n.iterations=n_update,
                     n.thin=n_thin,
                     variables.to.sample= variables.to.sample,
                     jags.filename=jags_model)

saveRDS(list.to.save, 
	file = file.path(cache_directory,
			 sprintf('recall_save_%s.Rds', model))
)
