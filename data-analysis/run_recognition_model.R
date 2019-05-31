# ================================= LOAD PACKAGES ETC =========================================================
library(brms)
source('utils.R')

# ================================ PARSE COMMAND LINE ARGUMENTS ===============================================
args = commandArgs(trailingOnly = TRUE)
model_version <- args[1]         # e.g. v1
model_sub_version <- args[2]     # e.g. k3

# ================================= OTHER PARAMETERS =========================================================
n_iterations <- 10000
n_warmup <- floor(n_iterations/2)
n_chains <- 4
n_cores <- n_chains

# ================================= READ AND PREPARE THE DATA ==============================================
cache_directory <- '../cache'
recognition_data_filename <- 'experiment_brisbane_recognition_memory_tests.csv'

Df <- get.recognition.data(file.path(cache_directory, 
                                     recognition_data_filename)
)

# ================================== CREATE ALL MODEL FORMULAE ================================================

model_version_1_template <- 'response ~ expected + %s + (1|text) + (1|subject) + (1|word)'
model_version_2_template <- 'response ~ expected + %s + (%s + expected|text) + (%s + expected|subject) + (1|word)'
model_version_3_template <- 'response ~ expected + %s + (%s + expected|text) + (%s|subject) + (1|word)'
model_version_4_template <- 'response ~ expected * %s + (%s + expected|text) + (%s|subject) + (1|word)'
model_version_5_template <- 'response ~ expected * %s + (%s + expected|text) + (%s + expected|subject) + (1|word)'
model_version_6_template <- 'response ~ expected * %s + (%s * expected|text) + (%s * expected|subject) + (1|word)'

model_version_1_null_template <- 'response ~ expected + (1|text) + (1|subject) + (1|word)'
model_version_2_null_template <- 'response ~ expected + (expected|text) + (expected|subject) + (1|word)'
model_version_3_4_null_template <- 'response ~ expected + (expected|text) + (1|subject) + (1|word)'
model_version_5_6_null_template <- 'response ~ expected + (expected|text) + (expected|subject) + (1|word)'

model_variables <- within(list(),{
  k3 <- '(cooccurrence.predictions + posterior.predictions + association.predictions)'
  k2_cp <- '(cooccurrence.predictions + posterior.predictions)'
  k2_pa <- '(posterior.predictions + association.predictions)'
  k2_ca <- '(cooccurrence.predictions + association.predictions)'
  k1_p <- 'posterior.predictions'
  k1_c <- 'cooccurrence.predictions'
  k1_a <- 'association.predictions'
})

make_model_v2_v6 <- function(template){
  lapply(model_variables, 
         function(model_variables){
           as.formula(sprintf(template,
                              model_variables, 
                              model_variables,
                              model_variables))}
  )
}

models <- within(list(), {
  v1 <- lapply(model_variables, 
               function(model_variables){
                 sprintf(model_version_1_template,
                         model_variables)
                 }
  )
  # models v2 to v6 can be made using the same function
  v2 <- make_model_v2_v6(model_version_2_template)
  v3 <- make_model_v2_v6(model_version_3_template)
  v4 <- make_model_v2_v6(model_version_4_template)
  v5 <- make_model_v2_v6(model_version_5_template)
  v6 <- make_model_v2_v6(model_version_6_template)
})

# There must be some more elegant way of doing this...
models$v1$null <- model_version_1_null_template
models$v2$null <- model_version_2_null_template
models$v3$null <- model_version_3_4_null_template
models$v4$null <- model_version_3_4_null_template
models$v5$null <- model_version_5_6_null_template
models$v6$null <- model_version_5_6_null_template

# ==================================== SET UP SEEDS FOR MCMC MODELS ============================================

master_seed <- 1010101
set.seed(master_seed)

model_version_names <- names(models)
n_versions <- length(model_version_names)
model_sub_version_names <- names(models[['v6']]) # use any version; they've all the same element names
n_sub_versions <- length(model_sub_version_names)

seeds <- matrix(floor(runif(n_versions * n_sub_versions, 1e5, 1e6)),
                nrow = n_versions,
                ncol = n_sub_versions,
                byrow = TRUE,
                dimnames = list(sort(model_version_names),
                                sort(model_sub_version_names))
)

# ==================================== RUN MODEL AND SAVE IT ======================================================

M <- brm(as.formula(models[[model_version]][[model_sub_version]]),
         data = Df,
         warmup = n_warmup,
         iter = n_iterations,
         chains = n_chains,
         cores = n_cores,
         seed = seeds[model_version, model_sub_version],
         family = bernoulli()
)

saveRDS(M, 
        file=file.path(cache_directory,
                       sprintf('stan_recognition_model__%s__%s__seed_%d__chains_%d__warmup_%d__iterations_%d__samples_%d.Rds', 
                               model_version, 
                               model_sub_version,
                               seeds[model_version, model_sub_version],
                               n_chains,
                               n_warmup,
                               n_iterations,
                               (n_iterations-n_warmup)*n_chains # total post warmp-up iterations
                               ))
        )
