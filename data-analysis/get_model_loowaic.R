# A script to calculate loo and waic for a specific brms/stan model
# Intended to be run from the OS command line using Rscript.
# 
# For example,
# Run model k3 version v6 
# > Rscript get_model_loowaic.R v6 k3
# 
# The loo calculations can take up to 50 minutes even on a fast machine, 
# e.g. a Xeon Gold 6154 3GHz), and they consume a ridiculous amount of RAM, 
# i.e., around 40-50GB at its peak.


library(brms)
library(assertthat)
source('utils.R')

args = commandArgs(trailingOnly = TRUE)
model_key <- args[2]
model_version <- args[1]

# Only versions v1, v2 ... v6 are recognized.
# Likewise, there are only 8 acceptable model names.

assert_that(
  model_version %in% paste0('v', seq(6))
)

assert_that(
  model_key %in% c(
    'k3',
    'k2_cp',
    'k2_pa',
    'k2_ca',
    'k1_p',
    'k1_c',
    'k1_a',
    'null'
  )
)

cache_directory <- '../cache/'

# We assume any required model has been run and is saved in the cache directory.
# Get model name to file name map.
model_filenames <- get_stanmodel_filenames(cache_directory)
model_name <- sprintf('M_%s_%s', model_version, model_key)

M <- readRDS(file = model_filenames[[model_name]])

M_loo <- loo(M) # This could take up to 50 minutes
M_waic <- waic(M)

M_loo$model_name <- sprintf('%s__%s', model_version, model_key)
M_waic$model_name <- sprintf('%s__%s', model_version, model_key)

save_filename <- sprintf('%s__loowaic.Rds', basename(tools::file_path_sans_ext(model_filenames[[model_name]])))

saveRDS(list(M_loo = M_loo,
             M_waic = M_waic),
        file=file.path(cache_directory, save_filename)
        )
