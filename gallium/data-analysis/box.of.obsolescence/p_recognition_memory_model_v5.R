library(parallel)
library(coda)

rm(list=ls()) # clear all

source('utils.R')
source('p_recognition_memory_model_common.R')
source('parutils.R')

job.names <- c('pp', 'cc', 'aa')
n.chains <- 3

the.seed <- 10101
set.seed(the.seed)

n.iterations  <- 10000

job.seeds <- get.randint(length(job.names))

job.parameters <- make.job.parameters(data, job.names, job.seeds, n.chains)

the.cluster <- makeCluster(length(job.names), type='FORK')

clusterEvalQ(cl=the.cluster, library('rjags'));
clusterExport(cl=the.cluster, c('make.jags.seeds', 'model.initialize', 'model.update', 'model.sample', 'model.dic', 'n.iterations'))

M <- parallel.initialize(the.cluster, job.parameters, 'jags/recognition_memory_model_v5.jags', n.iterations)
S <- parallel.sample(the.cluster, M, c('b', 'beta', 'alpha.0'), n.iterations, thin=10)
D <- parallel.dic(the.cluster, M, n.iterations, thin=10)

save('data', 'M', 'S', 'D', 'the.seed', 'n.iterations',
     file = file.path(cache.directory, 
                      sprintf('mcmc_recognition_model_v5_seed_%d_iterations_burn_%d_sample_%d_dic_%d_date_%s.Rda',
                              the.seed,
                              n.iterations,
                              n.iterations,
                              n.iterations,
                              format(Sys.time(), '%Y.%m.%d_%H.%M.%S')
                      )
     )
)
