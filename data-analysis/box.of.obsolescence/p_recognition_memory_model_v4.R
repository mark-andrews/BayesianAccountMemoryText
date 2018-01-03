library(parallel)
library(coda)

rm(list=ls()) # clear all

source('utils.R')
source('p_recognition_memory_model_common.R')
source('parutils.R')

job.names <- c('pp', 'cc', 'aa')
n.chains <- 3

the.seed <- 1801
set.seed(the.seed)

n.iterations <- 2

job.seeds <- get.randint(length(job.names))

job.parameters <- make.job.parameters(data, job.names, job.seeds, n.chains)

the.cluster <- makeCluster(length(job.names), type='FORK')

clusterEvalQ(cl=the.cluster, library('rjags'));
clusterExport(cl=the.cluster, c('make.jags.seeds', 'model.initialize', 'model.update', 'model.sample', 'model.dic'))
clusterExport(cl=the.cluster, c('n.iterations'))

M <- parallel.initialize(the.cluster, job.parameters, 'jags/recognition_memory_model_v4.jags', n.iterations)
S <- parallel.sample(the.cluster, M, c('gamma', 'b', 'beta', 'alpha.0'), n.iterations, thin=10)
D <- parallel.dic(the.cluster, M, n.iterations, thin=10)

save('data', 'M', 'S', 'D', 'the.seed', 'n.iterations',
     file = file.path(cache.directory, 
                      sprintf('mcmc_recognition_model_v4_seed_%d_iterations_burn_%d_sample_%d_dic_%d_date_%s.Rda',
                              the.seed,
                              n.iterations,
                              n.iterations,
                              n.iterations,
                              format(Sys.time(), '%Y.%m.%d_%H.%M.%S')
                      )
     )
)

# Results of analyses, as seen on 11:26, Apr 17, 2017
# Here, we used most recent values of the predictors, 
# which were recaled too.
# This simulation was done with 10000 iterations burn-in, 
# followed by 10000 for sampling, followed by 10000 for DIC.
# Seed was 1801.

# > gelman.diag(S[[1]], multivariate = F)
# Potential scale reduction factors:
#   
#   Point est. Upper C.I.
# alpha.0       1.00       1.02
# b             1.02       1.06
# beta          1.01       1.03
# gamma         1.03       1.10
# 
# > gelman.diag(S[[2]], multivariate = F)
# Potential scale reduction factors:
#   
#   Point est. Upper C.I.
# alpha.0          1       1.00
# b                1       1.00
# beta             1       1.00
# gamma            1       1.01
# 
# > gelman.diag(S[[3]], multivariate = F)
# Potential scale reduction factors:
#   
#   Point est. Upper C.I.
# alpha.0          1          1
# b                1          1
# beta             1          1
# gamma            1          1

# 
# $pp
# Mean deviance:  4657 
# penalty 581 
# Penalized deviance: 5238 
# 
# $cc
# Mean deviance:  4670 
# penalty 589.9 
# Penalized deviance: 5260 
# 
# $aa
# Mean deviance:  4639 
# penalty 676.6 
# Penalized deviance: 5316 