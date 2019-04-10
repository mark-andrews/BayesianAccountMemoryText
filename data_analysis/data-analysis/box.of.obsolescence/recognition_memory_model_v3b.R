library(runjags)
library(coda)

rm(list=ls()) # clear all

source('utils.R')
source('recognition_memory_model_common.R')

set.seed(1001)

n.chains = 3

monitor=c('deviance', 'pd', 'dic')

f <- function(data.set){
  autorun.jags('recognition_memory_model_v3.jags',
               data=data.set,
               monitor=monitor,
               #method='parallel', 
               inits=make.seeds(n.chains),
               n.chains = n.chains)
}

M.pp <- f(data.pp)
#S.pp <- as.mcmc.list(M.pp)
#sapply(seq(3), function(k) sum(lppd.waic(S.pp, k)))

M.cc <- f(data.cc)
#S.cc <- as.mcmc.list(M.cc)
#sapply(seq(3), function(k) sum(lppd.waic(S.cc, k)))