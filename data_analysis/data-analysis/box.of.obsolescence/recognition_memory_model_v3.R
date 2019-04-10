library(rjags)

rm(list=ls()) # clear all

source('utils.R')
source('recognition_memory_model_common.R')

n.chains <- 3
M.pp <- jags.model('recognition_memory_model_v3.jags', 
                data=data.pp,
                inits=make.seeds(n.chains = n.chains, seed=101),
                n.chains = n.chains)

M.cc <- jags.model('recognition_memory_model_v3.jags', 
                data=data.cc,
                inits=make.seeds(n.chains = n.chains, seed=1001),
                n.chains = 3)

M.aa <- jags.model('recognition_memory_model_v3.jags', 
                   data=data.aa,
                   n.chains = 3)

update(M.pp, 1e4)
update(M.cc, 1e4)
update(M.aa, 1e4)

S.pp <- coda.samples(M.pp, variable.names = c('b', 'beta', 'alpha.0'), 10000)
S.cc <- coda.samples(M.cc, variable.names = c('b', 'beta', 'alpha.0'), 10000)

S.aa <- coda.samples(M.aa, variable.names = c('ll'), 10000)

