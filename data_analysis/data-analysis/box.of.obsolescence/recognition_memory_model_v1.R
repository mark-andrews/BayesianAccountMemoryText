rm(list=ls()) # clear all

source('recognition_memory_model_common.R')

M.pp <- jags.model('recognition_memory_model_v1.jags', 
                data=data.pp,
                n.chains = 3)

M.cc <- jags.model('recognition_memory_model_v1.jags', 
                data=data.cc,
                n.chains = 3)

update(M.pp, 1e4)
update(M.cc, 1e4)

S.pp <- coda.samples(M.pp, variable.names = c('ll'), 10000)
S.cc <- coda.samples(M.cc, variable.names = c('ll'), 10000)
