rm(list=ls()) # clear all

source('recognition_memory_model_common.R')

data.pp$L <- NULL
data.pp$item <- NULL

data.cc$L <- NULL
data.cc$item <- NULL

M.pp <- jags.model('recognition_memory_model_v1_a.jags', 
                data=data.pp,
                n.chains = 3)

M.cc <- jags.model('recognition_memory_model_v1_a.jags', 
                data=data.cc,
                n.chains = 3)

update(M.pp, 1e2)
update(M.cc, 1e4)

S.pp <- coda.samples(M.pp, variable.names = c('ll'), 10000)
S.cc <- coda.samples(M.cc, variable.names = c('ll'), 10000)
