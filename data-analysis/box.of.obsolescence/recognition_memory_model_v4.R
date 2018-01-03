library(rjags)

rm(list=ls()) # clear all

set.seed(1001)

source('utils.R')
source('recognition_memory_model_common.R')

make.seeds <- function(n.chains=3, seed=42){
  set.seed(seed)
  jags.seeds <- ceiling(runif(n.chains, 101, 100001))
  lapply(jags.seeds,
         function(k) list(.RNG.name="base::Super-Duper", .RNG.seed=k))
}

n.chains <- 3

M.pp <- jags.model('jags/recognition_memory_model_v4.jags', 
                data=data.pp,
                inits=make.seeds(n.chains=n.chains, seed=101),
                n.chains = 3)

M.cc <- jags.model('jags/recognition_memory_model_v4.jags', 
                data=data.cc,
                inits=make.seeds(n.chains=n.chains, seed=102),
                n.chains = 3)

M.aa <- jags.model('jags/recognition_memory_model_v4.jags', 
                   data=data.aa,
                   inits=make.seeds(n.chains=n.chains, seed=103),
                   n.chains = 3)

update(M.pp, 1e4)
S.pp <- coda.samples(M.pp, variable.names = c('alpha.0', 'b', 'beta', 'gamma'), 10000)
gelman.diag(S.pp)
update(M.pp, 1e4)
S.pp <- coda.samples(M.pp, variable.names = c('alpha.0', 'b', 'beta', 'gamma'), 10000)
gelman.diag(S.pp)
dic.samples(M.pp, 10000, thin=10, type='pD')

update(M.cc, 1e4)
S.cc <- coda.samples(M.cc, variable.names = c('alpha.0', 'b', 'beta', 'gamma'), 10000)
gelman.diag(S.cc)
dic.samples(M.cc, 10000, thin=10, type='pD')

update(M.aa, 1e4)
S.aa <- coda.samples(M.aa, variable.names = c('alpha.0', 'b', 'beta', 'gamma'), 10000)
gelman.diag(S.aa)
dic.samples(M.aa, 10000, thin=10, type='pD')