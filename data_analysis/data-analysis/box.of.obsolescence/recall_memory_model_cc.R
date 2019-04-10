library(rjags)
library(coda)

rm(list=ls()) # clear all

set.seed(10001)

source('recall_memory_model_common2.R')

make.seeds <- function(n.chains=3){
  jags.seeds <- ceiling(runif(n.chains, 101, 100001))
  lapply(jags.seeds,
         function(k) list(.RNG.name="base::Super-Duper", .RNG.seed=k))
}

n.chains <- 3

data.cc <- data
data.cc$psi = log(cooccurrence.predictions.recall)
inits <- make.seeds(n.chains)

M.cc <- jags.model('recall_memory_model.jags',
                   data.cc,
                   n.chains=n.chains,
                   inits=inits)

update(M.cc, 10000)

######

data.pp <- data
data.pp$psi = log(posterior.predictions.recall)
inits <- make.seeds(n.chains)

M.pp <- jags.model('recall_memory_model.jags',
                   data.pp,
                   n.chains=n.chains,
                   inits=inits)

update(M.pp, 10000)