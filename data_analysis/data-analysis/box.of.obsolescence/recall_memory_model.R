library(rjags)
library(runjags)
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
inits <- make.seeds(n.chains)

parsamples <- run.jags('recall_memory_model.jags',
                       data=data,
                       monitor=c('b','b.text.sigma', 'b.subject.sigma'),
                       burnin=1000, 
                       sample=1000, 
                       method='parallel', 
                       inits=inits,
                       n.chains = n.chains)


parsamples <- autorun.jags('recall_memory_model.jags',
                           data=data,
                           monitor=c('b','b.text.sigma', 'b.subject.sigma'),
                           method='parallel', 
                           inits=make.seeds(n.chains),
                           n.chains = n.chains)






S <- as.mcmc.list(parsamples)