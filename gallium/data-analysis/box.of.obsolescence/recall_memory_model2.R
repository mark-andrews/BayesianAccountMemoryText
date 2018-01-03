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

parsamples.pp <- run.jags('recall_memory_model.jags',
                          data=data,
                          monitor=c('b','b.text.sigma', 'b.subject.sigma'),
                          burnin=1000, 
                          sample=1000, 
                          method='parallel', 
                          inits=inits,
                          n.chains = n.chains)

x.pp <- extend.jags(parsamples.pp,
                    add.monitor = c('density'), 
                    drop.monitor = c('b','b.text.sigma', 'b.subject.sigma'),
                    sample=1000)


inits <- make.seeds(n.chains)
data.cc <- data
data.cc$psi = log(cooccurrence.predictions.recall)

parsamples.cc <- run.jags('recall_memory_model.jags',
                          data=data.cc,
                          monitor=c('b','b.text.sigma', 'b.subject.sigma'),
                          burnin=1000, 
                          sample=1000, 
                          method='parallel', 
                          inits=inits,
                          n.chains = n.chains)

x.cc <- extend.jags(parsamples.cc,
                    add.monitor = c('density'), 
                    drop.monitor = c('b','b.text.sigma', 'b.subject.sigma'),
                    sample=1000)  



S.pp <- as.mcmc.list(parsamples)
S.cc <- as.mcmc.list(parsamples.cc)


M.pp <- jags.model('recall_memory_model.jags',
                   data = data,
                   n.chains = 1,
                   1000)