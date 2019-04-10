library(parallel)
library(assertthat)

parallel.initialize <- function(the.cluster, job.paramters, model.filename, n.update){
  parLapply(the.cluster,
            job.parameters,
            function(params) model.initialize(model.filename, n.update, params))
}

parallel.update <- function(the.models, n.iter, the.cluster){
  parLapply(the.cluster,
            the.models,
            function(the.model) model.update(the.model, n.iter))
}

parallel.sample <- function(the.cluster, the.models, variable.names, n.samples, thin=10){
  parLapply(the.cluster,
            the.models,
            function(the.model) model.sample(the.model, variable.names = variable.names, n.samples=n.samples, thin = thin))
}

make.job.parameters <- function(data, job.names, job.seeds, n.chains){
  
  assert_that(all(sapply(job.names, function(job.name) job.name %in% names(data))))

  job.parameters <- lapply(seq(length(job.names)),
                           function(i) list('job.seed' = job.seeds[i],
                                            'n.chains' = n.chains,
                                            'data' = data[[job.names[i]]]))
  
  names(job.parameters) <- job.names
  
  job.parameters
}

parallel.dic <- function(the.cluster, the.models, n.iter, thin=10){
  parLapply(the.cluster,
            the.models,
            function(the.model) model.dic(the.model, n.iter, thin=thin))
}