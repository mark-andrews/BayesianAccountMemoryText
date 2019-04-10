# Rscript mcmc_recall_memory_analysis_script.R [VARIABLES TO SAMPLE] [SEED] [ITERATIONS] [THIN]
# Rscript mcmc_recall_memory_analysis_script.R 'b' 1401 1000 10
args = commandArgs(trailingOnly = TRUE)

# List the variables you want to coda.sample like this 'a,b,c'
# I.e. quoted AND comma separated AND no-spaces
variables.to.sample.str <- args[1]
variables.to.sample <- unlist(strsplit(variables.to.sample.str, ','))

# 
master.seed <- as.integer(args[2])

# 
n.iterations <- as.integer(args[3])

# 
n.thin <- as.integer(args[4])

# ========================================================================
# ========================================================================

source('utils.R')

cache.directory <- '../cache'

job.names <- c('pp', 'cc', 'aa')
jags.filename <- 'jags/recall_memory_model.jags'
job.filenames <- list(pp = jags.filename,
                      cc = jags.filename,
                      aa = jags.filename)
n.chains <- 3
set.seed(master.seed)
job.seeds <- get.randint(length(job.names))

save.filename.template <- 'mcmc_recall_model_seed_%d_burn_%d_sample_%d_dic_%d_thin_%d.Rda'
save.filename <- file.path(cache.directory, 
                           sprintf(save.filename.template,
                                   master.seed,
                                   n.iterations,
                                   n.iterations,
                                   n.iterations,
                                   n.thin)
)

# This checksum will be correct only if all the above settings are kept exactly.
save.filename.checksum <- "53d79e8c6b05ba6247ee0597348d7f2d"

# ===============================================================================
model.predictions.list <- get.predicted.recall.probabilities('posterior_predictions_of_recalled_words.csv',
                                                             'cooccurrences_predictions_of_recalled_words.csv',
                                                             'association_predictions_of_recalled_words.csv')


word2index <- model.predictions.list[['word2index']]
model.predictions <- model.predictions.list[['model.predictions']]
Df <- get.recall.dataframe('experiment_brisbane_recall_memory_tests_results.csv', 
                           model.predictions)

data <- get.mcmc.data(Df, word2index, model.predictions)

job.parameters <- make.job.parameters(data, job.names, job.filenames, job.seeds, n.chains)

the.cluster <- makeCluster(length(job.names), type='FORK')

clusterEvalQ(cl=the.cluster, 
             library('rjags'));
clusterExport(cl=the.cluster, 
              c('make.jags.seeds', 
                'model.initialize', 
                'model.update', 
                'model.sample', 
                'model.dic'))

M <- parallel.initialize(the.cluster, job.parameters, n.iterations)
S <- parallel.sample(the.cluster, M, variables.to.sample, n.iterations, thin=n.thin)
D <- parallel.dic(the.cluster, M, n.iterations, thin=n.thin)

list.to.save <- list(data=data,
                     M=M,
                     S=S,
                     D=D,
                     master.seed=master.seed,
                     n.iterations=n.iterations,
                     n.thin=n.thin,
                     variables.to.sample.str=variables.to.sample.str,
                     jags.filename=jags.filename,
                     jags.filename.checksum=get.checksum(jags.filename))

saveRDS(list.to.save, file = save.filename)
assert_that(get.checksum(save.filename) == save.filename.checksum)
