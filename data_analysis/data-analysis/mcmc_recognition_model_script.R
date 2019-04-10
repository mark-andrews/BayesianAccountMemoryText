# This script will be run from *nix command line using Rscript
# Presumably, it will run fine on windows too.

# Run e.g. as follows:
# Rscript mcmc_recognition_memory_analysis_script.R [MODEL VERSION] [NULL MODEL VERSION] [VARIABLES TO SAMPLE] [SEED] [ITERATIONS] [THIN]
# Rscript mcmc_recognition_memory_analysis_script.R v5 v5_v6 'b,beta,alpha.0' 91810 3 1
args = commandArgs(trailingOnly = TRUE)

# We assume that we are running a jags file such as
# jags/recognition_memory_model_v5.jags
# jags/recognition_memory_model_v1.jags

jags_model_version <- args[1] # Should be 'v3' 'v4' etc
jags_null_model_version <- args[2] # Should be 'v3_v4' 'v5_v6' etc

# List the variables you want to coda.sample like this 'a,b,c'
# I.e. quoted AND comma separated AND no-spaces
variables.to.sample.str <- args[3]
variables.to.sample <- unlist(strsplit(variables.to.sample.str, ','))

# 
master.seed <- as.integer(args[4])

# 
n.iterations <- as.integer(args[5])

# 
n.thin <- as.integer(args[6])

# =========================
source('utils.R')

##########################################################################################
get.data <- function(cache.directory, recognition.data.filename){

	Df <- get.recognition.data(file.path(cache.directory, recognition.data.filename))

	Df$subject <- factor(Df$subject)
	Df$stimulus <- factor(Df$stimulus)

	N <- dim(Df)[1]
	J <- length(levels(Df$subject))
	L <- length(levels(Df$stimulus))
	K <- max(Df$text)
	subject <- as.numeric(Df$subject)
	item <- as.numeric(Df$stimulus)
	text <- Df$text
	present <- as.logical(Df$expected) * 1.0

	data <- list(y = as.logical(Df$response) * 1.0,
	             present = present,
	             psi = NA, # We'll overwrite this below
	             N = N,
	             J = J,
	             K = K,
	             L = L,
	             item = item,
	             subject = subject,
	             text = text)

	data <- list('pp'=data,
	             'cc'=data,
	             'aa'=data,
	             'null'=data)

	data[['pp']][['psi']] <- re.scale(Df$posterior.predictions)
	data[['cc']][['psi']] <- re.scale(Df$cooccurrence.predictions)
	data[['aa']][['psi']] <- re.scale(Df$association.predictions)

	data[['null']]$psi <- NULL 

	data

}
####################################################################################

cache.directory <- '../cache'
recognition.data.filename <- 'experiment_brisbane_recognition_memory_tests.csv'
data <- get.data(cache.directory, recognition.data.filename)

job.names <- c('pp', 'cc', 'aa', 'null')
n.chains <- 3

set.seed(master.seed)

job.seeds <- get.randint(length(job.names))

jags.filename <- sprintf('jags/recognition_memory_model_%s.jags', jags_model_version)
jags.null.filename <- sprintf('jags/recognition_memory_model_%s_null.jags', jags_null_model_version)
jags.filename.checksum <- get.checksum(jags.filename)
jags.null.filename.checksum <- get.checksum(jags.null.filename)

job.filenames <- list('pp'=jags.filename,
                      'cc'=jags.filename,
                      'aa'=jags.filename,
                      'null'=jags.null.filename)

job.parameters <- make.job.parameters(data, job.names, job.filenames, job.seeds, n.chains)

the.cluster <- makeCluster(length(job.names), type='FORK')

clusterEvalQ(cl=the.cluster, library('rjags'));

clusterExport(cl=the.cluster, 
              c('make.jags.seeds', 
                'model.initialize', 
                'model.update', 
                'model.sample', 
                'model.dic', 
                'n.iterations',
                'jags.filename',
                'jags.null.filename',
                'n.thin'))

M <- parallel.initialize(the.cluster, 
                         job.parameters, 
                         n.iterations)

S <- parallel.sample(the.cluster, M, variables.to.sample, n.iterations, thin=n.thin)
D <- parallel.dic(the.cluster, M, n.iterations, thin=n.thin)

save.filename.template <- 'mcmc_recognition_model_%s_seed_%d_burn_%d_sample_%d_dic_%d_thin_%d.Rda'

list.to.save <- list(data=data,
                     M=M,
                     S=S,
                     D=D,
                     master.seed=master.seed,
                     n.iterations=n.iterations,
                     n.thin=n.thin,
                     variables.to.sample.str=variables.to.sample.str,
                     jags.filename=jags.filename,
                     jags.null.filename=jags.null.filename,
                     jags.filename.checksum=jags.filename.checksum,
                     jags.null.filename.checksum=jags.null.filename.checksum)
                     
saveRDS(list.to.save,
        file = file.path(cache.directory, 
                      sprintf(save.filename.template,
                              jags_model_version,
                              master.seed,
                              n.iterations,
                              n.iterations,
                              n.iterations,
                              n.thin)
                      )
     )