library(rjags)


Df <- read.csv('_cache/recognition.csv')

variables.to.convert <- c('expected', 'response', 'correct')
Df[variables.to.convert] <- lapply(Df[variables.to.convert],
                                   function(variable) {levels(variable) <- c(F, T); variable})
Df['X'] <- NULL

logit <- function(p) log(p/(1-p))
ilogit <- function(x) 1/(1+exp(-x))


I <- !is.na(Df$posterior.predictions) & Df$cooccurrence.predictions > 0
Df <- Df[I,]

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
             psi = log(Df$cooccurrence.predictions),
             N = N,
             J = J,
             K = K,
             #L = L,
             subject = subject,
             #item = item,
             text = text)
               
data.cc <- data
data.pp <- data
data.pp$psi <- log(Df$posterior.predictions)
data.cc$psi <- log(Df$cooccurrence.predictions)

M.pp <- jags.model('recognition_memory_model_2.jags', 
                data=data.pp,
                n.chains = 3)

M.cc <- jags.model('recognition_memory_model_2.jags', 
                data=data.cc,
                n.chains = 3)

update(M.pp, 100000)
update(M.cc, 100000)

S.pp <- coda.samples(M.pp, variable.names = c('p'), 10000)
S.cc <- coda.samples(M.cc, variable.names = c('p'), 10000)


S <- coda.samples(M, variable.names = c('alpha.text.sigma'), 10000)
S.s <- coda.samples(M, variable.names = c('alpha.subject.sigma'), 10000)
S.i <- coda.samples(M, variable.names = c('alpha.item.sigma'), 10000)

S.a <- coda.samples(M, variable.names = c('alpha.0', 'beta'), 10000)

S <- coda.samples(M, variable.names = c('b'), 1000)

p^data$y * (1-p)^(1-data$y)
