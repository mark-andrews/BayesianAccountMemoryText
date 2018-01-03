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
             subject = subject,
             text = text)
               
data.cc <- data
data.pp <- data
data.pp$psi <- log(Df$posterior.predictions)
data.cc$psi <- log(Df$cooccurrence.predictions)

M.pp <- jags.model('recognition_memory_model_3.jags', 
                data=data.pp,
                n.chains = 3)

M.cc <- jags.model('recognition_memory_model_3.jags', 
                data=data.cc,
                n.chains = 3)

update(M.pp, 1000)
update(M.cc, 1000)

S.pp <- coda.samples(M.pp, variable.names = c('p'), 1000)
S.cc <- coda.samples(M.cc, variable.names = c('p'), 1000)

y <- data$y

f <- function(p) sum(y*log(p) + (1-y)*log(1-p))

quantile(apply(S.cc[[1]], 1, f))
quantile(apply(S.cc[[2]], 1, f))
quantile(apply(S.cc[[3]], 1, f))

quantile(apply(S.pp[[1]], 1, f))
quantile(apply(S.pp[[2]], 1, f))
quantile(apply(S.pp[[3]], 1, f))

S.ww <- coda.samples(M.pp, variable.names = c('ww'), 1000)
S.vv <- coda.samples(M.pp, variable.names = c('vv'), 1000)
