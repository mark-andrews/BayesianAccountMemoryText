
cache.directory <- '../_cache'
recognition.data.filename <- 'experiment_brisbane_recognition_memory_tests.csv'

Df <- read.csv(file.path(cache.directory, recognition.data.filename))

variables.to.convert <- c('expected', 'response', 'correct')
Df[variables.to.convert] <- lapply(Df[variables.to.convert],
                                   function(variable) {levels(variable) <- c(F, T); variable})
Df['X'] <- NULL

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
             L = L,
             item = item,
             subject = subject,
             text = text)

data <- list('pp'=data,
             'cc'=data,
             'aa'=data)

re.scale <- function(x) as.numeric(scale(x))

data[['pp']][['psi']] <- re.scale(log(Df$posterior.predictions))
data[['cc']][['psi']] <- re.scale(log(Df$cooccurrence.predictions))
data[['aa']][['psi']] <- re.scale(log(Df$association.predictions))
