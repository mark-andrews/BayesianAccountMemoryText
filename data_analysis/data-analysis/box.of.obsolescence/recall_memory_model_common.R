Df <- read.csv('_cache/recall.csv')

Df['X'] <- NULL

I <- !is.na(Df$posterior.predictions) & Df$cooccurrence.predictions > 0
Df <- Df[I,]

N <- dim(Df)[1]
J <- length(levels(Df$subject))
L <- length(levels(Df$response))
K <- max(Df$text)
subject <- as.numeric(Df$subject)
item <- as.numeric(Df$response)
text <- Df$text

data <- list(psi = log(Df$cooccurrence.predictions),
             N = N,
             J = J,
             K = K,
             L = L,
             item = item,
             subject = subject,
             text = text)

data.cc <- data
data.pp <- data
data.aa <- data
data.pp$psi <- log(Df$posterior.predictions)
data.cc$psi <- log(Df$cooccurrence.predictions)
data.aa$psi <- log(Df$association.predictions)