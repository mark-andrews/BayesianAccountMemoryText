Df <- read.csv('_cache/recall.csv')

Df['X'] <- NULL

I <- !is.na(Df$posterior.predictions) & Df$cooccurrence.predictions > 0
Df <- Df[I,]

Df <- Df[c('subject', 'text', 'word', 'posterior.predictions')]

N <- dim(Df)[1]
subject <- as.numeric(Df$subject)
words <- as.vector(sapply(Df$word, function(w) recalledwords.word2index[w]))
text <- Df$text
J <- length(levels(Df$subject))
L <- dim(posterior.predictions.recall)[2]
K <- dim(posterior.predictions.recall)[1]

data <- list(subject = subject,
             text = text,
             y = words,
             N = N,
             J = J,
             L = L,
             K = K,
             psi = log(posterior.predictions.recall))

