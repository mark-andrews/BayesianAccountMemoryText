library(assertthat)

get.word2index <- function(Q, rows=T){
  
  if (rows) 
    {the.names <- rownames(Q)}
  else 
    {the.names <- colnames(Q)}
  
  word2index <- seq(length(the.names))
  names(word2index) <- the.names
  
  return(word2index)
  
}

posterior.predictions.recall <- as.matrix(read.csv('_cache/posterior_predictions_of_recalled_words.csv', check.names = F))
cooccurrence.predictions.recall <- as.matrix(read.csv('_cache/cooccurrences_predictions_of_recalled_words.csv', check.names = F))

recalledwords.word2index <- get.word2index(posterior.predictions.recall, row=F)
texts.word2index <- get.word2index(posterior.predictions.recall, row=T)

.recalledwords.word2index <- get.word2index(cooccurrence.predictions.recall, row=F)
.texts.word2index <- get.word2index(cooccurrence.predictions.recall, row=T)

assert_that(are_equal(recalledwords.word2index, .recalledwords.word2index))
assert_that(are_equal(texts.word2index, texts.word2index))

rm(.recalledwords.word2index, .texts.word2index)

Df <- read.csv('_cache/recall.csv')

Df['X'] <- NULL

I <- !is.na(Df$posterior.predictions) & Df$cooccurrence.predictions > 0
Df <- Df[I,]

# Test if the prediction probabilities in Df match those of the matrices
for (i in seq(dim(Df)[1])){
  
  text.id <- Df[i, 'text']
  word <- Df[i, 'word']
  pp <- Df[i, 'posterior.predictions']
  cc <- Df[i, 'cooccurrence.predictions']
  
  assert_that(are_equal(posterior.predictions.recall[text.id, word], pp))
  assert_that(are_equal(cooccurrence.predictions.recall[text.id, word], cc))

}

Df <- Df[c('subject', 'text', 'word')]

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

