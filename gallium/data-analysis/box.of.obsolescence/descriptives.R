Df <- read.csv('_cache/recognition.csv')

variables.to.convert <- c('expected', 'response', 'correct')
Df[variables.to.convert] <- lapply(Df[variables.to.convert],
                                   function(variable) {levels(variable) <- c(F, T); variable})
Df['X'] <- NULL

Df$correct <- as.logical(Df$correct)
Df$response <- as.logical(Df$response)

# Overall average correct 
aggregate(correct ~ 1, data=Df, mean)

# Distributions of accuracy rates by subject
quantile(aggregate(correct ~ subject, data=Df, mean)$correct, prob=c(0.025, 0.5, 0.975))

# Distributions of accuracy rates by subject
quantile(aggregate(correct ~ text, data=Df, mean)$correct, prob=c(0.025, 0.5, 0.975))

# Overall false positive, true negative rates
aggregate(response ~ expected, data=Df, mean)

# Recall results
# ===============


Df <- read.csv('_cache/recall.csv')
variables.to.convert <- c('accuracy')
Df[variables.to.convert] <- lapply(Df[variables.to.convert],
                                   function(variable) {levels(variable) <- c(F, T); variable})
Df['X'] <- NULL
Df$accuracy <- as.logical(Df$accuracy)

# Overall accuracy of recall
aggregate(accuracy ~ 1, data=Df, mean)

# Distribution of number of recalled words per test
quantile(aggregate(accuracy ~ subject + text, data=Df, FUN=length)$accuracy, prob=c(0.025, 0.5, 0.975))

