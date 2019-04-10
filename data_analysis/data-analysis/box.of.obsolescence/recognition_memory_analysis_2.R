library(lme4)
library(xtable)

Df <- read.csv('_cache/recognition.csv')

variables.to.convert <- c('expected', 'response', 'correct')
Df[variables.to.convert] <- lapply(Df[variables.to.convert],
                                   function(variable) {levels(variable) <- c(F, T); variable})
Df['X'] <- NULL


I <- !is.na(Df$posterior.predictions) & Df$cooccurrence.predictions > 0
Df.1 <- Df[I,]


M.null <- glmer(response ~ expected  + (1|text) + (1|subject),
             data=Df.1,
             family=binomial)

M.pp <- glmer(response ~  expected + log(posterior.predictions)  + (1|text) + (1|subject),
           data=Df.1,
           family=binomial)

M.cc <- glmer(response ~  expected + log(cooccurrence.predictions)  + (1|text) + (1|subject),
              data=Df.1,
              family=binomial)

M.aa <- glmer(response ~  expected + log(association.predictions)  + (1|text) + (1|subject),
              data=Df.1,
              family=binomial)


# Model fit results 
models = list(TopicModel=M.pp,
              CooccurrenceModel=M.cc,
              AssociationModel=M.aa,
              NullModel=M.null)

results <- rbind(sapply(models, BIC), 
                 sapply(models, AIC), 
                 sapply(models, function(arg) -2*logLik(arg)[1]))

rownames(results) <- c('BIC', 'AIC', 'Deviance')

print(xtable(results))
