# Code, data files, etc. for a research project on a Bayesian model of memory for text

I've made every effort to make this entire research project fully reproducible. Everything from the creation of the stimuli for the behavioural experiment, the sample size determination of the experiment, the creation of the BNC corpus for the topic modelling, the code the topic modelling, the computational modelling of the alternative (alternative to topic modelling) computational models, the post-processing of the computational models's data, the pre-processing of all the behavioural and other data prior to the final regression analyses, the final Bayesian regression analyses, and the final to-be-peer-reviewed and to-hopefully-be-published with all the tables, figures etc are all reproducible using the scripts and data in this repository.

The sub-directories here are:

* data-analysis: R, Jags, Stan/brms code for Bayesian multilevel logistic regression.
* data-processing: Python code for pre-processing experimental data and preparing computational models' predictions for use in the regression analyses.
* make-corpus: Mostly Python code for the creation of the BNC corpus to be used in the probabilistic topic modelling.
* sample-size-determination: Mostly Python (calling R code) code for determining sample size for the behavioural experiment.
* stimuli-generation: Mostly Python code for generating the to-be-memorized texts and the memory word lists in the behavioural experiments.
* topic-modelling: Probabilistic topic modelling using the Gustav topic modelling toolbox.
* manuscript: The RMarkdown and LaTeX manuscript 
