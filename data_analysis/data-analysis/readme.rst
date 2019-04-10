Data analysis of experiment *Brisbane*
======================================

In experiment *Brisbane*, each participant performed up to three separate
memory tests.  In each test, the participant read a text and then had their
memory tested either by a recognition or by a recall test.  Here is the data
analysis of both the recognition and recall data. Ultimately, our aim is to
investigate the extent to which a topic model, an association model based on
word cooccurrence statistics, and an association model based on word
association norms, can predict the recall and recognition results.

Markdown notebook
=================

The notebook `data-analysis.Rmd` is the main source of all the results of the
analyses. However, some of these analyses, particularly the expensive MCMC
simulations, were run in batch modes using bash and R scripts, and their
results were cached (see below). These cached results were read in by the
`data-analysis.Rmd`.

Bayesian data analysis of recognition memory 
============================================

The recognition data was modelled using multilevel logistic regression models.
These were implemented in Jags, which allowed us to draw samples from the
posterior distribution over all unknown variables with a Gibbs sampler. Six
different models were implemented for the purposes of comparison. These can be
run using a bash script that calls a command line Rscript that calls parallel R
processes, each running separate samplers.

.. code:: bash

   bash run_mcmc_recognition_model.sh n

where `n` is a number between 1 and 6. This will re-run the models, using the
number of iterations, random seeds, etc that I used and then save their results
to the `_cache` directory
