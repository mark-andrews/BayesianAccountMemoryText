The shell scripts

	run_recall_model.sh
	run_recognition_model.sh

will run all the regression analyses of the recall and recognition data. They
produce a set of around 25 to 30 files, which are placed in the cache directory
(the location of which is specified in the main R scripts that are called from
the shell scripts).

Analyses of the recall data uses Jags, and the recognition analyses uses Stan/brms.

These analyses are REALLY computionally demanding. I used a 36 core Intel Xeon
Gold 6154, and they still take a total of around 5 days or more.

The scripts above can be run by using the following Make commands:

	make recall_memory_analyses
	make recognition_memory_analyses

Before everything is run, however, you must have certain csv files in your
cache (assuming the cache is ../cache). You can check if they are there, and
are the right files by doing:

	make dependencies_check
