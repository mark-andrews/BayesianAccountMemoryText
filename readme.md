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


# Downloading

There are two options for downloading all this code and data. Because the size
of the entire set of files is large (around 10GB), downloading it is not
completely trivial. The first option is to use `git` and `git fat`. The other
is to download a set of partial tar files from FigShare and then `cat` these together. Both of these
options are probably not part of everyone's everyday workflow and neither is as
simple as clicking a link.

## Git and git fat

First, you clone the repository as per normal.
```
git clone https://github.com/mark-andrews/BayesianAccountMemoryText.git
```

The next step is to get the "fat" data files, and for this, you first need to
install and set up [https://github.com/jedbrown/git-fat](git-fat). Personally,
I've found `git-fat` very easy to install and use, but I've only ever used it
on Linux, and so can't say how easy it would be to install and use on Macs and
Windows. I'd guess, it would work quite similarly, or even identically, to
Linux on Macs, but I believe it's a bit of a pain on Windows due to its
dependencies, and it might require [http://www.cygwin.com](cygwin).

Assuming that you've installed `git fat`, then do 
```
cd BayesianAccountMemoryText
git fat init
git fat pull http
```

The `git fat pull http` step will download around 20GB of files from a small
private server, and so could take 30 minutes or more or less depending on your
internet bandwidth.

## Download from Figshare

A `.tar` archive of all the files in Git repository is around 10GB in size.
Figshare, while generously allowing unlimited public storage space, requires
files to be less than 5GB. To use Figshare, I've `split` the `.tar` archive
into four parts using the Unix/Linux `split` command. These can be downloaded
from Figshare in one zip arcive, but then must be rejoined into a single `.tar`
archive, and extracted. This is procedure is a bit convoluted, but I did not
see an alternative.

Go to this Figshare page [https://doi.org/10.6084/m9.figshare.8246900](https://doi.org/10.6084/m9.figshare.8246900), and then click *Download all*. This will download a zip archive. Unzip this, and that should give you the following 4 large files.

	* BayesianAccountMemoryText_tar_part_aa
	* BayesianAccountMemoryText_tar_part_ab
	* BayesianAccountMemoryText_tar_part_ac
	* BayesianAccountMemoryText_tar_part_ad

These are `split`ed parts of a large `.tar` archive. On a Unix/Linux system (including Mac OSX and
presumably with Windows if you use Cygwin or other options), they can be rejoined as follows.
```
cat BayesianAccountMemoryText_tar_part_* > BayesianAccountMemoryText.tar
```

Now, `BayesianAccountMemoryText.tar` is the a `tar` archive and can be extracted as follows.
```
mkdir BayesianAccountMemoryText
tar xf BayesianAccountMemoryText.tar -C BayesianAccountMemoryText
```

# Related resources

* The probabilistic topic modelling was done using a Python/Fortran toolbox `Gustav`.

> Andrews, M. (2019). Gustav: A probabilistic topic modelling toolbox. Zenodo. [doi:10.5281/zenodo.51824](https://doi.org/10.5281/zenodo.51824).

* The Gibbs sampler for the main topic model used is described in the following manuscript.

> Andrews, M. (2019). A Gibbs Sampler for a Hierarchical Dirichlet Process Mixture Model. PsyArXiv. [doi:10.31234/osf.io/ebzt8](https://doi.org/10.31234/osf.io/ebzt8).

* The behavioural experiments were done using a Python/Javascript web app framework named `Wilhelm`.

> Andrews, M. (2019). Wilhelm: A web application framework for online behavioural experiments. Zenodo. [doi:10.5281/zenodo.2647481](https://doi.org/10.5281/zenodo.2647481).
