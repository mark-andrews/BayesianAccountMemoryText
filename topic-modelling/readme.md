
The probabilistic topic modelling used in this project was carried out using a custom made Python and Fortran toolbox 
named [`gustavproject`](https://github.com/mark-andrews/gustavproject).


# Setup 

To use `gustavproject` for topic modelling, first set up a Python 2.7 virtual
environment and pip install some required software.

```{.bash}
TMPDIR=/tmp
VENVDIR=$TMPDIR/topic_modelling_venv
GIT_TMPDIR=$TMPDIR/gustavproject
VENVCMD=virtualenv2

# Set up virtual environment and install everything in requirements.txt
if [[ -d $VENVDIR ]]; then
	rm -rf $VENVDIR
fi

$VENVCMD $VENVDIR
source $VENVDIR/bin/activate
pip install -r requirements.txt
```

Now we install the `gustavproject`.

```{.bash}
if [[ -d $GIT_TMPDIR ]]; then
	rm -rf $GIT_TMPDIR
fi

# clone it
git clone https://github.com/mark-andrews/gustavproject.git $GIT_TMPDIR

# compile it
CWD=`pwd`
cd $GIT_TMPDIR
git checkout 8869808  # this was latest version as of April 11, 2019
make all

# test it
python setup.py test

# pip install it 
pip install -e .

# return to original working directory
cd $CWD
```

All of the above shell commands are also available in `setup.sh`.

# Gibbs sampling

The final version of the topic model used in the this project is
`hdptm_201117172636_2290`. The posterior distribution was sampled, using a
Gibbs sampler, for 20000 iterations. The code used to sample from this model is
similar to the following code, which continues the sampler for 100 iterations,
running on 16 separate processors, and sampling hyperparameters.

```{.bash}
gustave model hdptm_201117172636_2290 update --parallel 16 --iterations=100 --hyperparameters
```
