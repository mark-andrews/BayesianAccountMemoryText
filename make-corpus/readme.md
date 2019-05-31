# Jupyter Python notebook to make a topic modelling corpus

The notebook creates a corpus of short documents from the BNC for use with
probabilistic topic models, particularly using the `gustavproject` topic
modelling toolbox.

It requires a Python package called `bnctools` and this will be installed, as
will all other requirements, if you do the following.

```{.bash}
TMPDIR=/tmp
VENVDIR=$TMPDIR/make_corpus_venv
VENVCMD=virtualenv2

# Set up virtual environment and install everything in requirements.txt
if [[ -d $VENVDIR ]]; then
	rm -rf $VENVDIR
fi

$VENVCMD $VENVDIR
source $VENVDIR/bin/activate
pip install -r requirements.txt

```

The data files in `cache` are required. See the `readme.md` inside `cache` to see how to get these.
