TMPDIR=/tmp
VENVDIR=$TMPDIR/apricot_data_processing_venv
GIT_TMPDIR=$TMPDIR/apricot_data_processing_gustavproject
VENVCMD=virtualenv2

# Set up virtual environment and install everything in requirements.txt
if [[ -d $VENVDIR ]]; then
	rm -rf $VENVDIR
fi

$VENVCMD $VENVDIR
source $VENVDIR/bin/activate
pip install -r requirements.txt

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

ln -fs "$(realpath ../make-corpus/cache/Brismo.cfg)" cache
ln -fs "$(realpath ../topic-modelling/data/corpora/bnc_texts_78639361_183975_251_499.npz)" cache
ln -fs "$(realpath ../data-analysis/cache/brisbane_06b643a_recall_results.pkl)" cache
for f in ../topic-modelling/data/samples/hdptm_201117172636_2290_state_*.npz; 
	do ln -fs `realpath $f` cache;
done

