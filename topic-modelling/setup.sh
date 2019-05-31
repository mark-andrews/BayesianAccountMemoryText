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

read -p "Do you want to run the sampler? (Enter 'Yes' if so)" run_sampler

if [[ $vpn == "Yes" ]]
   then 
      gustave model hdptm_201117172636_2290 update --parallel 16 --iterations=100 --hyperparameters
fi
