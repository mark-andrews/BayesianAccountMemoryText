# Jupyter notebooks and utils for data analysis of experiment Brisbane

Experiment Brisbane is a recall and recognition memory task applied to
memory of short texts. It was carried out online and it available here.

-   <https://www.cognitionexperiments.org/brisbane>

Our aim to analysis the recall and recognition memory results especially
with respect to how well they are predicted by three different
theoretical models.

## Installation

* Clone the repository and change directories so that it is your working directory:
```bash
git clone https://github.com/lawsofthought/gallium.git
cd gallium
```

* Create a virtual environment. It can be located anywhere, but I'll just assume it is in your home directory. After it is created, activate it.
```bash
virtualenv ~/gallium-virtual-env
source ~/gallium-virtual-env/bin/activate
```

* Pip install all your required Python packages. These are in the text file `requirements.txt`.
```bash
pip install -r requirements.txt
```

* Now deal with the annoying issue that the setup of `gustavproject` did not compile the external Fortran modules. First, go to the source, make it, and popd back to where you were.
```bash
pushd ~/gallium-virtual-env/src/gustav
make
popd
```

* Get all the "fat" files
```bash
git fat init
git fat pull http
```
