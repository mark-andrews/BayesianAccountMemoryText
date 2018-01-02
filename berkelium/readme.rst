Stimuli generation for experiment *Brisbane*
=============================================

* The IPython/Jupyter notebook `stimuli-generation.ipynb` describes how I selected the text and word-lists to be used in a memory experiment. 

* The `requirements.txt` provides the list of required Python packages. Install these using `pip install -r requirements.txt`, ideally in a virtual environment.

* All the necessary custom code for the notebook is in `utils.py`.

* The notebook and code assumes that you have a copy of British National Corpus (BNC) as a zip archive, which is available from http://ota.ox.ac.uk/desc/2554. See the notebook for more information.

* The bash script `prepare_bnc.sh` can be used to unzip the archive and check its contents.

* Inside the `vocab` directory are text files of stop words and a text file of common English words. See notebook for further information.

* Although not strictly necessary to run to notebook, I was using some IPython notebook extensions, particularly `Python Markdown`. These extensions need to be manually installed from https://github.com/ipython-contrib/IPython-notebook-extensions.git.

