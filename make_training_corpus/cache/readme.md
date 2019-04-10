The cache directory contains raw-data or built data files.

On initial `git clone`, this directory will have files, but they will be just
place-holders for the real files. If you view those files, e.g. with cat,
you'll see they just contain checksum hashes of the files for which they are
placeholders. 

To get the files, do the following:

* `git fat init`
* `git fat pull` 

Having done that, you'll need to unpack or uncompress some of them:

* `mkdir bnc`
* `unzip 2554.zip -d bnc`
* `bunzip2 -k bnc_paragraphs.pkl.bz2`
* `unzip vocabulary_lists.zip`

A shell script `setup.sh` with these commands is also available.
