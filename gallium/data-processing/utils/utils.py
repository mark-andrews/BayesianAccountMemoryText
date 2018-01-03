"""
Some general utils.

"""

#=============================================================================
# Standard library imports
#=============================================================================
import string 
import re
import os
import errno
import hashlib
import urllib
import bz2
import numpy
import cPickle as pickle

#================================ End Imports ================================

def deletechars(s, exclude_chars):
    ''' Fast deletion of characters from string.
    It uses a dummy translation table, and so no mapping is applied, and we
    just delete the exclude_chars characters.
    '''
    phony_translate_table = string.maketrans("","")
    return s.translate(phony_translate_table, exclude_chars)


def deletepunctuation(s):
    ''' Fast deletion of punctuation from string'''
    return deletechars(s,string.punctuation)


def tokenize(text, foldcase=True):
    ''' 
    A very cheap and easy tokenization.
    First, remove "'s". For example, "dog's" becomes "dog".
    Second, zap utf-8 chars.
    Then, remove all punctuation and, by default, fold upper and lower case words
    and then split by whitespace.
    '''

    text = re.sub(r'\'s','', text)
    s = ''.join([s for s in text if s in string.printable])

    s = str(s) # Got to convert it to str.
    s = deletepunctuation(s)

    if foldcase:
        s = s.lower()
    return s.split()


def mkdir_p(path):
    '''
    Make a directory, making parents if necessary.
    Taken verbatim from
    http://stackoverflow.com/a/600612
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def checksum(argument, algorithm='sha256'):
    '''
    Returns the hash checksum of `argument'.
    If `argument' is a name of a file, then perform the checksum on the file.
    Otherwise, the checksum is of the string `argument'.
    By default, it will be the sha1 checksum (and so equivalent to linux's
    sha1sum). Alternatively, the algorithm could be md5 (equivalent to linux's
    md5sum), or else sha224, sha256, sha384, sha512.
    '''

    h = hashlib.new(algorithm)

    if os.path.exists(argument) and os.path.isfile(argument):
        argument = open(argument,'rb').read()

    h.update(argument)

    return h.hexdigest()


def verify_cache_files(filenames, cache='_cache', verbose=False):

    '''
    Check if files exist in the cache directory. Check their integrity.

    '''

    mkdir_p(cache) # Make dir unless it already exists
    
    for filename, file_checksum in filenames:

        local_filepath = os.path.join(cache, filename)

        try:
            assert os.path.exists(local_filepath)
        except AssertionError:
            print('File %s does not exist in cache %s.' % (filename,
                                                           cache)
                  )
            raise

        try:
            if verbose:
                print('Check integrity of file %s.' % filename)

            assert checksum(local_filepath) == file_checksum 

            if verbose:
                print('Integrity check complete.')

        except AssertionError:
            raise AssertionError('File %s failed integrity check.' % filename)


# TODO (Tue 30 May 2017 20:30:06 BST): Obselete now?
def curl(root, filenames, cache='_cache', verbose=False):

    '''
    Download necessary files for the topic model, unless they are already there.
    Check file integrity then.

    '''

    mkdir_p(cache) # Make dir unless it already exists
    
    for filename, file_checksum in filenames:

        local_filepath = os.path.join(cache, filename)

        try:
            assert os.path.exists(local_filepath)
        except AssertionError:

            url = os.path.join(root, filename)

            if verbose:
                print('Downloading %s' % url)

            f = urllib.URLopener()
            f.retrieve(url, local_filepath)     

            if verbose:
                print('Download complete.')

        try:
            if verbose:
                print('Check integrity of file %s.' % filename)

            assert checksum(local_filepath) == file_checksum 

            if verbose:
                print('Integrity check complete.')

        except AssertionError:
            raise AssertionError('File %s failed integrity check.' % filename)

def _load_bz2(filename, cache, func, verbose=False):

    '''
    Load a bz2 data file.

    '''

    local_filepath = os.path.join(cache, filename)

    basename, extension = os.path.splitext(local_filepath)

    if extension == '.bz2':

        if os.path.exists(basename):

            if verbose:
                print('%s already exists. Skipping uncompression.' % 
                      os.path.basename(basename))

        else:

            if verbose:
                print('Uncompressing %s to %s' % (filename, basename))

            # Read bunzipped data to tmp_data
            with bz2.BZ2File(local_filepath) as f:
                tmp_data = f.read()

            # Write tmp_data to filename without bz2 extension
            with open(basename, 'wb') as f:
                f.write(tmp_data)

        return func(basename)

    else:
        raise Exception('Expecting a bz2 file.')

def loadcsv(filename, cache, verbose=False):

    '''
    Load a bz2-ed csv data file.

    '''

    return _load_bz2(filename=filename, 
                     cache=cache, 
                     func=lambda arg: open(arg).read().strip().split('\n'),
                     verbose=verbose)

def loadnpz(filename, cache, verbose=False):

    '''
    Load a bz2 numpy npz data file.

    '''

    return _load_bz2(filename=filename, 
                     cache=cache, 
                     func=numpy.load, 
                     verbose=verbose)


def bunzip(filename, cache, verbose=False):

    '''
    Unzip a bz2 data file that is in the cache. Store its
    unzipped file in the cache too. This will skip the 
    uncompression if the file exists already.

    '''
    
    
    return _load_bz2(filename=filename, 
                     cache=cache, 
                     func=lambda arg: None,
                     verbose=verbose)

def save_pkl(filename, **kwargs):
    with open(filename, 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
