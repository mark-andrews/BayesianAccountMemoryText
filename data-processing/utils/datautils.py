from __future__ import absolute_import, division

from collections import defaultdict
from scipy import sparse
import string
import re
import numpy
import os
import bz2


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


class Vocab(object):

    '''
    A simple vocabulary class
    '''

    def __init__(self, vocab):

        """
        vocab should be a list
        
        """

        word2index = {word:i for i, word in enumerate(vocab)}
        index2word = {i:word for i, word in enumerate(vocab)}

        for key in word2index:
            assert index2word[word2index[key]] == key

        for key in index2word:
            assert word2index[index2word[key]] == key

        self.vocab = vocab
        self.word2index = word2index
        self.index2word = index2word


class Cooccurrences(object):

    """
    Class for getting co-occurrence joint and conditional probabilities.
    """

    def __init__(self, filename, cache, vocab):

        self.filename = filename
        self.cache = cache
        self.vocabulary = vocab
        self.initialized = False

        self.texts = self.load_text(self.filename, self.cache)
        self.calculate_word_counts()
        self.calculate_cooccurrences()

    def load_text(self, filename, cache, verbose=False):

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

            return open(basename).read().strip().split('\n')

        else:
            raise Exception('Expecting a bz2 file.')


    def calculate_word_counts(self):

        self.counts = []
        for text in self.texts:
            self.counts.append(self.get_word_counts_per_text(text))

    def get_word_counts_per_text(self, text):

        counts = defaultdict(int)
        for word in text.split('|'):
            if word in self.vocabulary.word2index:
                counts[word] += 1
        return counts

    def calculate_cooccurrences(self):

        '''
        Return sparse VxV matrix giving coccurrence counts (frequencies).
        '''

        rows = []
        cols = []
        values = []

        for j, counts_text_j in enumerate(self.counts):
            for key, value in counts_text_j.iteritems():
                try:
                    cols.append(self.vocabulary.word2index[key])
                    rows.append(j)
                    values.append(value)
                except KeyError:
                    pass

        self.count_matrix =  sparse.csr_matrix((values, (rows, cols)),
                                               shape=(len(self.counts),
                                                      len(self.vocabulary.word2index)))

        C = self.count_matrix.T.dot(self.count_matrix)
        C.setdiag((self.count_matrix.power(2) - self.count_matrix).multiply(0.5).sum(0).A.flatten())

        mj = self.count_matrix.sum(1)
        assert sparse.triu(C).sum() == ((mj.A**2 - mj.A)/2).sum()

        self.C = C

    def get_cooccurrence_profile(self, word):

        '''
        For a given word, return its coccurrence counts for every other word.

        '''

        j = self.vocabulary.word2index[word]

        return self.C[j].A[0] # The zero index returns a one-dim array
    

    def get_sparse_matrix_ijv(self):

        C = sparse.coo_matrix(self.C)

        return (C.row, C.col, C.data) + C.shape
