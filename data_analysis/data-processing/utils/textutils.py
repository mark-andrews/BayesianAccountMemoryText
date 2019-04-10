"""
Tools for the processing of texts, vocabulary, especially using the British
National Corpus.

"""
#=============================================================================
# Standard library imports
#=============================================================================
import os
import itertools
import cPickle as pickle

from bs4 import BeautifulSoup

#================================ End Imports ================================

vocabulary_directory='vocab'
stopwords_lists_filenames = ('FoxStoplist.txt', 'SmartStoplist.txt')
vocabulary_filenames = ('2of4brif.txt',)


class Corpus(object):

    @classmethod
    def get_corpus_filenames(cls, corpus_xmlfiles_rootdir):

        corpus = cls(corpus_xmlfiles_rootdir)
        return corpus.corpus_filenames


    def __init__(self, corpus_xmlfiles_rootdir):

        self.corpus_xmlfiles_rootdir = corpus_xmlfiles_rootdir


    @property
    def corpus_filenames(self):

        """
        Get the list of all BNC xml corpus files. 

        """

        corpus_xmlfiles = []

        for root, dirs, filenames in os.walk(self.corpus_xmlfiles_rootdir):
            for filename in filenames:
                basename, extension = os.path.splitext(filename)
                if extension == '.xml':
                    corpus_xmlfiles.append(os.path.join(root, filename))


        return corpus_xmlfiles


    def _get_written_or_spoken_corpus_filenames(self, signature):


        return [filename for filename in self.corpus_filenames 
                if signature in open(filename).read()]


    def get_written_corpus_filenames(self):

        """
        Return list of xml files that correspond to the written portion of the
        BNC.

        """

        return self._get_written_or_spoken_corpus_filenames('<wtext')


    def get_spoken_corpus_filenames(self):

        """
        Return list of xml files that correspond to the spoken portion of the
        BNC.

        """

        return self._get_written_or_spoken_corpus_filenames('<stext')


def get_words(xmlelement):

    """
    Get all words, lower-cased, from the word tags in the BNC xmlelement.

    """

    return [word_tag.text.strip().lower() 
            for word_tag in xmlelement.find_all('w')]


def get_corpus_file_soup(corpus_filename):

    """
    For a given corpus xml filename, return its BeautifulSoup soup.

    """

    return BeautifulSoup(open(corpus_filename), 'xml')


def get_all_paragraphs(xmlfilename):

    """
    Return all paragraphs, indicating xml filename and div1 count and paragraph
    count in the div1.

    """

    soup = get_corpus_file_soup(xmlfilename)

    results = []
    for i, div in enumerate(soup.find_all('div', {'level': '1'})):

        all_paragraphs_in_div1 = div.find_all('p')

        for j, paragraph in enumerate(all_paragraphs_in_div1):

            words = get_words(paragraph)

            paragraph_details = dict(corpus_filename = xmlfilename,
                                     div1_index = i,
                                     paragraph_index = j,
                                     paragraph_count = len(all_paragraphs_in_div1),
                                     words = words,
                                     word_count = len(words))

            results.append(paragraph_details)

    return results


def get_all_paragraphs_parallel(view, xmlfilenames):

    _all_paragraphs = view.map(get_all_paragraphs, 
                               xmlfilenames)

    return list(itertools.chain(*_all_paragraphs))


def _read_wordlist(filename):
    
    """
    Read in file contents, return all newline delimited strings
    unless the line starts with "#".
    
    """

    filepath = os.path.join(vocabulary_directory, filename)
    
    file_contents = open(filepath).read().strip().split('\n')
    return [word for word in file_contents if word[0] != '#']


def _get_wordlists_from_filenames(words_list_filenames):

    """
    Read in all words lists. Create their set union.
    Return as new list.

    """

    words_sets = map(lambda arg: set(_read_wordlist(arg)), 
                     words_list_filenames)

    return list(set.union(*words_sets))


def get_stopwords_list():

    """
    Read in all stop words lists. Create their set union.
    Return as new list.

    """

    return _get_wordlists_from_filenames(stopwords_lists_filenames)


def get_brief_vocabulary():

    """
    Read in all stop words lists. Create their set union.
    Return as new list.

    """

    return _get_wordlists_from_filenames(vocabulary_filenames)


class Vocab(object):

    def __init__(self, vocab=None, stopwords=None):

        """
        Stopwords and vocab should be lists.

        """

        if stopwords is None:
            _stopwords = set(get_stopwords_list())
        else:
            _stopwords = set(stopwords)

        if vocab is None:
            _vocab = set(get_brief_vocabulary())
        else:
            _vocab = set(vocab)

        vocab = sorted(_vocab.difference(_stopwords))

        word2index = {word:i for i, word in enumerate(vocab)}
        index2word = {i:word for i, word in enumerate(vocab)}

        for key in word2index:
            assert index2word[word2index[key]] == key

        for key in index2word:
            assert word2index[index2word[key]] == key

        self.vocab = vocab
        self.word2index = word2index
        self.index2word = index2word


def dump(data, filename, protocol=2):

    """
    For pickle writing large lists to avoid memory errors.
    From http://stackoverflow.com/a/20725705/1009979

    """

    with open(filename, "wb") as f:
        pickle.dump(len(data), f, protocol=protocol)
        for value in data:
            pickle.dump(value, f, protocol=protocol)


def load(filename):

    """
    For pickle loading large pickled lists.
    From http://stackoverflow.com/a/20725705/1009979

    """

    data = []
    with open(filename, "rb") as f:
        N = pickle.load(f)
        for _ in xrange(N):
            data.append(pickle.load(f))

    return data


def get_corpus_vocabulary(paragraphs):

    """

    The vocabulary is defined as the intersection of the set of lower cased
    words in the BNC and the words in the vocab file minus the stopwords.

    Return the vocabularly with its frequencies.

    """


    stopwords = dict.fromkeys(get_stopwords_list())
    acceptable_word_list = get_brief_vocabulary()

    word_counter = dict.fromkeys(acceptable_word_list, 0)

    for paragraph_details in paragraphs:
        for word in paragraph_details['words']:
            try:
                word_counter[word] += 1
            except KeyError:
                pass
            

    # Clear out the stop words
    for word in word_counter.keys():
        if word in stopwords:
            del word_counter[word]

    return word_counter
