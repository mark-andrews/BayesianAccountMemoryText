import os
import configobj
import textwrap
import itertools
from collections import Counter, defaultdict, OrderedDict
from bs4 import BeautifulSoup
import cPickle as pickle
import numpy
from numpy import log, mean
from nltk.stem.snowball import SnowballStemmer


vocabulary_directory='vocab'
stopwords_lists_filenames = ('FoxStoplist.txt', 'SmartStoplist.txt')
vocabulary_filenames = ('2of4brif.txt',)
minimum_paragraph_size = 100

stem = SnowballStemmer('english').stem

words_only = lambda keywords: [word[0] for word in keywords]

clear_words = lambda args: [word for word in args[0] if word not in args[1]]

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
                                     paragraph_text = paragraph.text.strip(),
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


def get_inverse_document_frequency(paragraphs):

    """
    The inverse document frequency (idf) of word w is equal to 

    - log(n/N) 

    or equivalently

    log(N/n)

    where N is the number of "documents" in the corpus (defined as any
    paragraph greater than minimum_paragraph_size) and n is how many of the N
    documents the word w occurs in at least once.

    """

    f = Counter()
    N = 0 
    for paragraph in paragraphs:
        if paragraph['word_count'] > minimum_paragraph_size:
            N += 1
            f.update(set(paragraph['words']))

    return {w:log(N/n) for w,n in f.iteritems()}



def filter_paragraphs(paragraphs,
                      word_association,
                      minimum_length=200,
                      maximum_length=300,
                      density=(0.9, 0.5)):

    """
    Select paragraphs that are within the stated word length 
    and 
    have greater than the stated density of words in the vocabulary
    and
    have greater than the stated density of words in the word association
    stimulus words.

    """

    vocabulary = dict.fromkeys(get_brief_vocabulary())
    stimuli = dict.fromkeys(word_association.stimulus_words)

    paragraph_vocabulary_density\
        = lambda words: mean([word in vocabulary for word in words])

    stimuli_density\
        = lambda words: mean([word in stimuli for word in words])

    return [index for index, paragraph in enumerate(paragraphs)
            if (minimum_length < paragraph['word_count'] < maximum_length)
            and paragraph_vocabulary_density(paragraph['words']) > density[0]
            and stimuli_density(paragraph['words']) >= density[1]]


def get_word_stems(word_list):
    return set(map(stem, word_list))


def push_keyword(keyword_list, new_keyword):

    """
    Add new_keyword to list if it is not already there.

    """
    if new_keyword not in keyword_list:
        keyword_list.append(new_keyword)

    return keyword_list


class ParagraphSampler(object):

    def __init__(self,
                 paragraphs, 
                 paragraph_indices, 
                 idf, 
                 vocabulary, 
                 word_association,
                 number_of_paragraphs=50,
                 list_lengths=(20,10),
                 neighbour_paragraph_length=(150,250),
                 association_word_density_threshold=0.95,
                 random_seed=None):

        self.random = numpy.random.RandomState(random_seed)
        self.permutation = self.random.permutation

        self.paragraphs = paragraphs
        self.paragraph_indices = paragraph_indices
        self.idf = idf
        self.vocabulary = vocabulary

        self.word_association = word_association

        self.number_of_paragraphs = number_of_paragraphs

        self.wordlist_length = list_lengths[0]
        self.target_keywords_length = list_lengths[1]
        self.lures_keywords_length = list_lengths[1]

        self.neighbour_minimum_length = neighbour_paragraph_length[0]
        self.neighbour_maximum_length = neighbour_paragraph_length[1]

        self.shuffle_paragraph_indices()

        self.association_word_density_threshold\
            = association_word_density_threshold

    def shuffle_paragraph_indices(self):
        self.paragraph_indices = self.list_shuffle(self.paragraph_indices)


    def list_shuffle(self, the_list):
        N = len(the_list)
        return [the_list[i] for i in self.permutation(N)]


    def clean_lurewords(self, lurewords, main_text_words):

        """
        Return those lure words that are not in the main text nor are their
        morphological variants in the main text.

        """

        main_text_word_stems = get_word_stems(main_text_words)

        _lurewords = []
        for lureword in clear_words((lurewords, main_text_words)):

            if stem(lureword) not in main_text_word_stems:
                _lurewords.append(lureword)

        return _lurewords


    def unique_stem_words(self, words):

        """
        Do not have more than one morphological variant of any word in the list
        of target keywords.

        """

        word_stems = []
        _words = []
        for word in words:
            word_stem = stem(word)
            if word_stem not in word_stems:
                word_stems.append(word_stem)
                _words.append(word)

        return _words


    def check_paragraph_position_in_div(self, paragraph):
        """
        Is paragraph not the first nor the last in the div1?

        """

        return (1 < paragraph['paragraph_index'] 
                  < paragraph['paragraph_count']-1)


    def check_neighbour_paragraph_length(self,
                                         before_paragraph,
                                         after_paragraph):

        """
        Are both the paragraph before and the paragraph after the target
        paragraph within the length bounds?

        """

        return all([self.neighbour_minimum_length 
                    < _paragraph['wordcount'] 
                    < self.neighbour_maximum_length 
                    for _paragraph in (before_paragraph, after_paragraph)])


    def get_keywords(self, words):

        _keywords = get_keywords(words, self.vocabulary, self.idf)        
        return words_only(_keywords)

    def select_lurewords(self, 
                         before_paragraph_keywords, 
                         after_paragraph_keywords):

        """
        We want to use the top keywords in both the before|after keyword lists.
        We'll go through the first pair, the second pair, etc, and keep either
        or both if they have not already been pushed on.

        """

        lure_keywords = []
        for (before_kw, after_kw) in zip(before_paragraph_keywords,
                                         after_paragraph_keywords):

            for kw in (before_kw, after_kw):
                lure_keywords = push_keyword(lure_keywords, kw)
                lure_keywords = self.unique_stem_words(lure_keywords)
                if len(lure_keywords) == self.lures_keywords_length:
                    break

            if len(lure_keywords) == self.lures_keywords_length:
                break


        return lure_keywords

    @classmethod
    def select_paragraphs(cls,
                          paragraphs, 
                          paragraph_indices, 
                          idf, 
                          vocabulary, 
                          word_association,
                          number_of_paragraphs=50,
                          list_lengths=(20,10),
                          neighbour_paragraph_length=(150,250),
                          association_word_density_threshold=0.95,
                          random_seed=None):


        paragraph_sampler = cls(paragraphs=paragraphs, 
                                paragraph_indices=paragraph_indices, 
                                idf=idf, 
                                vocabulary=vocabulary, 
                                word_association=word_association,
                                number_of_paragraphs=number_of_paragraphs,
                                list_lengths=list_lengths,
                                neighbour_paragraph_length=neighbour_paragraph_length,
                                association_word_density_threshold=association_word_density_threshold,
                                random_seed=random_seed)


        return paragraph_sampler._select_paragraphs()


    def _select_paragraphs(self):
        
        P = []
        for index in self.paragraph_indices:

            paragraph = self.paragraphs[index]

            if self.check_paragraph_position_in_div(paragraph):

                before_paragraph, after_paragraph\
                    = get_neighbour_paragraphs(paragraph)

                if self.check_neighbour_paragraph_length(before_paragraph,
                                                         after_paragraph):

                    _paragraph_keywords\
                        = self.get_keywords(paragraph['words'])

                    paragraph_keywords\
                        = [word for word in _paragraph_keywords
                           if word in self.word_association.stimulus_words]

                    paragraph_keywords\
                        = self.unique_stem_words(paragraph_keywords)


                    before_paragraph_keywords\
                        = self.get_keywords(before_paragraph['words'])


                    after_paragraph_keywords\
                        = self.get_keywords(after_paragraph['words'])


                    before_paragraph_keywords\
                        = self.clean_lurewords(before_paragraph_keywords,
                                               paragraph['words'])

                    after_paragraph_keywords\
                        = self.clean_lurewords(after_paragraph_keywords,
                                               paragraph['words'])

                    lure_words\
                        = self.select_lurewords(before_paragraph_keywords,
                                                after_paragraph_keywords)

                    result\
                        = dict(text = paragraph['paragraph_text'],
                               corpus_filename = paragraph['corpus_filename'],
                               div1_index = paragraph['div1_index'],
                               all_paragraphs_index = index,
                               paragraph_index = paragraph['paragraph_index'],
                               keywords = paragraph_keywords,
                               wordlist = paragraph_keywords[:self.wordlist_length],
                               target_keywords = paragraph_keywords[:self.target_keywords_length],
                               lure_keywords = lure_words[:self.lures_keywords_length]
                               )

                    if all([len(result['lure_keywords']) == self.lures_keywords_length,
                            len(result['target_keywords']) == self.target_keywords_length,
                            len(result['wordlist']) == self.wordlist_length,
                            self.check_association_density(result['lure_keywords']),
                            self.check_association_density(result['target_keywords'])]):
 
                        P.append(result)

                    if len(P) == self.number_of_paragraphs:
                        break

        return P


    def check_association_density(self, words):

        if not words:
            return False
        else:
            return mean([word.encode('utf-8') in 
                         self.word_association.association_words 
                         for word in words]) >= self.association_word_density_threshold



#
#def choose_paragraph(paragraphs, 
#                     paragraph_indices, 
#                     idf, 
#                     vocabulary, 
#                     number_of_paragraphs=50,
#                     keyword_lengths=(20, 10),
#                     minimum_paragraph_length=150,
#                     maximum_paragraph_length=250,
#                     random_seed=None):
#
#    random = numpy.random.RandomState(random_seed)
#    permutation = random.permutation
#
#    wl_len, kw_len = keyword_lengths
#
#    def list_shuffle(the_list):
#        N = len(the_list)
#        return [the_list[i] for i in permutation(N)]
#
#    def stripout(keywords, text_words):
#        return [keyword[0] for keyword in keywords if keyword[0] not in text_words]
#
#    def push_keyword(keyword_list, keyword):
#        return list(set(keyword_list + [keyword]))
#
#
#    P = []
#    for index in list_shuffle(paragraph_indices):
#        print(len(P))
#
#        paragraph = paragraphs[index]
#
#        if 1 < paragraph['paragraph_index'] < paragraph['paragraph_count']-1:
#
#            before_paragraph, after_paragraph\
#                = get_neighbour_paragraphs(paragraph)
#
#            if all([minimum_paragraph_length 
#                    < _paragraph['wordcount'] 
#                    < maximum_paragraph_length 
#                    for _paragraph in (before_paragraph, after_paragraph)]):
#
#
#                paragraph_keywords\
#                    = get_keywords(paragraph['words'],
#                                   vocabulary,
#                                   idf)
#
#
#                before_paragraph_keywords\
#                    = get_keywords(before_paragraph['words'],
#                                   vocabulary,
#                                   idf)
#
#
#                after_paragraph_keywords\
#                    = get_keywords(after_paragraph['words'],
#                                   vocabulary,
#                                   idf)
#
#                before_paragraph_keywords\
#                    = stripout(before_paragraph_keywords,
#                               paragraph['words'])
#
#                after_paragraph_keywords\
#                    = stripout(after_paragraph_keywords,
#                               paragraph['words'])
#
#                all_target_keywords = [x[0] for x in paragraph_keywords]
#
#                target_keywords = all_target_keywords[:kw_len]
#                
#
#                lure_keywords = []
#                for (before_kw, after_kw) in zip(before_paragraph_keywords,
#                                                 after_paragraph_keywords):
#
#                    for kw in (before_kw, after_kw):
#                        lure_keywords = push_keyword(lure_keywords, kw)
#                        if len(lure_keywords) == kw_len:
#                            break
#
#                    if len(lure_keywords) == kw_len:
#                        break
#
#
#                assert set(all_target_keywords).intersection(lure_keywords) == set()
#                
#                p = dict(text=paragraph['paragraph_text'],
#                         wordlist=all_target_keywords[:wl_len],
#                         target_keywords=target_keywords,
#                         lure_keywords=lure_keywords)
#
#                if len(target_keywords) == kw_len\
#                        and len(lure_keywords) == kw_len:
#
#                    P.append(p)
#
#                if len(P) == number_of_paragraphs:
#                    break
#
#
#    return P

def paragraph_or_wordlist_to_str(paragraph, sep='\n-----\n'):

    text = textwrap.fill(paragraph['text'], 80)

    wordlist\
        = 'Wordlist: ' + textwrap.fill(', '.join(paragraph['wordlist']), 80)

    targets\
        = 'Targets: ' + textwrap.fill(', '.join(paragraph['target_keywords']), 80)

    lurewords\
        = 'Lures: ' + textwrap.fill(', '.join(paragraph['lure_keywords']), 80)

    if paragraph['as_wordlist']:
        return sep.join([wordlist, targets, lurewords])
    else:
        return sep.join([text, targets, lurewords])


def _paragraphs_or_wordlists_to_str(paragraphs_or_wordlists,
                                    sep='\n===========\n'):

    return '\n===========\n'.join(map(paragraph_or_wordlist_to_str, 
                                      paragraphs_or_wordlists)).encode('utf-8')


def write_wordlists_to_str(stimuli):

    sampled_wordlists = [stimulus for stimulus in stimuli
                          if stimulus['as_wordlist']]

    return _paragraphs_or_wordlists_to_str(sampled_wordlists)


def write_paragraphs_to_str(stimuli):

    sampled_paragraphs = [stimulus for stimulus in stimuli
                          if not stimulus['as_wordlist']]

    return _paragraphs_or_wordlists_to_str(sampled_paragraphs)


def write_paragraphs_to_file(stimuli, filename):

    with open(filename, 'w') as f:
        f.write(write_paragraphs_to_str(stimuli))


def write_wordlists_to_file(stimuli, filename):

    with open(filename, 'w') as f:
        f.write(write_wordlists_to_str(stimuli))


def write_stimuli_to_file(stimuli, filename, protocol=2):

    with open(filename, 'wb') as f:
        pickle.dump(stimuli, f, protocol=protocol)

def load_stimuli_from_file(filename):

    with open(filename, 'rb') as f:
        stimuli = pickle.load(f)

    return stimuli



def write_stimuli_as_configobj(stimuli, filename):

    sampled_paragraphs = [stimulus for stimulus in stimuli
                          if not stimulus['as_wordlist']]

    sampled_wordlists = [stimulus for stimulus in stimuli
                          if stimulus['as_wordlist']]

    text_memoranda = OrderedDict()
    for i, paragraph in enumerate(sampled_paragraphs):

        text_id = 'text_%d' % i

        text_memoranda[text_id] = OrderedDict()
        text_memoranda[text_id]['text']\
            = textwrap.fill(paragraph['text'].encode('utf-8'), 80)
        text_memoranda[text_id]['inwords']\
            = ','.join(paragraph['target_keywords']).encode('utf-8')
        text_memoranda[text_id]['outwords']\
            = ','.join(paragraph['lure_keywords']).encode('utf-8')

    wordlist_memoranda = OrderedDict()
    for i, wordlist in enumerate(sampled_wordlists):

        wordlist_id = 'wordlist_%d' % i

        wordlist_memoranda[wordlist_id] = OrderedDict()
        wordlist_memoranda[wordlist_id]['wordlist']\
            = ','.join(wordlist['wordlist']).encode('utf-8')
        wordlist_memoranda[wordlist_id]['inwords']\
            = ','.join(wordlist['target_keywords']).encode('utf-8')
        wordlist_memoranda[wordlist_id]['outwords']\
            = ','.join(wordlist['lure_keywords']).encode('utf-8')

    C = configobj.ConfigObj(indent_type='    ')
    C['text_memoranda'] = text_memoranda
    C['wordlist_memoranda'] = wordlist_memoranda
    C.filename = filename
    C.write()

def get_keywords(words, vocabulary, idf, keywords_as_dict=True):

    """
    Given a set of words, calculate the tfidf for all words in `vocabulary`
    using the `idf` dict. Sort the words by their tfidf and return either as a
    (keyword, tfidf) list or just a list of keywords.

    """

    tfidf= {word:count*idf.get(word, 0)
            for word, count in Counter(words).iteritems() 
            if word in vocabulary}

    keyword_dictionary_sorted = sorted(tfidf.items(), 
                                       key = lambda arg: arg[1],
                                       reverse=True)

    if keywords_as_dict:
        return keyword_dictionary_sorted
    else:
        return words_only(keyword_dictionary_sorted)


def get_neighbour_paragraphs(paragraph):

    xmlfilename = paragraph['corpus_filename']
    div1_index = paragraph['div1_index']
    paragraph_index = paragraph['paragraph_index']

    soup = get_corpus_file_soup(xmlfilename)

    div = soup.find_all('div', {'level': '1'})[div1_index]
    all_paragraphs_in_div = div.find_all('p')

    before_paragraph = all_paragraphs_in_div[paragraph_index-1]
    after_paragraph = all_paragraphs_in_div[paragraph_index+1]

    return (dict(text = before_paragraph.text,
                 words = get_words(before_paragraph),
                 wordcount = len(get_words(before_paragraph))),
            dict(text = after_paragraph.text,
                 words = get_words(after_paragraph),
                 wordcount = len(get_words(after_paragraph))))


class WordAssociations(object):
    

    def __init__(self, word_associations_filename):
        
        self.word_associations_data\
            = open(word_associations_filename).read().strip().split('\n')
            
        self.build_associations()
    

    def build_associations(self):
        
        self.associations = defaultdict(lambda: defaultdict(int))
        
        for row in self.word_associations_data:
            
            subject, stimulus, assoc1, assoc2, assoc3 = row.split(';')
            
            for associate in (assoc1, assoc2, assoc3):
            
                self.associations[stimulus][associate] += 1
    

    @property
    def stimulus_words(self):
        return self.associations.keys()
    

    @property
    def association_words(self):

        associates = [word.keys() for word in self.associations.values()]

        return dict.fromkeys(itertools.chain(*associates))
                

    def get_predicted_associates(self, text, vocab, threshold=None):
        
        words = [word for word in text['words']
                 if word in vocab]
        
        C = Counter()
        
        for word in words:
            c = dict(self.associations[word].items())
            C.update(c)
        
        predicted_associates = sorted(C.items(),
                                      key = lambda items: items[1],
                                      reverse=True)
        
        if threshold is None:
            return predicted_associates
        else:
            return [x for x,y in predicted_associates if y >= threshold]



def verify_sample_paragraphs(sample_paragraphs, word_association):

    association_words = word_association.association_words
    stimulus_words = word_association.stimulus_words

    for i, paragraph in enumerate(sample_paragraphs):

        wordlist_in_stimuli_words\
            = mean([w in stimulus_words for w in paragraph['wordlist']])
        
        keywords_in_stimuli_words\
            = mean([w in stimulus_words for w in paragraph['keywords']])
            
        targets_in_association_words\
            = mean([w in association_words for w in paragraph['target_keywords']])
            
        lure_in_association_words\
            = mean([w in association_words for w in paragraph['lure_keywords']])

            
        try:
            assert numpy.allclose([wordlist_in_stimuli_words,
                                   keywords_in_stimuli_words,
                                   targets_in_association_words,
                                   lure_in_association_words],
                                   [1.0, 1.0, 1.0, 1.0])

        except AssertionError: 
            print('Assertion error on papragraph %d' % i)


def randomly_sample_as_texts_and_wordlists(sampled_paragraphs, 
                                           random_seed=None):

    random = numpy.random.RandomState(random_seed)
    permutation = random.permutation


    N = len(sampled_paragraphs)
    half_N = int(N/2)
    I = permutation(N)
    
    for i, j in zip(I[:half_N], I[half_N:]):
        sampled_paragraphs[i]['as_wordlist'] = False
        sampled_paragraphs[j]['as_wordlist'] = True


def fix_stimulus_51(stimuli, paragraphs, vocabulary, idf):
    
    """
    Paragraph 51 contains the word "co-ordinator". The lure words contain the 
    word coordinator, which is technically not the same word from the BNC's 
    perspective. 
    However, it would be best to fix this.
    So, we will go through the list of keywords in the previous paragraph (which 
    was where the word "coordinator" was sampled) and find another lure word.
    """
    
    paragraph_51 = stimuli[51]
    index = paragraph_51['all_paragraphs_index']
    previous_paragraph_keywords\
        = get_keywords(paragraphs[index-1]['words'],
                             vocabulary,
                             idf, 
                             keywords_as_dict=False)
        
    for keyword in previous_paragraph_keywords:
        
        words = set(paragraph_51['keywords']\
                    + paragraph_51['lure_keywords']\
                    + paragraph_51['target_keywords'])
        
        word_stems = get_word_stems(words)
        
        if all([keyword not in words,
                stem(keyword) not in word_stems]):
            
            break
                
        
    paragraph_51['lure_keywords'][0] = keyword # replace "co-ordinator" in-place
