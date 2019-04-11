""" 

A module of tools for the processing and preparation of data collected from
a wilhelm based online experiment server.

"""


import shutil
import json
import tarfile
import tempfile
import urllib
import os
import pandas

import random

fake_subject_uids = [] # Override this at run time using get_fake_subject_uids

def get_data(url, rmdir=True):

    """
    Download a wilhelm experiment data archive and extract the data from the
    data.json file.

    """
    
    tmpdir = tempfile.mkdtemp()
    f = urllib.URLopener()
    f.retrieve(url, os.path.join(tmpdir, "data.tar.bz2"))

    tar = tarfile.open(os.path.join(tmpdir, 'data.tar.bz2'))
    tar.extractall(tmpdir)
    tar.close()

    data = json.load(open(os.path.join(tmpdir, 'data.json')))

    if rmdir:
        shutil.rmtree(tmpdir) # Delete the tmpdir.
    
    return data


def get_fake_subject_uids(fake_subject_list_filename):

    return [uid[:7] for uid in 
            open(fake_subject_list_filename).read().strip().split()]


class FakeSubjectException(Exception):
    pass


def parse_subject_information(subject_information, 
                              keys = ('Subject ID', 'Age', 'Sex')):

    try:
        assert not subject_information['Temporary subject']
        assert subject_information['Subject ID'] not in fake_subject_uids
        return map(subject_information.get, keys)
    except AssertionError:
        raise FakeSubjectException
    
def parse_session_information(session_information, 
                              keys=('Session ID',)):
    d = map(session_information.get, keys)
    d[0] = d[0][:7]
    return d

def parse_client_information(client_information, 
                              keys = ('browser', 
                                      'operating system', 
                                      'is_pc', 
                                      'is_tablet',
                                      'is_mobile',
                                      'is_touch_capable')):

    return map(client_information.get, keys)

def parse_geoip_information(geoip_information,
                            keys = ('city', 'country code')):
    
    return map(geoip_information.get, keys)

def parse_live_sessions(live_sessions):
    
    def _parse_live_session(live_session):
        return [live_session['Initiated']]\
               + parse_client_information(live_session['Client'])\
               + parse_geoip_information(live_session['GeoIP'])
    
    live_sessions_dict = {}
    for live_session in live_sessions:
        live_sessions_dict[live_session['uid']] = _parse_live_session(live_session)
    
    return live_sessions_dict

def parse_session(session):
    
    try:
        
        live_sessions_dict = parse_live_sessions(session['Live sessions'])
        
        session_information = parse_session_information(session['Experiment session'])
        subject_information = parse_subject_information(session['Subject information'])
        
        return session_information + subject_information
    except FakeSubjectException:
        
        return []

def get_tmp_uid(k=7):
    ''' Generate a k hex digit unique identifier.'''
    uidformat = '%%0%dx' % k
    return uidformat % random.getrandbits(k*4) # Each hex digit is 2^4 bits.

def parse_textrecognition_slides(slides):
    if slides[0]['Name']:
        pass
    

def parse_slide_information(slide_information,
                            keys = ('Completed',)):
    
    return map(slide_information.get, keys)   

def parse_textdisplay_information(text_display_information,
                                   keys = ('Text ID', 'Reading time')):
    return map(text_display_information.get, keys)

def parse_word_recall_test_information(word_recognition_test_information):

    return word_recognition_test_information['Word recall test data']


def parse_word_recognition_test_information(word_recognition_test_information,
                                            keys = ('stimulus_word', 
                                                    'expected_response',
                                                    'order',
                                                    'hit',
                                                    'response',
                                                    'response_accuracy',
                                                    'response_latency')):
    all_trials = []
    for word_recognition_trial in word_recognition_test_information['Word recognition test data']:
        
        all_trials.append(map(word_recognition_trial.get, keys))
    
    return all_trials                                               

widget_parsers_map = dict(TextDisplay = parse_textdisplay_information,
                          WordRecognitionTest = parse_word_recognition_test_information,
                          WordRecallTest = parse_word_recall_test_information)


def parse_textrecognitiontest_information(slide_information):
    
    data_dict = {}
    
    for widget in slide_information['Widgets']:
        widget_name = widget['Name']

        if widget_name in widget_parsers_map:
            data_dict[widget_name] = widget_parsers_map[widget_name](widget)

    slide_info = parse_slide_information(slide_information)
    tmp_uid = get_tmp_uid()
    text_data = data_dict['TextDisplay']
    all_data = []
    for trial_data in data_dict['WordRecognitionTest']:
        all_data.append([tmp_uid] + slide_info + text_data + trial_data)
        
    return all_data


def parse_textrecalltest_information(slide_information):
    
    data_dict = {}
    
    for widget in slide_information['Widgets']:
        widget_name = widget['Name']

        if widget_name in widget_parsers_map:
            data_dict[widget_name] = widget_parsers_map[widget_name](widget)

    slide_info = parse_slide_information(slide_information)
    tmp_uid = get_tmp_uid()
    text_data = data_dict['TextDisplay']
    all_data = []
    for trial_data in data_dict['WordRecallTest']:
        all_data.append([tmp_uid] + slide_info + text_data + [trial_data])
        
    return all_data


def parse_slides(slides):
    x = []
    for slide in slides:
        if slide['Name'] == 'TextRecognitionMemoryTest':
            x.extend(parse_textrecognitiontest_information(slide))          
    return x

def parse_textrecall_slides(slides):
    x = []
    for slide in slides:
        if slide['Name'] == 'TextRecallMemoryTest':
            x.extend(parse_textrecalltest_information(slide))          
    return x


def parse_textrecall_session(session):

    session_info = parse_session(session)
    trials = parse_textrecall_slides(session['Playlist information']['Slides'])
    
    if session_info:
        x = []
        for trial in trials:
            x.append(session_info + trial)
        return x
    else:
        return []


def parse_textrecognition_session(session):
    session_info = parse_session(session)
    trials = parse_slides(session['Playlist information']['Slides'])
    if session_info:
        x = []
        for trial in trials:
            x.append(session_info + trial)
        return x
    else:
        return []

def get_textrecall_data(sessions, seed=None):

    if seed:
        random.seed(seed)

    data = []
    for session in sessions:
        _data = parse_textrecall_session(session)
        if _data:
            data.extend(_data)

    Df = pandas.DataFrame(data)

    headers = ['session', 
               'subject', 
               'age', 
               'sex', 
               'slide', 
               'completed', 
               'text', 
               'readingtime', 
               'word']

    Df.columns = headers


    return Df
 
def get_textrecognition_data(sessions, seed=None):

    if seed:
        random.seed(seed)

    data = []
    for session in sessions:
        _data = parse_textrecognition_session(session)
        if _data:
            data.extend(_data)
    Df = pandas.DataFrame(data)

    headers = ['session', 
               'subject', 
               'age', 
               'sex', 
               'slide', 
               'completed', 
               'text', 
               'readingtime', 
               'word', 
               'expected',
               'order',
               'hit',
               'response',
               'correct',
               'rt']

    Df.columns = headers

    return Df

data2csv = lambda data: '\n'.join([','.join(map(str, datum)) for datum in data])
