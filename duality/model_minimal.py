# the minimal part for the sentiment analysis and the command extraction
# without a neural net model

import os
import pickle
import re
import difflib
import queue

# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


import spacy
import numpy as np
import Levenshtein as leven

from .context import voice_name_dict, song_name_dict;

"""
We can consider a document as (S_i), and we assume we are able to evaluate the
score of each simple sentence without any outer information. To model the seq, we choose
a simple RNN, since the number of sentence is not too big, but it's variable.


Although a markov chain model can be choosen, a rnn is more powerful to capture the dependency

@param: doc, a roughly preprocessed raw text used in the dataset
@param: a spacy english instance
@param: analyzer, a vader analyzer instance
"""
def doc_eval(doc, nlp, analyzer):
    components= list();
    commands=list();
    _doc= nlp(doc);
    for sent in _doc.sents:
        commands.extend(_imperative_classify(sent)); # extend the command list a.t. the single sentences
        components.append(sent_eval(sent.text, nlp, analyzer)); # evaluate each single sentence's sentiment values
    return _doc_eval(components), commands;

"""
@param: components, a list of sentence sentiment evaluations
"""
def _doc_eval(components):
    return np.average(components);


"""
Evaluation process for a single sentence, which has been roughly cleaned and classified(verb extracted for imperative sentence)

@param: alpha, a linear combination parameter
@param: sent, the raw text, without a deep clean
@param: nlp, spacy english instance
@param: analyzer, vader sentiment instance
@param:

"""
def sent_eval(sent, nlp, analyzer):
    # deep clean done here
    lexicon_values = analyzer.polarity_scores(sent);
    return lexicon_values['compound'];




"""
To evaluate whether a sentence is an imperative sentence, which is almost based on whether
there contains a verb with no subject
@return: the verb phase contained in this sentence, a list is better

"""
def _imperative_classify(soup):
    verbs = list();
    # collect the verb with no tag
    # extract the verb
    for word in soup:
        is_you=-1; # default -1, means no subject
        if(word.pos_=='VERB'):
            # check whether the subject is you
            for c in word.children:
                if(c.dep_=='nsubj'):
                    is_you= 1 if c.text=='you' else 0;
                    break;
            # deal with conjugated words with no explicit subject
            if(word.dep_=='conj' and is_you==-1):
                if(word.head in verbs):
                    verbs.append(word);
                continue;
            # deal with common situations
            if(word.tag_=='VB' and is_you==-1):
                verbs.append(word);
            elif((word.tag_=='VB' or word.tag_=='VBP') and is_you==1):
                verbs.append(word);
    return verbs; # currently, only filter out the move, in future, more things will be parsed out in the tree



"""
a wrapper for the above method
"""
def imperative_classify(sent, nlp):
    return _imperative_classify(nlp(sent));



# auxiliary method for song selection
def _find_song(name):
    max_similarity=-0.1;
    ind=-1;
    threshold=0.5;
    print("match with name %s" % (name));
    for i in song_name_dict.keys():
        sim=leven.jaro(song_name_dict[i], name)
        print("similarity %f" % (sim));
        if(sim>=max_similarity):
            ind=i;
            max_similarity=sim;

    return ind if(max_similarity>=threshold) else None;

def _random_song():
    return -1;



"""
1. sing the song named blissy land [sing -> 1 dobj  [acl.] -> (conj) ->   ]
2. sing blissy land [song -> (1 dobj)]
3. sing the song blissy land for me [sing -> 2 dobj]
"""

"""
1. dance to happy birthday
2. dance to happy birthday for me
3. dance to the song named happy birthday [the same case for]
"""



"""
this method will be invoked when miku finds the command to sing or dance

if it can not find a proper song with a potential name, just return something random

@param: sent, the sentence from where comes the command
@param: command, the recognized command
"""
def _recognize_song_name(sent, command):
    song_name=None;
    prep_to=command; # it is by default the word song
    if(command.text=="dance"):
        for c in command.children:
            if(c.dep_=="prep" and c.text=="to"):
                prep_to=c;
                break;
        if(prep_to==None):
            return _random_song();

    # only deal with the dance sentence with the prep 'to' and sing-, which is a conventional english usage
    if(prep_to!=None):
        beg_pos=prep_to.i+1;
        for c in prep_to.children:
            if(c.dep_=='pobj' or c.dep_=='dobj'):
                # first try with the direct object
                potential= _find_song(sent[beg_pos:c.i+1].text);
                beg_pos=c.i+1;
                if(potential==None):
                    # second trial, find the pp-clause
                    for k in c.children:
                        if(k.dep_=='acl' and k.i+1<len(sent)):
                            print(k.text);
                            potential=_find_song(sent[k.i+1:].text);
                            break;
                    # the final chance,
                    potential=_find_song(sent[beg_pos:].text);
        return potential if(potential!=None) else _random_song();
    # can't find a proper song
    return _random_song();






"""
@param: sentiment_param, a float
@param: is_command, a boolean
"""
# to select a voice a.t. the sentiment param and whether it is a command. from the voice dictionary
def select_voice(sentiment_param, is_command):
    # is command determines whether the voice selection  in a partially random bev
    ind= 0;
    if(sentiment_param>0.2):
        ind=1;
    elif(sentiment_param<-0.2):
        ind=2;
    else:
        ind=3;

    if(is_command and np.random.rand(1)[0]<=0.5):
        # then do a random choice from the permission and the current sentiment voice
        ind=4;
    return ind;


def select_action(command):
    return 1;



# for miku core usage
def analyze_command(sent, command):
    threshold=0.9;
    if(leven.jaro(command.text, 'sing')>=threshold or leven.jaro(command.text, 'dance')>=threshold):
        return 200, _recognize_song_name(sent, command);
    else:
        return select_action(command), -1;



# which is a soft waterlevel queue
class MemorySeq:
    def __init__(self, max_size):
        self.q=list();
        self.max_size=max_size;

    def put(self, item):
        if(len(self.q)+1>self.max_size):
            self.get();
        self.q.append(item);

    def get(self):
        return self.get(0);

    def get(self, i):
        if(i>=len(self.q)):
            return 999;
        else:
            return self.q[i];

    def size(self):
        return len(self.q);









# the interface
class MikuCore:
    def __init__(self):
        self.analyzer= SentimentIntensityAnalyzer();
        print('記憶体ロード、待ってくださいね');
        self.nlp= spacy.load('en');
        print('完成です');
        self.memory_size=20;
        self.memory_pool={
            'actions':MemorySeq(self.memory_size),
            'rewards':[],
            'options':MemorySeq(self.memory_size),
            'voices':MemorySeq(self.memory_size)
        }; # for storing the histo of interactions;

    def process(self, comment):
        sentiment_value, commands = doc_eval(comment, self.nlp, self.analyzer);
        self.memory_pool['rewards'].append(sentiment_value);
        print(commands);
        for command in commands:
            action_id, option = analyze_command(self.nlp(comment), command);
            self.memory_pool['actions'].put(action_id); # these two are one-to-one res.
            self.memory_pool['options'].put(option);
        self.memory_pool['voices'].put(select_voice(sentiment_value, len(commands)>0));

    # this modifies its own parameters to improve the entertaining skills
    def improve(self):
        pass;

    def respond(self):
        print ("\t %s\t %s\t%s\t%s" % ("Action", "Voice" ,"Reward", "Option"));
        for i in range(self.memory_pool['actions'].size()):
            print('\t%d\t%d\t%f\t%d' % (self.memory_pool['actions'].get(i), self.memory_pool['voices'].get(i), self.memory_pool['rewards'][i] if(i<len(self.memory_pool['rewards'])) else 999.0, self.memory_pool['options'].get(i)))







docs=[
'go and get me something',
'you go and get me something',
'you should go and get me something',
'you go and i may give you something',
'i go and you should tell me something',
'tell me something about you',
'never thank me',
'i want you to hug me', # a complex case
'sing the song named blissy land',
'sing blissy land',
'sing the song blissy land for me',
'to dance to happy birthday',
'to dance to happy birthday for me',
'to dance to the song happy birthday for me',
'to dance to the song named happy birthday'
]


if(__name__=='__main__'):

    # this will finally be initiated when starting the server
    # which works as a global instance, based on some lexicon methods
    miku= MikuCore();


    for i, doc in enumerate(docs):
        miku.process(doc);
    miku.respond();
