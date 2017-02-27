# the minimal part for the sentiment analysis and the command extraction
# without a neural net model

import os
import pickle
import re
import difflib

# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


import spacy
import numpy as np




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
    print(components);
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


docs=[
'go and get me something',
'you go and get me something',
'you should go and get me something',
'you go and i may give you something',
'i go and you should tell me something',
'tell me something about you',
'never thank me',
'i want you to hug me', # a complex case
]


if(__name__=='__main__'):

    # this will finally be initiated when starting the server
    # which works as a global instance, based on some lexicon methods
    analyzer= SentimentIntensityAnalyzer();
    # load the nlp
    print('load corpus... please wait');
    nlp= spacy.load('en');
    print('finished');


    for i, doc in enumerate(docs):
        value, commands=doc_eval(doc, nlp, analyzer);
        print("==========================BEGIN TEST %d=====================" %(i));
        print('Sentence: %s' % (doc));
        print('Sentiment Value: %f' % (value));
        print(commands);
        # print('Vocabulary Size: %d' %(len(normal)))
        print("==========================BEGIN TEST %d=====================" %(i));
