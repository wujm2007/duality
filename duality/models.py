
# this file works as an interface for the client to invoke the verb extractor and sentiment evaluator
# also the dictionary builder




TRAIN_PICK='train.pickle';
TEST_PICK='test.pickle';


NORM_TRAIN_PICK='norm_train.pickle';
NORM_TEST_PICK='norm_test.pickle';

VOCAB_PICK='vocab.pickle'; # for dumping training vocabulary
REP_RULE_PICK='rule.pickle'; # for dumping training replace rule
FREQ_DIST_PICK='freq.pickle'; # for dumping training freq list


NORMAL_MAP_PICK='mapping.pickle'; # for dumping indexing of vocabulary
INVERSE_MAP_PICK='reversed_mapping.pickle'; # for dumping the inversed dictioinary of the vocabulary


import os
import pickle
import re
import difflib

# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords
from nltk.probability import FreqDist



import spacy
import numpy as np

import Levenshtein as leven



# from .nn_model import nn_analysis
from .context import DIR
from .utils import util_dump, util_load



SPECIAL=['K', 'URL', 'MENTION'];


# load the train set and test set
#
#
#
# ## recover the cleared data
# train_storage=open(os.path.join(DIR, TRAIN_PICK), mode='rb');
# train_set=pickle.load(train_storage);
# train_storage.close();
#
#
# test_storage=open(os.path.join(DIR, TEST_PICK), mode='rb');
# test_set=pickle.load(test_storage);
# test_storage.close();










def is_stop_word(word, stop_words):
    return (word in stop_words);


"""
clean the text.
1. Remove the stop words
2. lemmatize the words

@return a list of tokens of the words
"""
def semantic_clean(sent, nlp, stop_words):
    _sent = nlp(sent);
    tokens = [];
    wordFinder= re.compile(r'^\w+$');
    # collect tokens that is not in stop words and is not of entity type and not the punctuation
    for word in _sent:
        text=word.text;
        if(word.ent_type_=='' and (not is_stop_word(text, stop_words)) and wordFinder.match(text)!=None):
            # which thus avoid those special upper case words to be lowered
            if(not word.text in SPECIAL):
                tokens.append(word.lemma_);
    return tokens;






"""
@param: a list of tuple (sentence, value)
@param: threshold , a real number in the [0, 1 ] to determine whether the words at the hapaxis should be
put into some bins that currently exist


some experiment should be conducted to determine the ratio
for example, SequenceMatcher(a='yeahhhhhhhhh', b='yeah').ratio() = 0.5


"""
def build_vocabulary(doc_set, nlp, stop_words, threshold=0.8, vocab_size=10000):
    # this now more
    # the pickled data is of this form [(sent., 0| 2| 4)]
    fdist=FreqDist();
    count=0;
    log_times=10000;
    total_size= len(doc_set);
    for pair in doc_set:
        # for log
        count+=1;
        if(np.mod(count, log_times)):
            print('%f finished' % (count/total_size));
        for word in semantic_clean(pair[0], nlp, stop_words):
            fdist[word]+=1;
    fdist.pprint(maxlen=50);
    print('Begin flush freq dist into disk')
    util_dump(fdist, os.path.join(DIR, FREQ_DIST_PICK));
    print('End flush freq dist into disk')
    return _build_vocabulary(fdist, threshold, vocab_size);


def _build_vocabulary(fdist, threshold, vocab_size):
    # remove the top three and the hapaxis
    # now consider the words that
    log_times=10000;

    sorted_fdist= sorted(fdist.items(), key=lambda item: -item[1]); # sort it in a decrement order
    vocab = set([pair[0] for pair in sorted_fdist[:vocab_size]]); # choose those common words
    hapaxes= set(fdist.keys()).difference(vocab);

    print('Begin flush vocabulary into disk')
    util_dump(vocab, os.path.join(DIR, VOCAB_PICK));
    print('End flush vocabulary into disk')

    replace_rule={};
    count=0;
    total_size= len(hapaxes);
    # the processing here can be illustrated as YEAH and YEAHHHHHH or some other variants
    for rare in hapaxes:
        count+=1;
        if(np.mod(count, log_times)):
            print('%f finished' % (count/total_size));
        # do some sequence matcher here and construct some mappings
        for word in vocab:
            if(leven.jaro(rare, word)>=threshold and set(rare)==set(word)):
                # find first one then break;
                replace_rule[rare]=word;
                # and remove the word from the all_vocabs
                try:
                    vocab.remove(rare);
                except KeyError:
                    pass;
                break;
    # flush the replace rule into the disk
    print('Begin flush replace rule into disk');
    util_dump(replace_rule, os.path.join(DIR, REP_RULE_PICK))
    print('End flush replace rule into disk');
    return vocab, replace_rule;


"""
@param: vocabulary, the prepickled set
"""
def create_mapping(vocabulary):
    normal_mapping = list(vocabulary);
    inversed_mapping = dict();
    for i in range(len(normal_mapping)):
        inversed_mapping[normal_mapping[i]] = i;
    util_dump(normal_mapping, os.path.join(DIR, NORMAL_MAP_PICK));
    util_dump(inversed_mapping, os.path.join(DIR, INVERSE_MAP_PICK));
    return normal_mapping, inversed_mapping;







# just load the vocabulary
# vocabulary = util_load(os.path.join(DIR, VOCAB_PICK));
# replace_rule= util_load(os.path.join(DIR, REP_RULE_PICK));








"""
We can consider a document as (S_i), and we assume we are able to evaluate the
score of each simple sentence without any outer information. To model the seq, we choose
a simple RNN, since the number of sentence is not too big, but it's variable.


Although a markov chain model can be choosen, a rnn is more powerful to capture the dependency

@param: doc, a roughly preprocessed raw text used in the dataset
@param: a spacy english instance
@param: analyzer, a vader analyzer instance
"""
def doc_eval(doc, nlp, analyzer, stop_words=None, normal=None, inversed=None, replace_rule=None):
    components= list();
    commands=list();
    _doc= nlp(doc);
    for sent in _doc.sents:
        commands.extend(_imperative_classify(sent)); # extend the command list a.t. the single sentences
        components.append(sent_eval(sent.text, nlp, analyzer, stop_words, normal, inversed, replace_rule)); # evaluate each single sentence's sentiment values
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
def sent_eval(sent, nlp, analyzer, stop_words, normal, inversed, replace_rule, alpha=1.0):
    # deep clean done here
    lexicon_values = analyzer.polarity_scores(sent);
    # ml_values= _sent_eval(semantic_clean(sent, nlp, stop_words), normal, inversed, replace_rule);
    # print('Lexicon: %f' % (lexicon_values['compound']));
    # print('Neural: %f' % (ml_values));
    return lexicon_values['compound'];




"""
@param: tokens, deep cleaned [deprecated]
"""
def _sent_eval(tokens, normal, inversed, replace_rule):
    outcome = final_clean(tokens, normal, inversed, replace_rule);
    print(outcome);
    return nn_analysis(outcome);

"""
which replaces the rare words that can be replaced and remove the rare words in a semanticly cleaned raw text
"""

def final_clean(tokens, normal, inversed, replace_rule):
    prob=0.5;
    for index, token in enumerate(tokens):
        # replace the word with a probability
        if(not token in normal):
            tokens[index]=replace_rule[token] if(token in replace_rule.keys() and np.random.rand(1)[0]>=prob) else None;
    return [inversed[x] for x in tokens if x is not None];



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




"""
the data set here is of the form of [tuple(text, label)]
"""


def rearrange(data_set, nlp, stop_words, normal, inversed, replace_rule, low=3, high=7):
    cleaned_data_set = {
        'x':[],
        'y':[]
    };
    count=0;
    log_times=10000;
    total_size=len(data_set);
    for pair in data_set:
        count+=1;
        if(np.mod(count, log_times)):
            print('%f finished' % (count/total_size));
        word_arr=final_clean(semantic_clean(pair[0], nlp, stop_words), normal, inversed, replace_rule);
        # normalize the data set
        if(len(word_arr)>=low and len(word_arr)<=high):
            cleaned_data_set['x'].append(final_clean(semantic_clean(pair[0], nlp, stop_words), normal, inversed, replace_rule));
            cleaned_data_set['y'].append(pair[1]/2-1); # normalize a.t. 0,2,4 as neg, neural, pos
    return cleaned_data_set;









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
    # which works as a global instance, based on some lexicon methods
    analyzer= SentimentIntensityAnalyzer();
    # load the nlp
    print('load corpus... please wait');
    nlp= spacy.load('en');
    stop_words=set(stopwords.words('english')); # load the english stop words for a deeper clean
    print('finished');

    # train_set= util_load(os.path.join(DIR, TRAIN_PICK));
    # test_set=util_load(os.path.join(DIR, TEST_PICK));


    # train_set=util_load(os.path.join(DIR, NORM_TRAIN_PICK));
    # test_set=util_load(os.path.join(DIR, NORM_TEST_PICK));

    # print(len(train_set['x']));

    ######################## this part for building vocabulary on the training set(Never try it on your own machine, which takes a really long time) ##############
    # a global vocabulary
    # vocabulary, replace_rule= build_vocabulary(train_set, nlp, stop_words, vocab_size=10000);
    # normal_mapping, inversed_mapping = create_mapping(vocabulary);
    ######################## this part for building vocabulary on the training set ###################

    # replace_rule= util_load(os.path.join(DIR, REP_RULE_PICK));
    # normal= util_load(os.path.join(DIR, NORMAL_MAP_PICK));
    # inversed= util_load(os.path.join(DIR, INVERSE_MAP_PICK));

    # rearranged_test_set = rearrange(test_set, nlp, stop_words, normal, inversed, replace_rule);
    # util_dump(rearranged_test_set, os.path.join(DIR, NORM_TEST_PICK));
    #
    # rearranged_train_set= rearrange(train_set, nlp, stop_words, normal, inversed, replace_rule);
    # util_dump(rearranged_train_set, os.path.join(DIR, NORM_TRAIN_PICK));



    for i, doc in enumerate(docs):
        value, commands=doc_eval(doc, nlp, analyzer);
        print("==========================BEGIN TEST %d=====================" %(i));
        print('Sentence: %s' % (doc));
        print('Sentiment Value: %f' % (value));
        print(commands);
        # print('Vocabulary Size: %d' %(len(normal)))
        print("==========================BEGIN TEST %d=====================" %(i));
