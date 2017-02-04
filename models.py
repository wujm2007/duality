
# this file works as an interface for the client to invoke the verb extractor and sentiment evaluator
# also the dictionary builder

DIR='/Users/morino/Downloads/sentiment_anaysis/tweet140/data';


TRAIN_PICK='train.pickle';
TEST_PICK='test.pickle';
VOCAB_PICK='vocab.pickle'; # for dumping training vocabulary
REP_RULE_PICK='rule.pickle'; # for dumping  training replace rule

import os
import pickle
import re
import difflib

# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
import spacy
import numpy as np

import Levenshtein as leven



from .nn_model import nn_analysis

SPECIAL=['K', 'URL', 'MENTION'];


# which works as a global instance, based on some lexicon methods
analyzer= SentimentIntensityAnalyzer();
# load the nlp
print('load corpus... please wait');
nlp= spacy.load('en');
stop_words=set(stopwords.words('english')); # load the english stop words for a deeper clean
print('finished');





# load the train set and test set



# recover the cleared data
train_storage=open(os.path.join(DIR, TRAIN_PICK), mode='rb');
# test_storage=open(os.path.join(DIR, TEST_PICK), mode='rb');


train_set=pickle.load(train_storage);
# test_set=pickle.load(test_storage);


train_storage.close();
# test_storage.close();







def is_stop_word(word):
    return (word in stop_words);



"""
clean the text.
1. Remove the stop words
2. lemmatize the words

@return a list of tokens of the words
"""
def semantic_clean(sent):
    _sent = nlp(sent);
    tokens = [];
    wordFinder= re.compile(r'^\w+$');
    # collect tokens that is not in stop words and is not of entity type and not the punctuation
    for word in _sent:
        text=word.text;
        if(word.ent_type_=='' and (not is_stop_word(text)) and wordFinder.match(text)!=None):
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
def build_vocabulary(doc_set, threshold=0.8):
    # this now more
    # the pickled data is of this form [(sent., 0| 2| 4)]
    fdist=FreqDist();
    for pair in doc_set:
        for word in semantic_clean(pair[0]):
            fdist[word]+=1;
    fdist.pprint(maxlen=50);
    # remove the top three and the hapaxis
    # now consider the words that
    all_vocabs= set(fdist.keys());
    one_times = set(fdist.hapaxes());
    more_than_one_times = all_vocabs.difference(one_times); # should we just contain those occur more than once
    replace_rule={};
    # the processing here can be illustrated as YEAH and YEAHHHHHH or some other variants
    for rare in one_times:
        # do some sequence matcher here and construct some mappings
        for vocab in more_than_one_times:
            if(leven.jaro(rare, vocab)>=threshold and set(rare)==set(vocab)):
                # find first one then break;
                replace_rule[rare]=vocab;
                # and remove the word from the all_vocabs
                try:
                    all_vocabs.remove(rare);
                except KeyError:
                    pass;
                break;
    return more_than_one_times, replace_rule;


# a global vocabulary
vocabulary, replace_rule= build_vocabulary(train_set);

vocab_storage=open(os.path.join(DIR, VOCAB_PICK), 'wb');
pickle.Pickler(vocab_storage).dump(vocabulary);
vocab_storage.close();

replace_rule_storage= open(os.path.join(DIR, REP_RULE_PICK), 'wb');
pickle.Pickler(replace_rule_storage).dump(replace_rule);
replace_rule_storage.close();

"""
We can consider a document as (S_i), and we assume we are able to evaluate the
score of each simple sentence without any outer information. To model the seq, we choose
a simple RNN, since the number of sentence is not too big, but it's variable.


Although a markov chain model can be choosen, a rnn is more powerful to capture the dependency

@param: doc, a roughly preprocessed raw text used in the dataset
"""
def doc_eval(doc):
    components= list();
    commands=list();
    _doc= nlp(doc);
    for sent in _doc.sents:
        commands.extend(_imperative_classify(sent)); # extend the command list a.t. the single sentences
        components.append(sent_eval(sent.text)); # evaluate each single sentence's sentiment values
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

"""
def sent_eval(sent, alpha=0.1):
    # deep clean done here
    lexicon_values = analyzer.polarity_scores(sent);
    ml_values= _sent_eval(semantic_clean(sent), vocabulary, replace_rule);
    print('Lexicon: %f' % (lexicon_values['compound']));
    print('Neural: %f' % (ml_values));
    return (alpha*lexicon_values['compound']+(1.0-alpha)*ml_values);




"""
@param: tokens, deep cleaned
"""
def _sent_eval(tokens, vocab, replace_rule):
    return nn_analysis(final_clean(tokens, vocab, replace_rule));

"""
which replaces the rare words that can be replaced and remove the rare words in a semanticly cleaned raw text
"""

def final_clean(tokens, vocab, replace_rule):
    prob=0.5;
    for index, token in enumerate(tokens):
        # replace the word with a probability
        if(not token in vocab):
            tokens[index]=replace_rule if(token in replace_rule.keys() and np.random.rand(1)[0]>=prob) else None;
    return [x for x in tokens if x is not None];



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
def imperative_classify(sent):
    return _imperative_classify(nlp(sent));





docs=[
'go and get me something',
'you go and get me something',
'you should go and get me something',
'you go and i may give you something',
'i go and you should tell me something',
'tell me something about you',
'never thank me',
'I want you to hug me', # a complex case
]



if(__name__=='__main__'):
    for doc in docs:
        value, commands=doc_eval(doc);
        print('Sentence: %s' % (doc));
        print('Sentiment Value: %f' % (value));
        print(commands);
