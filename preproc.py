# first build the dataset and then write it as a pickle file


# the place I store the raw data
DIR='/Users/morino/Downloads/sentiment_anaysis/tweet140/data';


TRAIN='train.csv';
TEST='test.csv';

TRAIN_PICK='train.pickle';
TEST_PICK='test.pickle';


TOPIC_TOKEN='K';
URL_TOKEN='URL';
MENTION_TOKEN='MENTION';


import os
import pickle


import pandas
import numpy as np
import re




# @Thanks https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
# A custom function to clean the text before sending it to pos tagging and lemmatize
def clean_text(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ").replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<").lower();
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    urlFinder=re.compile(r"(?P<protocol>\w+)://[\w./]+");
    text = mentionFinder.sub(MENTION_TOKEN, text); # remove those @MENTION, which is not of consideration
    # replace the url as where
    # replace HTML symbols
    text = urlFinder.sub(URL_TOKEN, text); # replace the concrete url as URL, which has nothing to do with the sentiment analysis in our assumption
    return text


# operations for the test set
reader=pandas.read_csv(open(os.path.join(DIR, TEST)),delimiter='","');
test_set=[];


for row in reader.values:
    topic_finder=re.compile(row[3], re.IGNORECASE);
    # replace the topic word as 'K'
    test_set.append((topic_finder.sub(TOPIC_TOKEN,clean_text(row[-1][:-1])),float(row[0][1:])));




test_storage=open(os.path.join(DIR, TEST_PICK), 'wb');
pickle.Pickler(test_storage).dump(test_set);

test_storage.close();




# operations for the training set
reader=pandas.read_csv(open(os.path.join(DIR, TRAIN)),delimiter='","');
train_set=[];


for row in reader.values:
    topic_finder=re.compile(row[3], re.IGNORECASE);
    train_set.append((topic_finder.sub(TOPIC_TOKEN,clean_text(row[-1][:-1])),float(row[0][1:])));

train_storage=open(os.path.join(DIR, TRAIN_PICK), 'wb');
pickle.Pickler(train_storage).dump(train_set);

train_storage.close();
