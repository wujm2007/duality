# which is a neural network model for the sentiment evaluation, which based on a deeply cleaned token list



# deprecated
# begin crafting the neural models

# in order to represent a meaningful model, a lower-level framework should be used
TRAIN_PICK='train.pickle';
TEST_PICK='test.pickle';


NORM_TRAIN_PICK='norm_train.pickle';
NORM_TEST_PICK='norm_test.pickle';

VOCAB_PICK='vocab.pickle'; # for dumping training vocabulary
REP_RULE_PICK='rule.pickle'; # for dumping training replace rule
FREQ_DIST_PICK='freq.pickle'; # for dumping training freq list


NORMAL_MAP_PICK='mapping.pickle'; # for dumping indexing of vocabulary
INVERSE_MAP_PICK='reversed_mapping.pickle'; # for dumping the inversed dictioinary of the vocabulary


MODEL_PATH='record/weights.{epoch:02d}-{val_loss:.2f}.hdf5';

import os

import keras
# which should still use a relatively lower-level api to do the manipulation
from keras.models import Model
from keras.layers import Dense, Input, Flatten, MaxPooling1D, TimeDistributed
from keras.layers import Embedding, Conv1D, Conv2D, AveragePooling1D, LSTM, Dropout, ZeroPadding1D, RepeatVector
from keras.preprocessing import sequence as SequenceHelper
from .utils import util_dump, util_load


from .context import DIR


# which may be done in a batch level
# a sequence of tokens
# return a value from [-1, 1]
def nn_analysis(tokens):
    return 0.0;




# this model not finished right now
# which return a keras neural model(not compiled)
def construct_model(vocab_size, embed_dim, low, high):
    # always low=3, high=7
    embedding_layer= Embedding(vocab_size+1, embed_dim, input_length=high);
    seq_input = Input(shape=(high, ), dtype='int32');
    x = LSTM(output_dim=64, return_sequences=True)(embedding_layer(seq_input));
    x = Dropout(0.5)(x);
    x = LSTM(output_dim=32)(x);
    x=  Dropout(0.25)(x);
    pred  = Dense(1, activation='tanh')(Dense(8, activation='tanh')(x));
    return Model(seq_input, pred);







# trainning the model according to the

# vocab_size: the size of the vocabulary, predetermined hyper-parameter
# embed_dim:  the dimension of the word vector, predetermined hyper-parameter
# low: the smallest number of sentence components
# so as high
def proc(train_set, test_set, vocab_size, embed_dim=128, low=3, high=7, _batch_size=128):
    cp_point= keras.callbacks.ModelCheckpoint(MODEL_PATH, verbose=1, monitor='val_loss',save_best_only=False, save_weights_only=False, mode='auto', period=1)
    print('Begin Padding');
    padded_x_train= SequenceHelper.pad_sequences(train_set['x'], maxlen=high, value=vocab_size, padding='post');
    padded_x_test=  SequenceHelper.pad_sequences(test_set['x'], maxlen=high, value=vocab_size, padding='post');
    print('End Padding');
    model=construct_model(vocab_size, embed_dim, low, high);
    model.compile(optimizer='adagrad',loss='mse',metrics=['accuracy']);
    model.summary();
    model.fit(padded_x_train, train_set['y'], nb_epoch=10, batch_size=_batch_size, callbacks=[cp_point], validation_data=(padded_x_test, test_set['y']));
    return model;



if(__name__=='__main__'):
    train_set=util_load(os.path.join(DIR, NORM_TRAIN_PICK));
    test_set=util_load(os.path.join(DIR, NORM_TEST_PICK));
    vocab_size=10000;
    proc(train_set, test_set, vocab_size);
