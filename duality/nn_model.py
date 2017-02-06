# which is a neural network model for the sentiment evaluation, which based on a deeply cleaned token list

# begin crafting the neural models

# which should still use a relatively lower-level api to do the manipulation
import keras
from .utils import util_dump, util_load

# which may be done in a batch level
# a sequence of tokens
# return a value from [-1, 1]
def nn_analysis(tokens):
    return 1.0;

# which return a keras neural model
def construct_model():
    pass;







# trainning the model according to the

# vocab_size
def proc(train_set, test_set, vocab_size):
    pass;


if(__name__=='__main__'):
    train_set=util_load(os.path.join(DIR, NORM_TRAIN_PICK));
    test_set=util_load(os.path.join(DIR, NORM_TEST_PICK));
    
