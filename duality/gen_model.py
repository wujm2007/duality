# this is for a deep reinforcement generative model
import numpy as np


def generate(doc):
    return _generate(doc).encode('ascii');

def _generate(doc):
    rand_num=np.random.rand(1)[0];
    if(rand_num<=0.5):
        return "-1.0";
    else:
        return "1.0";
