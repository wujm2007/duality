import pickle

def util_load(path):
    f= open(path, 'rb');
    return pickle.load(f);


def util_dump(obj, path):
    f= open(path, 'wb');
    pickle.Pickler(f).dump(obj);
    f.close();
