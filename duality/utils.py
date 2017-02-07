import pickle
import socket
from .context import PORT


def util_load(path):
    f= open(path, 'rb');
    return pickle.load(f);


def util_dump(obj, path):
    f= open(path, 'wb');
    pickle.Pickler(f).dump(obj);
    f.close();


# data should be in a byte array format
# addr should be a string that represents the target address
def push_bev(addr, data):
    tcp_skt=socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    tcp_skt.connect((addr, PORT));
    tcp_skt.send(data);
    tcp_skt.close();
