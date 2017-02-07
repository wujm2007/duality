import socketserver
import time
import queue
import threading
import socket
from .gen_model import generate
from .utils import push_bev



command_q = queue.Queue();
comment_q = queue.Queue();
# init the client ip address pool, a mechanism to maintain the client pool should be implemented
client_pool= set();

def produce():
    while(True):
        if(not command_q.empty()):
            send_all(command_q.get());
        else:
            send_all(b'0.0'); # which sends a default value



# which will be a multi-thread one in future
def send_all(data):
    for addr in client_pool:
        push_bev(addr, data);
    return;


def consume():
    while(not comment_q.empty()):
        comment_q.put(generate(command_q.get()));


class MyTCPHandler(socketserver.BaseRequestHandler):
    def setup(self):
        self.get_request=None;
        self.consumer=threading.Thread(target=consume, name='consumer');
        self.consumer.start();
        self.producer=threading.Thread(target=produce, name='producer');
        self.producer.start();

    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print(self.data);
        comment_q.put(self.data);
        # just send back the same data, but upper-cased
        client_pool.add(self.request.getpeername()[0]);
        # you should always send the new bev data to all the client in the pool, where a server has been started



if __name__ == "__main__":
    HOST, PORT = "localhost", 4000

    # Create the server, binding to localhost on port 4000
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
