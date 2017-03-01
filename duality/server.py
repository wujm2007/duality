import socketserver
import queue
import sys
import socket
import json
import threading
from http import server as http_server, HTTPStatus

from .gen_model import generate
from .utils import push_bev

INTERVAL = 1
HOST, TCP_PORT, UDP_PORT = "localhost", 4000, 13000

command_q = queue.Queue()
comment_q = queue.Queue()
# init the client ip address pool, a mechanism to maintain the client pool should be implemented
client_pool = set()


def produce():
    while True:
        if not command_q.empty():
            send_all(command_q.get())
        else:
            pass
            # send_all(b'0.0'); # which sends a default value


# which will be a multi-thread one in future
def send_all(data):
    for addr in client_pool:
        push_bev(addr, data)
    return


def consume():
    while not comment_q.empty():
        command_q.put(generate(comment_q.get()))


class POSTHandler(http_server.BaseHTTPRequestHandler):
    """a simple HTTP server that only serves POST request with a JSON content"""

    def do_POST(self):
        """Serve a POST request."""
        try:
            length = int(self.headers.get('content-length'))
        except:
            print('No or incorrect content-len gth found.', file=sys.stderr)
        self.data = self.rfile.read(length)
        self.send_head()
        comment_q.put(self.data)
        client_pool.add(self.address_string())
        # you should always send the new bev data to all the client in the pool, where a server has been started

    def send_head(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """a mix-in class that mix threading in a TCPServer"""
    pass


def timer_send_datagrams():
    """use a timer to send datagram at some interval"""
    t = threading.Timer(INTERVAL, send_datagrams)
    t.start()


def udp_send(ip, port, message):
    """send message to ip:port over UDP"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(bytes(message, "utf-8"), (ip, port))
        # response = str(sock.recv(1024), "utf-8")
        # print("Received: {}".format(response))


def send_datagrams():
    """send all comment in current queue to all clients in client_pool over UDP"""
    comment_list = list()
    if not comment_q.empty():
        # extract the comments from the queue and append them to a list
        while not comment_q.empty():
            comment_str = str(comment_q.get(), "utf-8")
            comment_list.append(eval(comment_str))

        # data to be send NEED TO BE MODIFIED
        data = {"action_id": "null", "voice_id": "null", "option": "null", "comments": comment_list}
        # encode data to json format
        json_data = json.JSONEncoder().encode(data)
        # send to all clients in client_pool
        for host in client_pool:
            udp_send(host, UDP_PORT, json_data)
    timer_send_datagrams()


if __name__ == "__main__":
    timer_send_datagrams()

    with ThreadingHTTPServer((HOST, TCP_PORT), POSTHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()
