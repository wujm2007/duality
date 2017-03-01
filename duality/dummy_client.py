import http.client
import json

HOST, PORT = "localhost", 4000
client = None
try:
    headers = {"Content-type": "application/json"}
    data = {"timestamp": 0, "comment": "what a comment!", "id": "some_user"}

    client = http.client.HTTPConnection(HOST, PORT)
    client.request("POST", "", json.JSONEncoder().encode(data), headers)
    response = client.getresponse()

    print("status", response.status)
    print("reason", response.reason)
    print("headers", response.getheaders())
    print("data", response.read())

finally:
    if client:
        client.close()
