# Duality (the server part)
- `server.py`

including a server that serves only HTTP POST with JSON content and a UDP client.
- `dummy_client.py`

a client that sends dummy request over TCP (for testing)
- `dummy_udp_server.py`

a UDP server that receives the response (for testing)
## How to test
1. change host and port in corresponding files.
2. run `server.py` in computer A
3. run `dummy_udp_server.py` in computer B
4. run `dummy_client.py` in computer B. You should see the logs on both computer A and B.