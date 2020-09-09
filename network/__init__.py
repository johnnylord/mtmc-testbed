import socket
import pickle


__all__ = [ "NetworkAgent" ]

class NetworkAgent(object):
    """Base class for network agent

    It gives the derived class the ability to send data with TCP/IP protocal.

    Attributes:
        conn (socket.socket): socket connection
        addr (tuple): ip address, port
    """
    HEADER_SIZE = 10

    def __init__(self, conn, addr, **kwargs):
        self._conn = conn
        self._addr = addr

    def recv(self):
        """Read any python object from the connected endpoint

        Returns:
            If get something, return dict, otherwise, return None
        """
        raw_data = b''

        while True:
            raw_bytes = self._conn.recv(4096)

            # Socket connect lost
            if len(raw_bytes) == 0:
                return None

            if len(raw_data) == 0:
                payload_size = int(raw_bytes[:NetworkAgent.HEADER_SIZE])

            raw_data += raw_bytes
            if len(raw_data)-NetworkAgent.HEADER_SIZE == payload_size:
                data = pickle.loads(raw_data[NetworkAgent.HEADER_SIZE:])
                return data


    def send(self, data):
        """Send python object to the connected endpoint"""
        raw_data = pickle.dumps(data)
        header = bytes(f"{len(raw_data):<{NetworkAgent.HEADER_SIZE}}", "utf-8")
        raw_data = header + raw_data
        self._conn.sendall(raw_data)

    def close(self):
        """Close the socket connection"""
        self._conn.close()
