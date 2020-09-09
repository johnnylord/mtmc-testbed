import logging
import signal
import socket
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from multiprocessing import Event

from network import NetworkAgent
from worker import LazyWorker

# Logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)
s_format = logging.Formatter('%(levelname)s - PID[%(process)d] - %(message)s')
s_handler.setFormatter(s_format)

f_handler = logging.FileHandler("log-server.txt")
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

logger.addHandler(s_handler)
logger.addHandler(f_handler)

# Commandline parser
parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="0.0.0.0", type=str, help="server ip")
parser.add_argument("--port", default=6666, type=int, help="server port")

# Ctrl+C handler
def signal_handler(signum, frame):
    raise Exception("Ctrl+C is triggered")

signal.signal(signal.SIGINT, signal_handler)

def main(args):
    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    # Bind server socket
    server_socket.bind((args['ip'], args['port']))
    server_socket.listen(10)
    logger.info("Server launch at {}:{}".format(args['ip'], args['port']))

    # Handling incoming connection request
    workers = []
    try:
        while True:
            conn, addr = server_socket.accept()
            logger.info("Connection from {}:{}".format(addr[0], addr[1]))

            # Create new worker process for handling new client
            shutdown_event = Event()
            worker = LazyWorker(conn=conn,
                                addr=addr,
                                device="cuda",
                                shutdown_event=shutdown_event)
            worker.start()
            workers.append((worker, shutdown_event))

            # Parent process release socket
            conn.close()

    except Exception as e:
        # Shutdown child process gracefully
        for worker, event in workers:
            event.set()

        # Shutdown server socket
        server_socket.close()
        logger.info("Shutdown server")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)
