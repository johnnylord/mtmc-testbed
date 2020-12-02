import signal
import socket
import random
import logging
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from multiprocessing import Event

import GPUtil
import torch

from network import NetworkAgent
from worker import LazyWorker


class LogFilter(object):

    def __init__(self, level):
        self._level = level

    def filter(self, logRecord):
        return logRecord.levelno == self._level


# Logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Stream Handler
s_handler = logging.StreamHandler()
s_handler.setLevel(logging.INFO)
s_format = logging.Formatter('%(asctime)s, %(levelname)s, PID[%(process)d] %(name)s, %(message)s')
s_handler.setFormatter(s_format)

# Error file handler
f1_handler = logging.FileHandler("log-server-error.txt")
f1_handler.setLevel(logging.ERROR)
f1_handler.addFilter(LogFilter(logging.ERROR))
f1_format = logging.Formatter('%(asctime)s, PID[%(process)d], %(name)s, %(message)s')
f1_handler.setFormatter(f1_format)

# Info file handler
f2_handler = logging.FileHandler("log-server-info.txt")
f2_handler.setLevel(logging.INFO)
f2_handler.addFilter(LogFilter(logging.INFO))
f2_format = logging.Formatter('%(asctime)s, PID[%(process)d], %(name)s, %(message)s')
f2_handler.setFormatter(f2_format)

# Register handler on root logger
logger.addHandler(s_handler)
logger.addHandler(f1_handler)
logger.addHandler(f2_handler)

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

            if torch.cuda.is_available():
                n_devices = torch.cuda.device_count()
                deviceIDs = GPUtil.getAvailable(order='memory', limit=n_devices)
                random.shuffle(deviceIDs)
                device = "cuda:{}".format(deviceIDs[0])
            else:
                device = "cpu"

            # Create new worker process for handling new client
            shutdown_event = Event()
            worker = LazyWorker(conn=conn,
                                addr=addr,
                                device=device,
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
        logger.info("Shutdown server", exc_info=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = vars(parser.parse_args())
    main(args)
