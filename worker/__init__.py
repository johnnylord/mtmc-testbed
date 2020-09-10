import os
import sys
import socket
import logging
import threading
from multiprocessing import Process, Event, Queue

import cv2
import numpy as np

from network import NetworkAgent

from .base import Worker
from .echoworker import EchoWorker
from .detworker import DetWorker
from .motworker import MOTWorker

logger = logging.getLogger(__name__)


class LazyWorker(Worker):
    """An unknown worker in lazy state

    Every worker process will be first in this state before turn itself into
    true worker. The remote connected client will tell the worker to turn what
    the client wants.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def boot(self, config):
        raise RuntimeError("You cannot boot LazyWorker")

    def close(self, config):
        raise RuntimeError("You cannot close LazyWorker")

    def run(self):
        """Communicate with remote client and cast itself to target worker"""
        self._spawn_channel_server()

        try:
            # Cast to target worker
            msg = self.recv()
            if msg is None:
                raise Exception("Remote client has been closed")

            worker_name = msg['content']
            worker_cls = vars(sys.modules[__name__])[worker_name]
            self.__class__ = worker_cls
            logger.info(f"Casting lazyworker to {worker_cls.__name__}")

            # Send back worker default configuration
            worker_config = worker_cls.DEFAULT_CONFIG
            message = { 'action': 'config', 'content': worker_config }
            self.send(message)

            # Use modified worker config from remote client to build runtime
            response = self.recv()
            if response is None:
                raise Exception("Remote client has been closed")

            worker_config = response['content']
            logger.info(f"Configure worker with {worker_config}")

        except Exception as e:
            logger.info("Remote client has been closed")
            self.channel_server.close()
            NetworkAgent.close(self)

            # Shutdown video channel gracefully
            for channel, shares in self.channel_to_shares.items():
                shares[0].close() # close queue
                shares[1].set() # set shutdown event flag
                channel.join()  # wait for child process shutdown

            return

        self.boot(worker_config)
        logger.info(f"Run {worker_cls.__name__}, pid: {os.getpid()}")
        self.run()
        logger.info(f"Shutdown {worker_cls.__name__}, pid: {os.getpid()}")

        # Close socket connections
        self.channel_server.close()
        NetworkAgent.close(self)

        # Shutdown video channel gracefully
        for channel, shares in self.channel_to_shares.items():
            shares[0].close() # close queue
            shares[1].set() # set shutdown event flag
            channel.join()  # wait for child process shutdown

    def _spawn_channel_server(self):
        # Create server socket for listening channel sockets
        self.channel_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.channel_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.channel_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        # Bind channel server socket
        self.channel_server.bind((self._conn.getsockname()[0], 0))
        self.channel_server.listen(10)
        self._channel_thread = threading.Thread(target=self._accept_channel, daemon=True)
        self._channel_thread.start()

        # Tell remote client where to establish video channels
        channel_server_info = {
                'ip': self.channel_server.getsockname()[0],
                'port': self.channel_server.getsockname()[1]
                }
        self.send({ 'action': 'info', 'content': channel_server_info })

        # Logging
        channel_ip = self.channel_server.getsockname()[0]
        channel_port = self.channel_server.getsockname()[1]
        logger.info("Start channel server at {}:{}".format(channel_ip, channel_port))

    def _accept_channel(self):
        """Accept video channel sockets from client"""
        while True:
            conn, addr = self.channel_server.accept()

            # Spawn process to recv videos
            shutdown_event = Event()
            share_queue = Queue(maxsize=-1)
            channel = VideoChannel(conn, addr, share_queue, shutdown_event)
            channel.start()
            self.channel_to_shares[channel] = (share_queue, shutdown_event)
            logger.info("Establish new video channel from {}:{}".format(addr[0], addr[1]))

            # Parent process close socket
            conn.close()


class VideoChannel(Process):
    """Independent process to recv video frame from client

    Arguments:
        conn (str): channel socket
        addr (int): remote client address
        queue (Queue): shared queue between parent and child process
        shutdown_event (Event): shutdown flag
    """
    def __init__(self, conn, addr, queue, shutdown_event):
        super().__init__()
        self.channel = NetworkAgent(conn, addr)
        self.queue = queue
        self.shutdown_event = shutdown_event

    def run(self):
        while not self.shutdown_event.is_set():
            data = self.channel.recv()
            if data is None:
                break

            # Decode incoming packets
            target_pid = data['pid']
            target_frame = cv2.imdecode(data['frame_bytes'], cv2.IMREAD_COLOR)
            target_frame = target_frame.astype(float)
            self.queue.put({ 'pid': target_pid, 'frame': target_frame })

        # Close connection
        self.channel.close()
        logger.info("Shutdown video channel gracefully")
