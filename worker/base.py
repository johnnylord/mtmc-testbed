import socket
import logging
import threading

from multiprocessing import Process
from abc import ABC, abstractmethod, abstractproperty

import cv2

from network import NetworkAgent


logger = logging.getLogger(__name__)


class Worker(ABC, Process, NetworkAgent):
    """Abstract class for all worker class

    Arguments:
        conn (socket): socket for communicating with remote client
        addr (tuple): ip and port information of remote client
        device (str): device that worker can use
    """
    def __init__(self, conn, addr, device="cpu", **kwargs):
        super().__init__(**kwargs)
        NetworkAgent.__init__(self, conn, addr)

        # Assign values later
        self.channel_to_queue = {} # channel -> share_queue
        self.device = device
        self.config = None

    @abstractmethod
    def boot(self, config):
        """Prepare runtime environment for worker"""
        pass

    @abstractmethod
    def run(self):
        pass

    def close(self):
        # Close socket
        NetworkAgent.close(self)

    def parallel_recv_videos(self):
        videos = []
        for channel, share_queue in self.channel_to_queue.items():
            video = share_queue.get()
            videos.append(video)

        return videos
