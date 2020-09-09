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
    def __init__(self, conn, addr, device, shutdown_event=None, **kwargs):
        super().__init__(**kwargs)
        NetworkAgent.__init__(self, conn, addr)
        self.device = device
        self.shutdown_event = shutdown_event

        # Assign values later
        self.channel_to_shares = {} # channel -> share_queue, shutdown_event
        self.config = None

    @abstractmethod
    def boot(self, config):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def parallel_recv_videos(self):
        videos = []
        for channel, shares in self.channel_to_shares.items():
            video = shares[0].get()
            videos.append(video)

        return videos
