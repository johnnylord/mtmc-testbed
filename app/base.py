import time
import socket
import logging
import threading
from multiprocessing import Process, Queue
from abc import ABC, abstractmethod

import cv2
import numpy as np

from network import NetworkAgent
from .gui.container import Container, check_ready
from .gui.media import MediaType


logger = logging.getLogger(__name__)


class Keyboard:
    """Helper class to refer keyboard code"""
    ESC = 27
    ENTER = 13
    SPACE = 32


class App(ABC, Container, NetworkAgent):
    """Abstract class for all kinds of app

    Arguments:
        ip (str, optional): server ip
        port (int): server port
        winname (str): gui window name
        barname (str): trackbar name
        trans_resolution (tuple): transmitted video resolution over network
    """
    SELECT_MODE = 1
    OPERATION_MODE = 2

    def __init__(self,
                ip="localhost",
                port=6666,
                winname="Application",
                barname="Frame",
                trans_resolution=(512, 512),
                **kwargs):
        # App metadata
        self.ip = ip
        self.port = port
        self.winname = winname
        self.barname = barname
        self.trans_resolution = trans_resolution
        self.mode = App.SELECT_MODE

        # Establish connection to server
        app_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        app_socket.connect((ip, port))

        # Explicit initialize parent classes
        NetworkAgent.__init__(self, conn=app_socket, addr=(ip, port))
        Container.__init__(self, **kwargs)

        # Receive server video channel info
        response = self.recv()
        self.channel_server_ip = response['content']['ip']
        self.channel_server_port = response['content']['port']

        # Assign value laters
        self.panel_to_channel = {} # panel -> { 'proc': channel, 'queue': queue }

    @abstractmethod
    def boot(self):
        """Prepare runtime environment for worker"""
        pass

    @abstractmethod
    def run(self):
        """Enter app loop"""
        pass

    @abstractmethod
    def keyboaord_handler(self, key):
        """Common key handlers"""
        if key == ord('q'):
            self.stop()

        elif key == Keyboard.SPACE:
            if self.is_start(): self.pause()
            elif self.is_pause(): self.start()

    @abstractmethod
    def mouse_callback(self, event, x, y, flags, param):
        """Select panel to be focused"""
        if event == cv2.EVENT_LBUTTONDOWN:
            panel = self.find_panel((x, y))
            if panel is None:
                return

            logger.info(f"Use has clicked {panel}")
            panel.focus = not panel.focus

            self.focus_panel = panel if panel.focus else None
            self.mode = App.OPERATION_MODE if panel.focus else App.SELECT_MODE

    @abstractmethod
    def trackbar_callback(self, value):
        self.jump(value)

    def close(self):
        # Release video sources
        Container.stop(self)

        # Release application socket
        NetworkAgent.close(self)

        # Kill all channel processes
        for panel, vchannel in self.panel_to_channel.items():
            vchannel['proc'].terminate()
            vchannel['proc'].channel.close()
            vchannel['queue'].close()

    def ready(self):
        # Establish video channels
        for panel in self.panels:
            share_queue = Queue(maxsize=-1)
            process = VideoChannel(ip=self.channel_server_ip,
                                port=self.channel_server_port,
                                queue=share_queue)
            process.start()
            self.panel_to_channel[panel] = { 'proc': process, 'queue': share_queue }

        # Setup GUI environment
        Container.ready(self)
        cv2.namedWindow(self.winname, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.winname, self.mouse_callback)
        if self.stype == MediaType.VIDEO:
            cv2.createTrackbar(self.barname, self.winname,
                                0, self.total_frames,
                                self.trackbar_callback)
        logger.info(f"Ready to run {self.__class__.__name__}")

    def parallel_send_videos(self, videos):
        """Parallely send videos through their own dedicated channels"""
        for video in videos:
            target_panel = video['panel']
            frame_bytes = video['frame_bytes']
            payload = { 'pid': target_panel.pid, 'frame_bytes': frame_bytes }
            vchannel = self.panel_to_channel[target_panel]
            vchannel['queue'].put(payload)


class VideoChannel(Process):
    """Independent process to send video frame to server

    Arguments:
        ip (str): channel server ip
        port (int): channel server port
        queue (Queue): shared queue between parent and child process
    """
    def __init__(self, ip, port, queue):
        super().__init__()
        self.ip = ip
        self.port = port
        self.queue = queue

        # Establish channel
        channel_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        channel_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        channel_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        channel_socket.connect((ip, port))
        self.channel = NetworkAgent(conn=channel_socket, addr=channel_socket.getsockname())

    def run(self):
        while True:
            if self.queue.qsize() > 0:
                packet = self.queue.get()
                self.channel.send(packet)
            else:
                time.sleep(0.01)
