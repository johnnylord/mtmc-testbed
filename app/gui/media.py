import re
import time
import queue
import threading

import cv2
import numpy as np


__all__ = [ "MediaPlayer", "MediaType" ]


class MediaType:
    """Helper class for referring meida type"""
    VIDEO = 1
    STREAM = 2


class MediaPlayer:
    """General media player

    Arguments:
        src (str or int): opencv video source
        queue_size (int, optional): size of frame buffering queue, default 64
    """
    STATE_START = 1
    STATE_PAUSE = 2
    STATE_STOP = 3

    def __init__(self, src, queue_size=1):
        # Opencv capture
        # =====================================================
        self.capture = cv2.VideoCapture(src if not src.isdecimal() else int(src))
        self.capture_lock = threading.Lock()

        if not self.capture.isOpened():
            raise RuntimeError("Cannot open camera source '{}'".format(src))

        # Player metadata
        # =====================================================
        src = str(src)
        self.src = src
        self.queue_size = queue_size

        # Check source type
        if src.startswith("http") or src.startswith("rtsp") or src.isdecimal():
            self.stype = MediaType.STREAM
            self.frame_queue = queue.Queue(maxsize=1)
        else:
            self.stype = MediaType.VIDEO
            self.frame_queue = queue.Queue(maxsize=queue_size)

        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Player state
        # =====================================================
        self.state = MediaPlayer.STATE_PAUSE
        self.prev_frame = np.zeros((1, 1, 3))
        self.curr_frame = np.zeros((1, 1, 3))
        self.fid = -1

        # Buffering thread
        # =====================================================
        self._thread = threading.Thread(target=self._buffering, daemon=True)
        self._thread.start()

    def start(self):
        self.state = MediaPlayer.STATE_START
        return self

    def pause(self):
        self.state = MediaPlayer.STATE_PAUSE
        return self

    def stop(self):
        self.state = MediaPlayer.STATE_STOP

        # Critical sections
        with self.capture_lock:
            self.capture.release()
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        return self

    def jump(self, index):
        """Move frame pointer to specified point (index)"""
        if self.state == MediaPlayer.STATE_PAUSE:
            self.fid = index

            # Move frame pointer & Critical section
            with self.capture_lock:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                _, frame = self.capture.read()
                if frame is not None:
                    self.prev_frame = frame

            # For previewing jumped frame
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
                self.frame_queue.queue.append(frame)

    def read(self):
        """Return next processing frame"""
        # Read next frame from frame queue
        if self.state == MediaPlayer.STATE_START:
            self.fid += 1
            self.prev_frame = self.curr_frame
            try:
                self.curr_frame = self.frame_queue.get_nowait()
            except:
                self.curr_frame = self.prev_frame
            ret_frame = self.curr_frame.copy()

        elif self.state == MediaPlayer.STATE_PAUSE:
            ret_frame = self.prev_frame.copy()

        else:
            raise RuntimeError("You cannot fetch frame from terminated player")

        return self.fid, ret_frame

    def _buffering(self):
        """Buffering video frames into frame queue"""
        while self.state != MediaPlayer.STATE_STOP:
            # Fetch new frame and buffer it
            if not self.frame_queue.full():
                # Critical section
                with self.capture_lock:
                    ret, frame = self.capture.read()
                if not ret:
                    continue
                self.frame_queue.put(frame)

            # Slow down buffering task
            else:
                time.sleep(0.01)
