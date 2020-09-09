import logging

import cv2
import numpy as np

from ..base import Worker
from ..detection import get_detector


logger = logging.getLogger(__name__)

__all__ = [ "DetWorker" ]

class DetWorker(Worker):
    """Detect person object in the received frame from client """
    DEFAULT_CONFIG = {
        "detection_model": "YOLOv5",
    }

    def __init__(self):
        raise RuntimeError("You cannot directly instantiate EchoWorker")

    def boot(self, config):
        """Prepare environment for worker to run"""
        self.config = config

        # Constructor detector
        model_name = config['detection_model']
        model_config = { 'device': self.device }
        self.detector = get_detector(model_name, model_config)

        # Construct event handlers
        self.event_handler = { 'detect': self._detect_handler,
                                'reset': self._reset_handler,
                                'nop': self._nop_handler }

    def run(self):
        """Worker job"""
        try:
            while not self.shutdown_event.is_set():
                # Recieve message
                request = self.recv()
                if request is None:
                    break

                # Route request to associate event handler
                handler = self.event_handler[request['action']]
                response = handler(request)

                # Send back message
                self.send(response)

        except Exception as e:
            logger.error(f"Error occur in detworker", exc_info=True)

        # Cleanup process
        self.close()

    def close(self):
        del self.detector

    def _detect_handler(self, request):
        response = { 'action': 'detect', 'content': [] }

        # Parallel receive video
        videos = self.parallel_recv_videos()

        # Process panel by panel
        video_results = []
        pids = [ video['pid'] for video in videos ]
        frames = [ video['frame'] for video in videos ]
        results = self.detector(frames)

        for pid, bboxes in zip(pids, results):
            video_results.append({ 'pid': pid, 'bboxes': bboxes })

        response['content'] = video_results
        return response

    def _reset_handler(self, request):
        response = { 'action': 'reset', 'content': [] }
        return response

    def _nop_handler(self, request):
        response = { 'action': 'nop', 'content': [] }
        return response
