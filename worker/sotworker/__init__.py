import logging

import cv2
import numpy as np

from ..base import Worker
from ..detection import get_detector
from ..recognition import get_recognizer

from .tracker import DeepTracker
from .tracker.track import DeepTrack
from .tracker.utils import compute_iou
from .utils import crop_image

logger = logging.getLogger(__name__)

__all__ = [ "SOTWorker" ]

class SOTWorker(Worker):
    """Detect person object in the received frame from client """
    DEFAULT_CONFIG = {
        "detection_model": "FasterRCNN",
        "recognition_model": "Resnet18",
    }

    def __init__(self):
        raise RuntimeError("You cannot directly instantiate MOTWorker")

    def boot(self, config):
        """Prepare environment for worker to run"""
        self.config = config

        # Construct detector
        model_name = config['detection_model']
        model_config = { 'device': self.device }
        self.detector = get_detector(model_name, model_config)
        logger.info("Construct detector")

        # Construct recognizer
        model_name = config['recognition_model']
        model_config = { 'device': self.device }
        self.recognizer = get_recognizer(model_name, model_config)
        logger.info("Construct recognizer")

        # Construct trackers (initialized when get videos from client)
        self.trackers = None

        # Construct event handlers
        self.event_handler = { 'track': self._track_handler,
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
            logger.error(f"Error occur in sotworker", exc_info=True)

        # Cleanup process
        self.close()

    def close(self):
        del self.detector
        del self.recognizer

    def _track_handler(self, request):
        
        response = { 'action': 'track', 'content': [] }
        # Parallel receive video
        videos = self.parallel_recv_videos()
        pids = [ video['pid'] for video in videos ]
        frames = [ video['frame'] for video in videos ]
        
        # Propagate trackers
        if self.trackers is None:
            self.trackers = dict([ (pid, DeepTracker()) for pid in pids ])

        for _, tracker in self.trackers.items():
            tracker.propagate()

        logger.info("Propagate tracker")
        

        # STAGE_1: Detect people in videos
        # =========================================================
        detect_results = []
        results = self.detector(frames)
        for pid, bboxes in zip(pids, results):
            detect_results.append({ 'pid': pid, 'bboxes': bboxes })
        
        

        logger.info("Perform detection")
        
        # STAGE_2: Recongize people in videos
        # ========================================================
        recognize_results = []
        for detect_result, frame in zip(detect_results, frames):
            pid = detect_result['pid']
            people_imgs = []
            for bbox in detect_result['bboxes']:
                person_img = crop_image(frame, bbox['bbox'])
                people_imgs.append(person_img)
            embeddings = self.recognizer(people_imgs)
            recognize_results.append({ 'pid': pid, 'embeddings': embeddings })
            
        logger.info("Perform recognition")

        # STAGE_3: Perform association in videos
        # =======================================================
        if request['sync']['anchor']:
            for measurment in zip(detect_results, recognize_results):
                detect_result = measurment[0]
                recognize_result = measurment[1]

                # Select associated tracker
                pid = detect_result['pid']
                tracker = self.trackers[pid]

                # Form measurements
                if pid == request['sync']['panel']:
                    bboxes = np.array([ bbox['bbox'] for bbox in detect_result['bboxes'] ])
                    embeddings = np.array(recognize_result['embeddings'])
                    choose_idx = np.argmax([compute_iou(bbox, request['sync']['tlbr']) for bbox in bboxes]) 
                    choose_embedding = embeddings[choose_idx]
                    measurments = np.concatenate([[bboxes[choose_idx]], [embeddings[choose_idx]]], axis=1)
                    tracker.associate(measurments)


            for measurment in zip(detect_results, recognize_results):
                detect_result = measurment[0]
                recognize_result = measurment[1]

                # Select associated tracker
                pid = detect_result['pid']
                tracker = self.trackers[pid]

                # Form measurements
                bboxes = np.array([ bbox['bbox'] for bbox in detect_result['bboxes'] ])
                embeddings = np.array(recognize_result['embeddings'])

                if len(bboxes) == 0:
                    continue

                if pid != request['sync']['panel']:

                    cosines = choose_embedding.dot(embeddings.T)
                    choose_idx = np.argmax(cosines)
                    measurments = np.concatenate([[bboxes[choose_idx]], [embeddings[choose_idx]]], axis=1)
                    # Perform association
                    tracker.associate(measurments)
        else:
            for measurment in zip(detect_results, recognize_results):
                detect_result = measurment[0]
                recognize_result = measurment[1]

                # Select associated tracker
                pid = detect_result['pid']
                tracker = self.trackers[pid]

                # Form measurements
                bboxes = np.array([ bbox['bbox'] for bbox in detect_result['bboxes'] ])
                embeddings = np.array(recognize_result['embeddings'])

                if len(bboxes) == 0:
                    continue

                measurments = np.concatenate([bboxes, embeddings], axis=1)
                tracker.associate(measurments)
            
        logger.info("Perform association")

        # Prepare result to send back to remote client
        video_results = []
        for pid, tracker in self.trackers.items():
            video_results.append({ 'pid': pid, 'tracks': tracker.tracks })
        response['content'] = video_results
        return response

    def _reset_handler(self, request):
        self.trackers = None
        response = { 'action': 'reset', 'content': [] }
        return response

    def _nop_handler(self, request):
        videos = self.parallel_recv_videos()
        response = { 'action': 'nop', 'content': [] }
        return response
