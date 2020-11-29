import logging

import cv2
import numpy as np

from ..utils.time import timeit
from ..base import Worker
from ..detection import get_detector
from ..recognition import get_recognizer

from .tracker import DeepTracker
from .tracker.track import DeepTrack
from .tracker.utils import compute_iou
from .cluster import AdaptiveKmeans
from .utils import crop_image


logger = logging.getLogger(__name__)


__all__ = [ "SOTWorker" ]

class SOTWorker(Worker):
    """Detect person object in the received frame from client """
    DEFAULT_CONFIG = {
        "detection_model": "YOLOv5",
        "recognition_model": "Resnet18",
    }

    def __init__(self):
        raise RuntimeError("You cannot directly instantiate MCMTWorker")

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
        self.target_label = -1

        # Construct tracker synchronizer
        self.kmeans = AdaptiveKmeans()

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
            logger.error(f"Error occur in detworker", exc_info=True)

        # Cleanup process
        self.close()

    def close(self):
        del self.detector
        del self.recognizer

    @timeit(logger)
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


        # STAGE_1: Detect people in videos
        # =========================================================
        detect_results = []
        results = self.detector(frames)
        for pid, bboxes in zip(pids, results):
            detect_results.append({ 'pid': pid, 'bboxes': bboxes })

        # STAGE_2: Recongize people in videos
        # ========================================================
        recognize_results = []
        for detect_result, frame in zip(detect_results, frames):
            pid = detect_result['pid']
            # Compute embedding in specific video
            people_imgs = []
            for bbox in detect_result['bboxes']:
                person_img = crop_image(frame, bbox['bbox'])
                people_imgs.append(person_img)
            embeddings = self.recognizer(people_imgs)
            recognize_results.append({ 'pid': pid, 'embeddings': embeddings })

        # STAGE_3: Perform association in videos
        # =======================================================
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

            # Perform association
            tracker.associate(measurments)

        # STAGE_4: Synchronize tracked trackers
        # =======================================================
        # Form groups of tracked embedding
        if len(frames) > 1:
            group_embeddings = []
            for tracker in self.trackers.values():
                embeddings = [  t['feature']
                                for t in tracker.tracks
                                if t['state'] == 'tracked'  ]
                if len(embeddings) > 0:
                    group_embeddings.append(np.array(embeddings))

            # Update tracked clusters
            if len(group_embeddings) > 0:
                n_clusters = np.max([ len(embeddings) for embeddings in group_embeddings ])
                self.kmeans.fit(group_embeddings, n_clusters=n_clusters)
            else:
                self.kmeans.miss()

        # STAGE_5: Remap track IDs
        # =======================================================
        group_tracks = {}
        for pid, tracker in self.trackers.items():
            tracks = tracker.tracks
            if len(frames) > 1:
                tracked_tracks = [ t for t in tracks if t['state'] == 'tracked' ]
                if len(tracked_tracks) > 0:
                    embeddings = np.array([ t['feature'] for t in tracked_tracks ])
                    tids = self.kmeans.predict(embeddings)
                    for t, tid in zip(tracked_tracks, tids):
                        t['tid'] = tid
                group_tracks[pid] = tracked_tracks

        # STAGE_6: Find chosen object
        # =======================================================
        if (
            self.target_label == -1
            and len(request['remote']['tlbr']) == 4
        ):
            pid = request['remote']['pid']
            tracks = group_tracks[pid]
            ious = [ compute_iou(t['bbox'], request['remote']['tlbr']) for t in tracks ]
            if max(ious) > 0.6:
                idx = np.argmax(ious)
                self.target_label = tracks[idx]['tid']
        else:
            n_targets = 0
            for tracks in group_tracks.values():
                if len([ t for t in tracks if t['tid'] == self.target_label ]) > 0:
                    n_targets += 1
            if n_targets == 0:
                self.target_label = -1

        # STAGE_5: Send back result to client
        # =======================================================
        video_results = []
        for pid, tracks in group_tracks.items():
            # Remove feature vectors to reduce transmission size
            for t in tracks:
                del t['feature']

            if self.target_label >= 0:
                target_track = [ t for t in tracks if t['tid'] == self.target_label ]
            else:
                target_track = []

            video_results.append({ 'pid': pid, 'tracks': target_track })

        response['content'] = video_results
        return response

    def _reset_handler(self, request):
        self.target_label = -1
        response = { 'action': 'reset', 'content': [] }
        return response

    def _nop_handler(self, request):
        response = { 'action': 'nop', 'content': [] }
        return response
