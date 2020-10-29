import numpy as np
from scipy.optimize import linear_sum_assignment

from .track import DeepTrack
from .kalman import chi2inv95
from .utils import tlbr_to_xyah


class DeepTracker:

    def __init__(self):
        self._tracks = []
        self._counter = 0

    @property
    def tracks(self):
        return [ {  'tid': t.tid,
                    'state': t.state,
                    'bbox': t.bbox,
                    'covar': t.covariance[:2, :2] }
                for t in self._tracks ]

    def propagate(self):
        for track in self._tracks:
            if track.state == "lost":
                track.predict(hold_covariance=True)
            else:
                track.predict()


    def associate(self, measurements):
        """Associate meansurements to tracks in state-wise fashion"""
        hit_tracks = []
        miss_tracks = []
        

        # Split tracks by their states
        tracked_tracks = [ t for t in self._tracks if t.state == "tracked" ]
        lost_tracks = [ t for t in self._tracks if t.state == "lost" ]
        tentative_tracks = [ t for t in self._tracks if t.state == "tentative" ]

        # STAGE_1: Associate with tracked tracks
        # =============================================================
        match_tindices, match_mindices = self._match(tracks=tracked_tracks,
                                                    measurements=measurements,
                                                    metric="cosine", threshold=0.7)
        hit_tracks += [ t for i, t in enumerate(tracked_tracks) if i in match_tindices ]
        miss_tracks += [ t for i, t in enumerate(tracked_tracks) if i not in match_tindices ]
        measurements = np.array([ m for i, m in enumerate(measurements) if i not in match_mindices ])

        # STAGE_2: Associate with lost tracks
        # =============================================================
        match_tindices, match_mindices = self._match(tracks=lost_tracks,
                                                    measurements=measurements,
                                                    metric="cosine", threshold=0.7)
        hit_tracks += [ t for i, t in enumerate(lost_tracks) if i in match_tindices ]
        miss_tracks += [ t for i, t in enumerate(lost_tracks) if i not in match_tindices ]
        measurements = np.array([ m for i, m in enumerate(measurements) if i not in match_mindices ])

        # STAGE_3: Associate with tentative tracks
        # =============================================================
        match_tindices, match_mindices = self._match(tracks=tentative_tracks,
                                                    measurements=measurements,
                                                    metric="iou", threshold=0.7)
        hit_tracks += [ t for i, t in enumerate(tentative_tracks) if i in match_tindices ]
        miss_tracks += [ t for i, t in enumerate(tentative_tracks) if i not in match_tindices ]
        measurements = np.array([ m for i, m in enumerate(measurements) if i not in match_mindices ])

        # STAGE_4: Remove dead tracks & Create new tracks
        # =================================================================
        _ = [ t.hit() for t in hit_tracks ]
        _ = [ t.miss() for t in miss_tracks ]
        self._tracks = [ t for t in self._tracks if t.state != "inactive" ]
        
        if self._counter < 1:     
            new_tracks = []
            for measurement in measurements:
                bbox = measurement[:4]
                embedding = measurement[4:]
                new_tracks.append(DeepTrack(bbox, embedding, tid=self._counter))
                self._counter += 1
                break
            self._tracks += new_tracks

    def _match(self, tracks, measurements, metric, threshold):
        # Edge cases
        if (
            (len(tracks) == 0 and len(measurements) == 0)
            or (len(tracks) == 0 and len(measurements) != 0)
            or (len(tracks) != 0 and len(measurements) == 0)
        ):
            return [], []

        # Compute cost matrix
        bboxes = measurements[:, :4]
        embeddings = measurements[:, 4:]

        if metric == 'iou':
            costs = 1 - np.array([ t.iou(bboxes) for t in tracks ])
        elif metric == 'cosine':
            dcosts = np.array([ t.mahalanobis_distance(bboxes, only_position=True) for t in tracks ])
            costs = np.array([ t.cosine_similarity(embeddings) for t in tracks ])
            costs[dcosts > chi2inv95[2]] = 10000.

        # Perform linear assignment
        tindices, mindices = linear_sum_assignment(costs)
        match_pairs = [ pair
                        for pair in zip(tindices, mindices)
                        if costs[pair[0], pair[1]] <= threshold ]

        # Update track state
        for tind, mind in match_pairs:
            track = tracks[tind]
            track.update(bboxes[mind])
            track.add_feature(embeddings[mind])
        # Return matched indice
        match_tindices = [ tind for tind, _ in match_pairs ]
        match_mindices = [ mind for _, mind in match_pairs ]
        return match_tindices, match_mindices
