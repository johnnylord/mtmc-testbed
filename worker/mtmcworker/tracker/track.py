import scipy
import numpy as np

from .kalman import KalmanFilter, chi2inv95
from .utils import xyah_to_tlbr, tlbr_to_xyah, compute_iou


__all__ = [ "DeepTrack" ]


class BaseTrack:
    """Base class for all kind of track class

    State transition machine:

                              [hit]
                            +------+
                            v      |
    3 continous at begin  +--------+-+    5 continuous [miss]
            +------------>| Tracked  +-------------+
            |             +----------+             v   +----+
       +----+-----+             ^             +--------++   |
    -->| Tentative|             +-------------+ Lost    |  [miss]
       +----+-----+                  1 [hit]  +----+----+   |
            |                                      |   ^    |
            |             +----------+             |   +----+
            +------------>+ Inactive |<------------+
            1 [miss]      +----------+      exceed miss_quota [miss]

    Attrs:
        id (int): track id number
        state (string): state of the track (see above transition graph)
        _hit_count (int): number of track hits from the existence of the track
        _miss_count (int): number of track miss from the existence of the track
        _miss_quota (int): quota of track miss before the track become inactive
    """
    MISS_QUOTA = 60
    ACTIVE_THRESHOLD = 3
    CONFIRM_THRESHOLD = 5

    def __init__(self, tid, **kwargs):
        self.tid = tid
        self.state = "tentative"

        self._hit_count = 1
        self._miss_count = 0
        self._recent_actions = []
        self._miss_quota = BaseTrack.MISS_QUOTA

    def __str__(self):
        return "{}({}):'{}'".format(self.__class__.__name__, self.tid, self.state)

    def __repr__(self):
        return self.__str__()

    def hit(self):
        # Update metadata of the track itself
        self._hit_count += 1
        self._recent_actions.append("hit")
        if len(self._recent_actions) > BaseTrack.CONFIRM_THRESHOLD:
            self._recent_actions = self._recent_actions[1:]

        # Update the state of the track
        if (
            self.state == "tentative"
            and self._hit_count >= BaseTrack.ACTIVE_THRESHOLD
        ):
            self.state = "tracked"

        elif (
            self.state == "lost"
            and len([ action
                    for action in self._recent_actions
                    if action == "hit" ]) >= 1
        ):
            self.state = "tracked"
        else:
            self.state = self.state

        self._miss_quota = BaseTrack.MISS_QUOTA

    def miss(self):
        # Update metadata of the track itself
        self._miss_count += 1
        self._miss_quota -= 1
        self._recent_actions.append("miss")
        if len(self._recent_actions) > BaseTrack.CONFIRM_THRESHOLD:
            self._recent_actions = self._recent_actions[1:]

        # Update the state of the track
        if self.state == "tentative":
            self.state = "inactive"

        elif (
            self.state == "tracked"
            and len([ action
                    for action in self._recent_actions
                    if action == "miss" ]) >= BaseTrack.CONFIRM_THRESHOLD
        ):
            self.state = "lost"

        elif (
            self.state == "lost"
            and self._miss_quota <= 0
        ):
            self.state = "inactive"

    def predict(self, *args, **kwargs):
        """Predict next state of the track"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Update internal state of the track"""
        raise NotImplementedError


class DeepTrack(BaseTrack):

    POOL_SIZE = 100

    def __init__(self, bbox, embedding, **kwargs):
        super().__init__(**kwargs)
        # Spatial information
        xyah = tlbr_to_xyah(bbox)
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(xyah)

        # Appearance information
        self.feature_pool = [ embedding ]

    @property
    def tlbr(self):
        return xyah_to_tlbr(self.mean.tolist()[:4])

    @property
    def bbox(self):
        return xyah_to_tlbr(self.mean.tolist()[:4])

    @property
    def velocity(self):
        return self.mean.tolist()[4:4+2]

    @property
    def feature(self):
        return np.mean(np.array(self.feature_pool), axis=0)

    def predict(self, hold_covariance=False):
        if not hold_covariance:
            self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        else:
            self.mean, _ = self.kf.predict(self.mean, self.covariance)

        return self.mean, self.covariance

    def update(self, bbox):
        xyah = tlbr_to_xyah(bbox)
        self.mean, self.covariance = self.kf.update(mean=self.mean,
                                                    covariance=self.covariance,
                                                    measurement=xyah)
        return self.mean, self.covariance

    def add_feature(self, feature):
        if len(self.feature_pool) < DeepTrack.POOL_SIZE:
            self.feature_pool.append(feature)
        else:
            self.feature_pool.pop(0)
            self.feature_pool.append(feature)

        return self

    def iou(self, bboxes):
        tlbr = xyah_to_tlbr(self.mean.tolist()[:4])
        ious = np.array([ compute_iou(tlbr, bbox) for bbox in bboxes ])
        return ious

    def mahalanobis_distance(self, bboxes, only_position=False):
        xyahs = np.array([ tlbr_to_xyah(bbox) for bbox in bboxes ])
        mean, covariance = self.kf._project(self.mean, self.covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            xyahs = xyahs[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = np.array(xyahs) - mean
        z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
        squared_maha = np.sum(z*z, axis=0)

        return squared_maha.tolist()

    def cosine_similarity(self, features):
        cosines = features.dot(np.array(self.feature_pool).T)
        cosines = 1. - np.mean(cosines, axis=1)

        return cosines.tolist()
