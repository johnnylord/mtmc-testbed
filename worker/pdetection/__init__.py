import sys

from .bodypose import BodyPoseDetector


def get_detector(model_name, model_config={}):
    model_cls = vars(sys.modules[__name__])[model_name]
    detector = model_cls(**model_config)
    return detector
