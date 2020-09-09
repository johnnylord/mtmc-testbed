import sys

from .reid import Resnet18


def get_recognizer(model_name, model_config={}):
    model_cls = vars(sys.modules[__name__])[model_name]
    recognizer = model_cls(**model_config)
    return recognizer
