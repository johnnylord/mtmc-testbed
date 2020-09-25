import logging
import numpy as np
from abc import ABC, abstractmethod

from ..utils.time import timeit

logger = logging.getLogger(__name__)


class PersonDetector(ABC):

    def __init__(self,
                min_height=100,
                min_confidence=0.6,
                pretrain_model=None,
                device="cpu",
                **kwargs):
        self.min_height = min_height
        self.min_confidence = min_confidence
        self.pretrain_model = pretrain_model
        self.device = device
        self.model = None

    @timeit(logger)
    def __call__(self, imgs):
        self._check_input(imgs)
        input_ = self.preprocessing(imgs)

        output = self.model(input_)

        output = self.postprocessing(output, imgs)
        self._check_output(output)

        return output

    def _check_input(self, input_):
        assert type(input_) == list
        for img in input_:
            assert type(img) == np.ndarray

    def _check_output(self, output):
        assert type(output) == list
        for result in output:
            assert type(result) == list
            for bbox in result:
                assert type(bbox) == dict
                assert 'conf' in bbox
                assert 'bbox' in bbox

    @abstractmethod
    def preprocessing(self, imgs):
        pass

    @abstractmethod
    def postprocessing(self, output, imgs):
        pass
