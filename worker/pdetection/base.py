import logging
import numpy as np
from abc import ABC, abstractmethod

from ..utils.time import timeit

logger = logging.getLogger(__name__)


class PoseDetector(ABC):

    def __init__(self,
                stride=8,
                hthreshold=0.1,
                pthreshold=0.05,
                pretrain_model=None,
                device="cpu",
                **kwargs):
        self.stride = stride
        self.hthreshold = hthreshold
        self.pthreshold = pthreshold
        self.pretrain_model = pretrain_model
        self.device = device
        self.model = None

    @timeit(logger)
    def __call__(self, imgs):
        self._check_input(imgs)
        data = self.preprocessing(imgs)

        pafs, heatmaps = self.model(data)

        peoples = self.postprocessing(imgs, heatmaps, pafs)
        self._check_output(peoples)

        return peoples

    def _check_input(self, imgs):
        assert type(imgs) == list
        # img is numpy array
        for img in imgs:
            assert type(img) == np.ndarray

        # all imgs with same size
        size = tuple(imgs[0].shape)
        for img in imgs:
            assert tuple(img.shape) == size

    def _check_output(self, peoples):
        assert type(peoples) == list
        for people in peoples:
            assert type(people) == list
            for person in people:
                assert 'conf' in person
                assert 'bbox' in person
                assert 'n_parts' in person
                assert 'keypoints' in person

    @abstractmethod
    def preprocessing(self, imgs):
        pass

    @abstractmethod
    def postprocessing(self, imgs, heatmaps, pafs):
        pass
