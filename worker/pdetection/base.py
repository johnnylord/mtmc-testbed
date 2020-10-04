import logging
import numpy as np
from abc import ABC, abstractmethod

from ..utils.time import timeit

logger = logging.getLogger(__name__)


class PoseDetector(ABC):

    def __init__(self,
                stride=8,
                thre1=0.1,
                thre2=0.05,
                pretrain_model=None,
                device="cpu",
                **kwargs):
        self.stride = stride
        self.thre1 = thre1
        self.thre2 = thre2
        self.pretrain_model = pretrain_model
        self.device = device
        self.model = None

    @timeit(logger)
    def __call__(self, imgs):
        self._check_input(imgs)
        data = self.preprocessing(imgs)

        pafs, heatmaps = self.model(data)

        candidates, subsets = self.postprocessing(imgs, heatmaps, pafs)
        self._check_output(candidates, subsets)

        return candidates, subsets

    def _check_input(self, imgs):
        assert type(imgs) == list
        # img is numpy array
        for img in imgs:
            assert type(img) == np.ndarray

        # all imgs with same size
        size = tuple(imgs[0].shape)
        for img in imgs:
            assert tuple(img.shape) == size

    def _check_output(self, candiates, subsets):
        assert type(candidates) == list
        assert type(subsets) == list

    @abstractmethod
    def preprocessing(self, imgs):
        pass

    @abstractmethod
    def postprocessing(self, output, imgs):
        pass
