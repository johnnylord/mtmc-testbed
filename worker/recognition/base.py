from abc import ABC, abstractmethod

import numpy as np
import torch


class PersonRecognizer(ABC):

    def __init__(self, size=(128, 256), pretrain_model=None, device="cpu", **kwargs):
        self.size = size
        self.pretrain_model = pretrain_model
        self.device = device
        self.model = None

    def __call__(self, imgs):
        if len(imgs) == 0:
            return np.ndarray([])

        self._check_input(imgs)
        input_ = self.preprocessing(imgs)

        output = self.model(input_)

        output = self.postprocessing(output)
        self._check_output(output)

        return output

    def _check_input(self, input_):
        assert type(input_) == list
        for img in input_:
            assert type(img) == np.ndarray

    def _check_output(self, output):
        assert type(output) == np.ndarray
        assert len(output.shape) == 2

    @abstractmethod
    def preprocessing(self, img):
        pass

    @abstractmethod
    def postprocessing(self, output):
        pass

