# For unpickling yolov5 pretrain weights
import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np
import torch
from torch.hub import download_url_to_file

from ..base import PersonDetector
from .models.experimental import Ensemble, attempt_load
from .models.utils import check_img_size, non_max_suppression


__all__ = [ "YOLOv5" ]

class YOLOv5(PersonDetector):

    # PRETRAIN_URL = 'https://www.dropbox.com/s/v22et6uin4jjtel/best_5x_640.pt?dl=1'
    # PRETRAIN_URL = 'https://www.dropbox.com/s/hjbg6xqqrhhym39/yolov5x.pt?dl=1'
    PRETRAIN_URL = 'https://www.dropbox.com/s/vjbv9uzqd81t41v/yolov5m.pt?dl=1'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load pretrained model
        if self.pretrain_model is not None:
            self.model = torch.load(self.pretrain_model)['model']

        # Load pretained model from url
        else:
            # Check cache directory
            cache_dir = osp.expanduser("~/.cache/torch/checkpoints/")
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)

            # Cache file name
            cache_file = osp.basename(YOLOv5.PRETRAIN_URL).split("?")[0]
            if not osp.exists(osp.join(cache_dir, cache_file)):
                download_url_to_file(YOLOv5.PRETRAIN_URL, osp.join(cache_dir, cache_file))

            self.model = torch.load(osp.join(cache_dir, cache_file))['model']

        self.model.to(self.device)
        self.model.eval()

        if "cuda" in self.device:
            self.model = self.model.half()

    def preprocessing(self, imgs):
        inputs = []
        for img in imgs:
            normalized = torch.from_numpy(img / 255.0)
            normalized = normalized.permute(2, 0, 1)
            normalized = normalized.to(self.device)
            normalized = normalized.half() if "cuda" in self.device else normalized.float()
            inputs.append(normalized)

        return torch.stack(inputs)

    def postprocessing(self, output):
        output = non_max_suppression(output[0], conf_thres=0.2, iou_thres=0.6, classes=0)

        results = []
        for result in output:
            if result is None:
                results.append([])
                continue

            boxes = result[:,:4].detach().cpu().numpy().tolist()
            labels = result[:,5].detach().cpu().numpy().tolist()
            scores = result[:,4].detach().cpu().numpy().tolist()

            # Filter out invalid bboxes
            target_boxes = []
            for box, label, score in zip(boxes, labels, scores):
                box_height = int(box[3]-box[1])
                if (
                    label == 0 # person class
                    and score > self.min_confidence
                    and box_height > self.min_height
                    ):
                    target_boxes.append({ 'conf': score, 'bbox': box })

            results.append(target_boxes)

        return results
