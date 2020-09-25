import logging
import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from ...utils.time import timeit
from ..base import PersonDetector

logger = logging.getLogger(__name__)

__all__ = [ "FasterRCNN" ]

class FasterRCNN(PersonDetector):

    PRETRAIN_URL = "https://www.dropbox.com/s/yzp1taohl75coch/fasterrcnn_dance_010.pth?dl=1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load pretrained model
        self.model = fasterrcnn_resnet50_fpn(num_classes=91)
        if self.pretrain_model is not None:
            state_dict = torch.load(self.pretrain_model)
        else:
            state_dict = load_state_dict_from_url(FasterRCNN.PRETRAIN_URL)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @timeit(logger)
    def preprocessing(self, imgs):
        inputs = []
        for img in imgs:
            normalized = torch.from_numpy(img/255.)
            normalized = normalized.permute(2, 0, 1)
            normalized = normalized.float().to(self.device)
            inputs.append(normalized)

        return inputs

    @timeit(logger)
    def postprocessing(self, output):
        results = []
        for result in output:
            boxes = result['boxes'].detach().cpu().numpy().tolist()
            labels = result['labels'].detach().cpu().numpy().tolist()
            scores = result['scores'].detach().cpu().numpy().tolist()

            # Filter out invalid bboxes
            target_boxes = []
            for box, label, score in zip(boxes, labels, scores):
                box_height = int(box[3]-box[1])
                if (
                    label == 1 # person class
                    and score > self.min_confidence
                    and box_height > self.min_height
                    ):
                    target_boxes.append({ 'conf': score, 'bbox': box })

            results.append(target_boxes)

        return results
