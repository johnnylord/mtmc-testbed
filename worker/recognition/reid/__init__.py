import cv2
import numpy as np
from PIL import Image

import torch
from torch.hub import load_state_dict_from_url
from torchvision import transforms

from ..base import PersonRecognizer
from .model import resnet18_reid


__all__ = [ "Resnet18" ]

class Resnet18(PersonRecognizer):

    PRETRAIN_URL = "https://www.dropbox.com/s/lnr9megu682n6ef/crossentropytriplet_market1501dukemtmc_resnet18reid.pth?dl=1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load pretrained model
        self.model = resnet18_reid(features=128)
        if self.pretrain_model is not None:
            state_dict = torch.load(self.pretrain_model)
        else:
            state_dict = load_state_dict_from_url(Resnet18.PRETRAIN_URL)

        # Drop classifier layer
        drop_keys = [ k for k in state_dict.keys() if 'classifier' in k ]
        for k in drop_keys:
            del state_dict[k]

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing layer
        self.preprocess = transforms.Compose([
                            transforms.Resize((self.size[1], self.size[0])),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                            ])

    def preprocessing(self, imgs):
        inputs = []
        for img in imgs:
            pil_img = Image.fromarray(img)
            input_ = self.preprocess(pil_img)
            inputs.append(input_)

        inputs = torch.stack(inputs)
        inputs = inputs.to(self.device)

        return inputs

    def postprocessing(self, output):
        embeddings = output.detach().cpu().numpy()
        return embeddings
