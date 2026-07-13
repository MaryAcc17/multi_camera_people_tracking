import torch
import numpy as np
from pathlib import Path
from pic4people_tracking.StrongSORT.utils import logger as LOGGER

from pic4people_tracking.StrongSORT.appearance.backends.base_backend import BaseModelBackend
from pic4people_tracking.StrongSORT.utils.checks import TestRequirements

tr = TestRequirements()


class TorchscriptBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

        LOGGER.info(f"Loading {w} for TorchScript inference...")
        self.model = torch.jit.load(w)
        self.model.half() if self.half else self.model.float()

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features
