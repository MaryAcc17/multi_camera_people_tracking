import numpy as np
import cv2
from pathlib import Path
import openvino as ov

class ReIDInference:
    def __init__(self, weights):
        self.core = ov.Core()
        w = weights[0] if isinstance(weights, list) else weights
        if not Path(w).is_file():  # if not *.xml
            try:
                w = next(
                    Path(w).glob("*.xml")
                )  # get *.xml file from *_openvino_model dir
            except Exception as e:
                print(e)
        self.model = self.core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
        if self.model.get_parameters()[0].get_layout().empty:
            self.model.get_parameters()[0].set_layout(ov.Layout("NCWH"))
        self.executable_network = self.core.compile_model(
            self.model, device_name="CPU"
        )
        self.output_layer = next(iter(self.executable_network.outputs))

    def get_crops(self, xyxys, img):
        crops = []
        h, w = img.shape[:2]
        resize_dims = (128, 256)
        interpolation_method = cv2.INTER_LINEAR
        mean_array = np.array([0.485, 0.456, 0.406])
        std_array = np.array([0.229, 0.224, 0.225])

        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            
            crop = cv2.resize(crop, resize_dims, interpolation=interpolation_method)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - mean_array) / std_array
            crop = np.transpose(crop, (2, 0, 1))
            crops.append(crop)

        return np.stack(crops, axis=0)

    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            features = self.executable_network([crops])[self.output_layer]
            features = features.squeeze()
        else:
            features = np.array([])
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        return features
    