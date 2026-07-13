import torch
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from ultralytics.yolo.utils import ops
from pathlib import Path
import ipywidgets as widgets
import openvino as ov

class yolov8_OpenVINO():

    def __init__(self, model_filepath, min_conf_threshold=0.65):
        self.filepath = Path(__file__).resolve()
        self.models_dir = Path(model_filepath) # self.filepath.parent/'yolo_models/'
        self.models_dir.mkdir(exist_ok=True)

        self.min_conf_threshold = min_conf_threshold

        self.seg_model_name = "best_1_seg"

        # instance segmentation model
        self.seg_model_path = self.models_dir / f"{self.seg_model_name}_int8_openvino_model/{self.seg_model_name}.xml"

        try:
            self.scale_segments = ops.scale_segments
        except AttributeError:
            self.scale_segments = ops.scale_coords

        self.core = ov.Core()

        device = 'AUTO'
        self.seg_ov_model = self.core.read_model(self.seg_model_path)
        if device!= "CPU":
            self.seg_ov_model.reshape({0: [1, 3, 480, 640]})
        self.seg_compiled_model = self.core.compile_model(self.seg_ov_model, device)

    def letterbox(self, img: np.ndarray, new_shape:Tuple[int, int] = (480, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
        """
        Resize image and padding for detection. Takes image as input,
        resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

        Parameters:
        img (np.ndarray): image for preprocessing
        new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
        color (Tuple(int, int, int)): color for filling padded area
        auto (bool): use dynamic input size, only padding for stride constrins applied
        scale_fill (bool): scale image to fill new_shape
        scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
        stride (int): input padding stride
        Returns:
        img (np.ndarray): image after preprocessing
        ratio (Tuple(float, float)): hight and width scaling ratio
        padding_size (Tuple(int, int)): height and width padding size


        """
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def preprocess_image(self, img0: np.ndarray):
        """
        Preprocess image according to YOLOv8 input requirements.
        Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

        Parameters:
        img0 (np.ndarray): image for preprocessing
        Returns:
        img (np.ndarray): image after preprocessing
        """
        # resize 
        shape = img0.shape[:2]
        if shape != (480,640):
            img = self.letterbox(img0)[0] ## only if img0 shape != (480, 640)
        else:
            img = img0

        # Convert HWC to CHW
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img


    def image_to_tensor(self, image:np.ndarray):
        """
        Preprocess image according to YOLOv8 input requirements.
        Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

        Parameters:
        img (np.ndarray): image for preprocessing
        Returns:
        input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
        """
        input_tensor = image.astype(np.float32)  # uint8 to fp32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        # add batch dimension
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor


    def postprocess(
        self,
        pred_boxes:np.ndarray,
        input_hw:Tuple[int, int],
        orig_img:np.ndarray,
        min_conf_threshold:float = 0.25,
        nms_iou_threshold:float = 0.7,
        agnosting_nms:bool = False,
        max_detections:int = 300,
        pred_masks:np.ndarray = None,
        retina_mask:bool = False
    ):
        """
        YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
        Parameters:
            pred_boxes (np.ndarray): model output prediction boxes
            input_hw (np.ndarray): preprocessed image
            orig_image (np.ndarray): image before preprocessing
            min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
            nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
            max_detections (int, *optional*, 300):  maximum detections after NMS
            pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
            retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
        Returns:
        pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and
                                            segment - segmentation polygons for each element in batch
        """
        nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
        # if pred_masks is not None:
        #     nms_kwargs["nm"] = 32
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes),
            min_conf_threshold,
            nms_iou_threshold,
            nc=1,
            **nms_kwargs
        )
        results = []
        proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            if not len(pred):
                results.append({"det": np.empty((0,6), dtype=np.uint8), "segment": np.empty((0, shape[0], shape[1]), dtype=np.uint8)}) 
                continue
            if proto is None:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                results.append({"det": pred})
                continue
            if retina_mask:
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
                segments = [self.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
                binary_masks = self.create_binary_masks(segments, shape)
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
                pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
                segments = [self.scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
                binary_masks = self.create_binary_masks(segments, shape)

            results.append({"det": pred[:, :6].numpy(), "segment": binary_masks})
        return results
    
    def create_binary_masks(self, segments, img_shape):
        
        binary_masks = []
        h = img_shape[0]
        w = img_shape[1]

        for mask in segments:
            binary_mask = np.zeros((h,w),dtype=np.uint8)
            contour_points = np.array(mask, dtype=np.int32)
            contour_points = contour_points.reshape((-1,1,2))
            cv2.fillPoly(binary_mask, [contour_points], color=1)

            binary_masks.append(binary_mask)
        
        return binary_masks


    def detect(self, image:np.ndarray):
        """
        OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
        Parameters:
            image (np.ndarray): input image.
            model (Model): OpenVINO compiled model.
        Returns:
            detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
        """
        model = self.seg_compiled_model
        num_outputs = len(model.outputs)
        preprocessed_image = self.preprocess_image(image)
        input_tensor = self.image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        masks = None
        if num_outputs > 1:
            masks = result[model.output(1)]
        input_hw = input_tensor.shape[2:]
        detections = self.postprocess(pred_boxes=boxes, input_hw=input_hw, 
                                      orig_img=image, min_conf_threshold=self.min_conf_threshold, pred_masks=masks)
        return detections


def main(args=None):
    ""
    ""
    yolo_ov = yolov8_OpenVINO()

if __name__ == '__main__':
    main()

