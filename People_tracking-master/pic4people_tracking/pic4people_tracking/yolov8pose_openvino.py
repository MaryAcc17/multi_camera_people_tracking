import torch
import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from pathlib import Path
import ipywidgets as widgets
import openvino as ov
import sys

def nms(boxes, scores, iou_threshold=0.45):
    # assicura numpy array
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if isinstance(scores, list):
        scores = np.array(scores)

    # OpenCV vuole liste Python
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()

    indices = cv2.dnn.NMSBoxes(
        boxes_list,
        scores_list,
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return []

    return indices.flatten()

class yolov8pose_OpenVINO():

    def __init__(self, model_filepath, min_conf_threshold=0.25,ros_logger=None):
        self.filepath = Path(__file__).resolve()
        self.models_dir = Path(model_filepath)

        self.min_conf_threshold = min_conf_threshold
        self.ros_logger = ros_logger

        # pose detection model
        self.pose_model_path = self.models_dir
        self.core = ov.Core()

        device = 'AUTO'
        self.pose_ov_model = self.core.read_model(self.pose_model_path)
        if device != "CPU":
            self.pose_ov_model.reshape({0: [1, 3, 480, 640]})
        self.pose_compiled_model = self.core.compile_model(self.pose_ov_model, device)
        
    # Funzione log centralizzata
    def log(self, msg):
        if self.ros_logger:
            self.ros_logger.info(msg)
        else:
            print(msg)

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
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
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
        nms_iou_threshold:float = 0.45,
        agnosting_nms:bool = False,
        max_detections:int = 80,
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
        Returns:
        pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and
                                            kpt - 17 keypoints in format [x1, y1, score1]
        """
        results = []

        # rimuovi batch dimension
        pred_boxes = pred_boxes[0]  # (6300, 56)

        boxes = []
        scores = []
        keypoints = []

        for pred in pred_boxes:
            conf = pred[4]

            if conf < min_conf_threshold:
                continue

            x, y, w, h = pred[:4]

            # converti da xywh a xyxy
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # formato OpenCV
            scores.append(float(conf))
            keypoints.append(pred[5:])

        if len(boxes) == 0:
            return [{"box": np.empty((0, 6)), "kpt": np.empty((0, 17, 3))}]

        # NMS
        indices = nms(boxes, scores, nms_iou_threshold)

        final_boxes = []
        final_kpts = []

        for i in indices:
            x, y, w, h = boxes[i]
            score = scores[i]

            final_boxes.append([x, y, x + w, y + h, score, 0])

            kpt = np.array(keypoints[i]).reshape(17, 3)

            # scaling keypoints
            gain_w = orig_img.shape[1] / input_hw[1]
            gain_h = orig_img.shape[0] / input_hw[0]

            kpt[:, 0] *= gain_w
            kpt[:, 1] *= gain_h

            final_kpts.append(kpt)

        return [{
            "box": np.array(final_boxes),
            "kpt": np.array(final_kpts)
        }]

    def detect(self, image:np.ndarray):
        """
        OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
        Parameters:
            image (np.ndarray): input image.
            model (Model): OpenVINO compiled model.
        Returns:
            detections (np.ndarray): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and
                                    kpt - 17 keypoints in format [x1, y1, score1]
        """
        self.log(">>> DETECT CHIAMATO <<<")
        preprocessed_image = self.preprocess_image(image)
        input_tensor = self.image_to_tensor(preprocessed_image)
        self.log(f"Input tensor shape: {input_tensor.shape}")
        self.log(f"Input min: {input_tensor.min()}, max: {input_tensor.max()}")

        # inferenza OpenVINO
        result = self.pose_compiled_model([input_tensor])
        boxes = result[self.pose_compiled_model.outputs[0]]
        boxes = boxes.transpose(0, 2, 1)
        
        
        
        
        self.log("=== DEBUG DETECT ===")
        self.log(f"Input tensor shape: {input_tensor.shape}")
        

        if boxes.size>0:
            self.log(f"RAW boxes shape: {boxes.shape}")
            self.log(f"MIN score: {boxes[...,4].min()}, MAX score: {boxes[...,4].max()}")
            self.log(f"Sample boxes: {boxes.flatten()[:20]}")
        else:
            self.log("Boxes vuoti!")

        input_hw = input_tensor.shape[2:]
        detections = self.postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image,
                                      min_conf_threshold=self.min_conf_threshold)
        self.log(f"Detections after postprocess: {detections}")
        return detections

def main(args=None):
    ""
    ""
    yolo_ov = yolov8pose_OpenVINO(model_filepath='/workspaces/hunavsim_devcontainer/src/People_tracking-master/pic4people_tracking/yolov8s-pose_openvino_model/yolov8s-pose.xml')

    main()

