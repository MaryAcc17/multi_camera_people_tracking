import os
import cv2
import time
import math
import numpy as np


def display(img, window_name):
    """
    Display the image img.
    """
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(window_name, img)
        #cv2.waitKey(1)
    except :
        print("Error in opening frame. Image shape: {}".format(img.shape))



def plot_tracked_BB(bb, masks, img, poses, tr_centroids, name='Tracking'):

    for i in range(len(bb)):
        xmin, ymin, xmax, ymax, ID  = tuple(bb[i])

        font        = cv2.FONT_HERSHEY_SIMPLEX
        if poses is not None and len(poses) >= i+1 and poses[i][2] is not None:
            text        = f"ID:{ID}, x={poses[i][0]:.3f}, y={poses[i][1]:.3f}, z={poses[i][2]:.3f}"
        else:
            text        = f"ID:{ID}"

        position_min= (int(xmin), int(ymin))
        position_max= (int(xmax), int(ymax))
        fontScale   = 0.5
        fontColor   = (0,255,0)
        lineType    = 2

        if masks is not None:
            mask = masks[i]
            mask = mask[...,None]
            masked_img = np.where(mask, [0, 255, 0], img).astype(np.uint8)
            #print(masked_img.shape)

            img = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)

        cv2.putText(img,
            text, 
            position_min, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.rectangle(img,
            position_min,
            position_max,
            (0,255,0),
            lineType
            )
        
        point=tuple(tr_centroids[i][:2])
        
        cv2.circle(
            img,
            point,
            2,
            (0,255,0),
            -1
            )
    
    display(img, name)
    return img

def plot_disp_roi(disp, text, delta, poses, centroids):

    for i in range(len(poses)):
        if centroids[i] is not None:
            x = centroids[i][0]
            y = centroids[i][1]

            # Get disparity frame for nicer depth visualization
            disp = (disp * (255 / 95)).astype(np.uint8)
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

            text.rectangle(disp, (x-delta, y-delta), (x+delta, y+delta))
            text.putText(disp,
                "X: " + ("{:.3f}m".format(poses[i][0]/1000) if not math.isnan(poses[i][0]) else "--"),
                (x + 10, y + 20))
            text.putText(disp, 
                "Y: " + ("{:.3f}m".format(poses[i][1]/1000) if not math.isnan(poses[i][1]) else "--"),
                (x + 10, y + 35))
            text.putText(disp, 
                "Z: " + ("{:.3f}m".format(poses[i][2]/1000) if not math.isnan(poses[i][2]) else "--"),
                (x + 10, y + 50))

    # Show the frame
    display(disp, "Disparity with ROI poses")


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.text_scale = 0.5
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, self.text_scale, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, self.text_scale, self.color, 1, self.line_type)
    def rectangle(self, frame, p1, p2):
        cv2.rectangle(frame, p1, p2, self.bg_color, 3)
        cv2.rectangle(frame, p1, p2, self.color, 1)

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time() + 1
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)