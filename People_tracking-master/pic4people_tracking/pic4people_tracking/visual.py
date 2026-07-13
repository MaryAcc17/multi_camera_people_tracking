import os
import cv2
import time
import math
import numpy as np
import hashlib
import colorsys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

def plot_tracked_BB(img, tracks, velocities, masks=None, points=None, name='Tracking'):

    for i, track in enumerate(tracks):
        xmin, ymin, xmax, ymax, x, y, z, ID  = tuple(track[:8])
        # vx, _, vz, _ = tuple(velocities[i]) 

        font = cv2.FONT_HERSHEY_SIMPLEX
        if x is not None:
            text        = f"ID:{ID}, x={x:.3f}, y={y:.3f}, z={z:.3f}"
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

        cv2.putText(img, text, position_min, font, fontScale, fontColor, lineType)

        cv2.rectangle(img, position_min, position_max, (0,255,0), lineType)
        
        if points is not None:

            point = tuple(points[i][:2])
            cv2.circle(img, point, 2, (0,255,0), -1)
    
    # display(img, name)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)# img

links = [
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6), (11, 12),  # Shoulders and hips
    (5, 11), (6, 12)  # Torso connections
]

confidence_threshold = 0.5 # for pose keypoints plot

def plot_tracked_BB_pose(img, tracks, keypoints=None, points=None, name='Tracking'):

    for i, track in enumerate(tracks):
        xmin, ymin, xmax, ymax, x, y, z, ID  = tuple(track[:8]) 

        font = cv2.FONT_HERSHEY_SIMPLEX
        if x is not None:
            text        = f"ID:{ID}, x={x:.3f}, y={y:.3f}, z={z:.3f}"
        else:
            text        = f"ID:{ID}"

        position_min= (int(xmin), int(ymin))
        position_max= (int(xmax), int(ymax))
        fontScale   = 0.5
        fontColor   = (0,255,0)
        lineType    = 2

        if keypoints is not None:
            pose = keypoints[i]
            for keypoint in pose:
                if keypoint[2] > confidence_threshold:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            for link in links:
                pt1 = pose[link[0]]
                pt2 = pose[link[1]]
                if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
             

        cv2.putText(img, text, position_min, font, fontScale, fontColor, lineType)

        cv2.rectangle(img, position_min, position_max, (0,255,0), lineType)
        
        if points is not None:
            point = tuple(points[i][:2])
            cv2.circle(img, point, 2, (0,255,0), -1)
    
    # display(img, name)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def plot_tracked_BB_pose2(img, tracks, keypoints=None, points=None, yaws=None, name='Tracking'):
    for i, track in enumerate(tracks):
        xmin, ymin, xmax, ymax, x, y, z, ID = tuple(track[:8])
        yaw = yaws[i][0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        if x is not None:
            text = f"ID:{ID}, x={x:.3f}, y={y:.3f}, z={z:.3f}"
        else:
            text = f"ID:{ID}"

        position_min = (int(xmin), int(ymin))
        position_max = (int(xmax), int(ymax))
        fontScale = 0.5
        fontColor = (0, 255, 0)
        lineType = 2

        # Calculate the midpoint of the bounding box for placing the ID number
        midpoint = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

        # Draw the keypoints if provided
        if keypoints is not None:
            for idx, kpoints in enumerate(keypoints):
                indices = [
                    1, # left eye
                    2, # right eye
                    3, # left ear
                    4, # right ear
                    5, # left shoulder
                    6, # right shoulder
                    11, # left hip
                    12, # right hip
                ] 
                l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_hip, r_hip = kpoints[indices]

                # Draw keypoints
                
                cv2.circle(img, (int(l_sh[0]), int(l_sh[1])), 5, (0, 255, 255), thickness=-1)  
                cv2.circle(img, (int(r_sh[0]), int(r_sh[1])), 5, (255, 0, 255), thickness=-1)  
                cv2.circle(img, (int(l_hip[0]), int(l_hip[1])), 5, (0, 255, 255), thickness=-1)
                cv2.circle(img, (int(r_hip[0]), int(r_hip[1])), 5, (255, 0, 255), thickness=-1)
                if l_eye[2] > 0.6:
                    cv2.circle(img, (int(l_eye[0]), int(l_eye[1])), 5, (0, 255, 255), thickness=-1)
                if r_eye[2] > 0.6:
                    cv2.circle(img, (int(r_eye[0]), int(r_eye[1])), 5, (255, 0, 255), thickness=-1)
                if l_ear[2] > 0.6:
                    cv2.circle(img, (int(l_ear[0]), int(l_ear[1])), 5, (0, 255, 255), thickness=-1)
                if r_ear[2] > 0.6:
                    cv2.circle(img, (int(r_ear[0]), int(r_ear[1])), 5, (255, 0, 255), thickness=-1)
                
                # Draw vector
                cv2.arrowedLine(img, (int(l_sh[0]), int(l_sh[1])), (int(r_sh[0]), int(r_sh[1])), (0, 255, 0), 2)
                cv2.line(img, (int(l_hip[0]), int(l_hip[1])), (int(r_hip[0]), int(r_hip[1])), (0, 255, 0), 2)
                cv2.line(img, (int(l_hip[0]), int(l_hip[1])), (int(l_sh[0]), int(l_sh[1])), (255, 255, 0), 2)
                cv2.line(img, (int(r_hip[0]), int(r_hip[1])), (int(r_sh[0]), int(r_sh[1])), (255, 255, 0), 2)

                text = f"Estimated Orientation: {np.degrees(yaw):.2f} degrees"
                cv2.putText(img, text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        # Draw the bounding box and the ID in the middle of the bounding box
        # cv2.rectangle(img, position_min, position_max, (0, 255, 0), lineType)
        # cv2.putText(img, str(ID), midpoint, font, fontScale, fontColor, lineType)

        # Draw additional text near the top-left corner of the bounding box
        # cv2.putText(img, text, position_min, font, fontScale, fontColor, lineType)

        # Draw the point if provided
        if points is not None:
            point = tuple(points[i][:2])
            cv2.circle(img, point, 2, (0, 255, 0), -1)

    # display(img, name)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


COLORS = [
    '#FF0000',     # Red
    '#00FF00',     # Bright Green
    '#0000FF',     # Blue
    '#FF9900',     # Orange
    '#9900FF',     # Purple
    '#00FFFF',     # Cyan
    '#FF00FF',     # Magenta
    '#FFFF00',     # Yellow
]

def mpl_to_rgb(color): 
    rgb = mcolors.to_rgb(color)
    return tuple(int(val * 255) for val in rgb)  

def plot_tracked_BB_pose3(img, tracks, keypoints=None, points=None, name='Tracking'):

    for i, track in enumerate(tracks):
        xmin, ymin, xmax, ymax, x, y, z, ID  = tuple(track[:8]) 

        color_idx = (int(ID)-1) % len(COLORS)
        mpl_color = COLORS[color_idx]
        print(f"ID {ID}, color_idx {color_idx}, mpl_color {mpl_color}")
        bgr_color = mpl_to_rgb(mpl_color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # if x is not None:
        #    text        = f"ID:{ID}, x={x:.3f}, y={y:.3f}, z={z:.3f}"
        # else:
        text        = f"ID:{ID}"

        position_min= (int(xmin), int(ymin))
        position_max= (int(xmax), int(ymax))
        fontScale   = 0.5
        fontColor   = bgr_color
        lineType    = 2

        if keypoints is not None:
            pose = keypoints[i]
            for keypoint in pose:
                if keypoint[2] > confidence_threshold:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            for link in links:
                pt1 = pose[link[0]]
                pt2 = pose[link[1]]
                if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
             

        cv2.putText(img, text, position_min, font, fontScale, fontColor, lineType)

        cv2.rectangle(img, position_min, position_max, fontColor, lineType)
        
        if points is not None:
            point = tuple(points[i][:2])
            cv2.circle(img, point, 2, fontColor, -1)
    
    # display(img, name)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)