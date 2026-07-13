import os
import cv2
import time
import math
import numpy as np

import pyrealsense2 as rs
from .visual_utils import *

#FOCAL_DISTANCE_X = 613.237854003906
#FOCAL_DISTANCE_Y = 612.938232421875


class RealSense():
    def __init__(self):

        self.device_ID = '117222250314'
        self.color = True
        self.depth = True
        self.aligned = True
        self.infra1 = False # left infrared imager
        self.infra2 = False # right infrared imager
        self.points_bool = False
        self.W = 640 #1280 #640
        self.H = 480 #720 #480
        self.fps = 30
        self.visualize = False
        self.max_distance = 4.0
        self.pipe = rs.pipeline()     # Create a context object. This object owns the handles to all connected realsense devices

        realsense_ctx = rs.context()
        for i in range(len(realsense_ctx.devices)):
            camera_name = realsense_ctx.devices[i].get_info(rs.camera_info.name)
            camera_number = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            print("Camera "+camera_name+" with serial number: "+camera_number)

        # Configure streams
        self.config = rs.config() 

        self.config.enable_stream(      # to enable depth data stream
            rs.stream.depth, 
            self.W, 
            self.H,
            rs.format.z16, 
            self.fps
            )
        self.config.enable_stream(      # to enable RGB data stream
            rs.stream.color, 
            self.W, 
            self.H,
            rs.format.rgb8, 
            self.fps
            )
        if self.infra1:
            self.config.enable_stream(
                rs.stream.infrared, 
                1, 
                self.W,
                self.H, 
                rs.format.y8, 
                self.fps
                )
        if self.infra2:
            self.config.enable_stream(
                rs.stream.infrared, 
                2, 
                self.W,
                self.H, 
                rs.format.y8, 
                self.fps
                )
        
        #self.config.enable_device(self.device_ID)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Start streaming with requested config
        d400_attempt = False
        while d400_attempt == False:
            try:
                self.pipe.start(self.config)                # Start streaming
                print("D435i Found!")
                d400_attempt = True
            except RuntimeError:
                print("Device not connected. "
                    "Connect your Intel RealSense D435i")

        # Available filters and control options for the filters
        # https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude,5)
        self.spatial.set_option(rs.option.holes_fill, 5)
        #self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter(4)
        #self.hole_filling = rs.hole_filling_filter(0)


    def take_color_depth_aligned(self):
        """
        Returns self.color_image
        """
        try:
            frames  = self.pipe.wait_for_frames()
            aligned_frames = self.align.process(frames) # Align the depth frame to color frame

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame() # to get the color_frame
            self.color_image = np.asanyarray(color_frame.get_data()) # to convert the image to numpy array

            self.aligned_depth_frame = aligned_frames.get_depth_frame() # to get the depth_frame
            self.depth_intrinsic = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics # to retrieve intrinsic parameters
            #print(self.depth_intrinsic)

            self.aligned_depth_image = np.asanyarray(
                self.aligned_depth_frame.get_data(), 
                dtype= np.float32
                )

            #if self.visualize:
                #display(self.color_image, "Color image")
            return True, self.color_image

        except:
            print("ERROR taking color frame")
            return False, None
        
    def take_depth_frame(self): # Non manca il return????
        """
        Returns self.depth_image
        """
        try:
            frames = self.pipe.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            self.depth_image = np.asanyarray(depth_frame.get_data(), 
                dtype= np.float32)

            if self.visualize:
                display(self.depth_image, "Depth image")
        except:
            print("ERROR taking depth frame")

    def take_aligned_points(self, points = False):
        """
        Returns self.aligned_depth_image and self.points (depth pointcloud)
        """
        try:
            frames = self.pipe.wait_for_frames()
            aligned_frames = self.align.process(frames)

            aligned_frames = aligned_frames.get_depth_frame()

            self.aligned_depth_image = np.asanyarray(
                aligned_frames.get_data(), 
                dtype= np.float32
                )

            if points:
                depth_frame = self.spatial.process(aligned_frames)
                #depth_frame = self.hole_filling.process(depth_frame)
                aligned_depth_decimate = self.decimate.process(depth_frame)
                pcl = self.pc.calculate(aligned_depth_decimate)

                self.points = np.asanyarray(
                    pcl.get_vertices()).view(np.float32).reshape(-1,3)

            if self.visualize:
                display(self.aligned_depth_image, "Aligned depth image")
        except:
            print("ERROR taking pointcloud")

    def take_infra1(self):
        """
        Returns self.color_image, self.depth_image, self.aligned_depth_image
        and self.points
        """

        frames = self.pipe.wait_for_frames()

        frames = frames.get_infrared_frame(1)
        self.infra1_frames = np.asanyarray(frames.get_data())
        #self.infra1_frames = np.asanyarray(frames.first(rs.stream.infrared).get_data())

        if self.visualize:
            display(self.infra1_frames, "Infrared1 image")


    def take_infra2(self):
        """
        Returns self.color_image, self.depth_image, self.aligned_depth_image
        and self.points
        """

        frames = self.pipe.wait_for_frames()

        frames = frames.get_infrared_frame(2)
        self.infra2_frames = np.asanyarray(frames.get_data())
        #self.infra2_frames = np.asanyarray(frames.second(rs.stream.infrared).get_data())
        if self.visualize:
            display(self.infra2_frames, "Infrared2 image")

    def process_frame(self):
        """
        Method for processing frame (can be expanded in future)
        """
        size = (641, 481)
        return cv2.resize(self.color_image, size)


def main(args=None):
    ""
    ""
    realsense = RealSense()

if __name__ == '__main__':
    main()
