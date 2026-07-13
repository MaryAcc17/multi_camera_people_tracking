import pyrealsense2 as rs
import numpy as np
import cv2
import os.path

class RealSense_readBag():

    def __init__(self, filepath):

        self.filepath = filepath

        if not filepath:
            raise ValueError("No input paramater have been given.")
        
        if os.path.splitext(self.filepath)[1] != ".bag":
            raise ValueError("The given file is not of correct file format.\nOnly .bag files are accepted")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        rs.config.enable_device_from_file(self.config, self.filepath, repeat_playback=False)

        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Start streaming from file
        self.profile = self.pipeline.start(self.config)

        self.playback = self.profile.get_device().as_playback()
        # self.playback.set_real_time(False)
        self.playback.pause()


        self.W = None
        self.H = None
        self.max_distance = 4.0
        
    def take_color_frame(self):
        """
        Returns self.color_image
        """
        try:
            frames  = self.pipe.wait_for_frames()

            color_frame = frames.get_color_frame() # to get the color_frame
            color_image = np.asanyarray(color_frame.get_data()) # to convert the image to numpy array

            #if self.visualize:
                #display(self.color_image, "Color image")
            return True, color_image

        except:
            print("ERROR taking color frame")
            return False, None        

    def take_color_depth_aligned(self):
        try:
            frames  = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames) # Align the depth frame to color frame

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame() # to get the color_frame
            self.color_image = np.asanyarray(color_frame.get_data()) # to convert the image to numpy array

            self.aligned_depth_frame = aligned_frames.get_depth_frame() # to get the depth_frame
            self.depth_intrinsic = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics # to retrieve intrinsic parameters
            #print(self.depth_intrinsic)

            if self.W == None:
                self.W = self.depth_intrinsic.width
                self.H = self.depth_intrinsic.height

            self.aligned_depth_image = np.asanyarray(
                self.aligned_depth_frame.get_data(), 
                dtype= np.float32
                )

            return True, self.color_image

        except Exception as e:
            print("ERROR taking color frame")
            print(e)
            return False, None

    