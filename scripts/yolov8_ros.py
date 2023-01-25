#!/usr/bin/env python3

import rospy
import os
import numpy as np
import yaml
import sys

import yolov8_ros_utils.yolo8

class YOLOv8Detector:
	
			def __init__(self, config_file=None):

				rospy.init_node("YOLOv8Detector", anonymous=False)

				script_dir = os.path.dirname(os.path.realpath(__file__))

				try:
						with open(f"{script_dir}/../config/{config_file}") as f:
								self.config = yaml.safe_load(f)
				except Exception as e:
						rospy.logerr(f"Failed to load config: {e}")
						sys.exit()

				self.initalize_parameters()
				self.setup_publishers()
				self.setup_subscribers()
				self.initalize_detector()

			def initalize_parameters(self):

				self.model_path = rospy.get_param("yolo_model_path")

			def setup_subscribers(self):
				pass


			def setup_publishers(self):
				pass

			def initalize_detector(self):
				self.detector = yolov8_ros_utils.yolo8(self.model_path)