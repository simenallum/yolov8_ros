#!/usr/bin/env python3

import rospy
import os
import numpy as np
import yaml
import sys

from yolov8_ros_utils.yolo8 import YOLOv8

class YOLOv8Detector:
	
			def __init__(self, config_file=None):

				rospy.init_node("YOLOv8Detector", anonymous=False)

				script_dir = os.path.dirname(os.path.realpath(__file__))

				if config_file is None:
						config_file = rospy.get_param("~config_file")

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

				self.model_path = rospy.get_param("~yolo_model_path")

			def setup_subscribers(self):
				pass


			def setup_publishers(self):
				pass

			def initalize_detector(self):
				self.detector = YOLOv8(self.model_path)

			def _shutdown():
				rospy.loginfo("Shutting down yolov8_ros node")

			def start(self):
				rospy.loginfo("Starting yolov8_ros node")
				rospy.loginfo(f"Model class IDs and names: {self.detector.get_class_names()}")

				rospy.on_shutdown(self._shutdown)

				while not rospy.is_shutdown():

					rospy.spin()

def main():
		Detector = YOLOv8Detector()
		Detector.start()

if __name__ == "__main__":
		main()