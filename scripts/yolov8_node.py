#!/usr/bin/env python3

import rospy
import os
import numpy as np
import yaml
import sys
from cv_bridge import CvBridge

from yolov8_ros_utils.yolo8 import YOLOv8
from yolov8_ros.msg import BoundingBox, BoundingBoxes
import sensor_msgs.msg

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
		self.bridge = CvBridge()
		self.model_path = self.config["model"]["path"]
		self.publish_annotated_image = self.config["settings"]["publish_detection_images"]


	def setup_subscribers(self):
		rospy.Subscriber(
			self.config["topics"]["input"]["image"], 
			sensor_msgs.msg.Image, 
			self._new_image_cb
		)


	def setup_publishers(self):
		if self.publish_annotated_image:
			self.detection_image_pub = rospy.Publisher(
				self.config["topics"]["output"]["detectionImage"], 
				sensor_msgs.msg.Image, 
				queue_size=10
			)

		self.detection_boxes_pub = rospy.Publisher(
			self.config["topics"]["output"]["boxes"], 
			BoundingBoxes, 
			queue_size=10
		)

	def initalize_detector(self):
		self.detector = YOLOv8(self.model_path)
		self.class_labels = self.detector.get_class_labels()

	def _new_image_cb(self, image):
		image_msg = self.bridge.imgmsg_to_cv2(image, "bgr8")

		boxes, confidence, classes, annotated_image =  self.detector(image_msg, self.publish_annotated_image)

		msg = self._prepare_boundingbox_msg(boxes, confidence, classes)
		self._publish_detected_boundingboxes(msg)

		if self.publish_annotated_image:
			self._publish_detected_image(annotated_image)

	def _prepare_boundingbox_msg(self, boxes, confidence, classes):
		boundingBoxes = BoundingBoxes()
		boundingBoxes.header.stamp = rospy.Time.now()

		for i in range(len(boxes)):
			boundingBox = BoundingBox()
			boundingBox.probability = confidence[i]
			boundingBox.xmin = int(boxes[i][0])
			boundingBox.ymin = int(boxes[i][1])
			boundingBox.xmax = int(boxes[i][2])
			boundingBox.ymax = int(boxes[i][3])
			boundingBox.id = classes[i]
			boundingBox.Class = self.class_labels[classes[i]]

			boundingBoxes.bounding_boxes.append(boundingBox)

		return boundingBoxes

	def _publish_detected_boundingboxes(self, msg):
		self.detection_boxes_pub.publish(msg)

	def _publish_detected_image(self, image):
		msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
		self.detection_image_pub.publish(msg)


	def _shutdown():
		rospy.loginfo("Shutting down yolov8_ros node")

	def start(self):
		rospy.loginfo("Starting yolov8_ros node")
		rospy.loginfo(f"Model class names: {self.detector.get_class_labels()}")

		rospy.on_shutdown(self._shutdown)

		while not rospy.is_shutdown():

			rospy.spin()

def main():
		Detector = YOLOv8Detector()
		Detector.start()

if __name__ == "__main__":
		main()