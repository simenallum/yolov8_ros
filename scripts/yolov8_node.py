#!/usr/bin/env python3

import rospy
import os
import numpy as np
import yaml
import sys
from cv_bridge import CvBridge

import time

from yolov8_ros_utils.yolo8 import YOLOv8
from yolov8_ros.msg import BoundingBox, BoundingBoxes
from std_srvs.srv import SetBool, SetBoolResponse
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

		self._initalize_parameters()
		self._setup_publishers()
		self._setup_subscribers()
		self._initalize_services()
		self._initalize_detector()

	def _initalize_parameters(self):
		self.bridge = CvBridge()
		self.model_path = self.config["model"]["path"]
		self.publish_annotated_image = self.config["settings"]["publish_detection_images"]
		self.conf_threshold = self.config["model"]["confidence_threshold"]
		self.iou_threshold = self.config["model"]["iou_threshold"]
		self.print_terminal_output = self.config["settings"]["print_terminal_output"]
		self.include_image_in_bb_msg = self.config["settings"]["include_image_in_bb_msg"]

		# Control signal parameters
		self.process_image = False
		self.model_initialized = False

	def _initalize_detector(self):
			rospy.logwarn("Wait for model to be initialized!")
			self.detector = YOLOv8(self.model_path, self.conf_threshold, self.iou_threshold, self.print_terminal_output)
			self.class_labels = self.detector.get_class_labels()

			_ =  self.detector(np.zeros((1980, 1080, 3))) # Warmup pass through network
			rospy.logwarn("Model initialized!")
			self.model_initialized = True

			
	def _setup_subscribers(self):
		rospy.Subscriber(
			self.config["topics"]["input"]["image"], 
			sensor_msgs.msg.Image, 
			self._new_image_cb
		)

	def _initalize_services(self):
		self.srv_process_image = rospy.Service(
			self.config["control"]["service"]["process_image"],
			SetBool, 
			self._handle_process_image
		)


	def _setup_publishers(self):
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

	
	def _new_image_cb(self, image):
		if self.process_image and self.model_initialized:
			image_msg = self.bridge.imgmsg_to_cv2(image, "bgr8")

			boxes, confidence, classes, annotated_image =  self.detector(image_msg, self.publish_annotated_image)

			msg = self._prepare_boundingbox_msg(boxes, confidence, classes, image_msg)
			self._publish_detected_boundingboxes(msg)

			if self.publish_annotated_image:
				self._publish_detected_image(annotated_image)

	def _handle_process_image(self, req):
		if req.data:
			self.process_image = True
			res = SetBoolResponse()
			res.success = True
			res.message = "Started processing data"
			return res 
		else:
			self.process_image = False
			res = SetBoolResponse()
			res.success = True
			res.message = "Stopped processing data"
			return res


	def _prepare_boundingbox_msg(self, boxes, confidence, classes, frame):
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

		if self.include_image_in_bb_msg:
			boundingBoxes.frame = self.bridge.cv2_to_imgmsg(frame, "bgr8")

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