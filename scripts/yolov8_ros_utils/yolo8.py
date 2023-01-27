import ultralytics
import pandas as pd
import cv2

from yolov8_ros_utils.utilities import draw_detections as DD
from yolov8_ros_utils.utilities import nms as non_maxima_suppression

class YOLOv8:

		def __init__(self, path, conf_thres=0.7, iou_thres=0.5, print_terminal_output=False):
				self.conf_threshold = conf_thres
				self.iou_threshold = iou_thres
				self.print_terminal_output = print_terminal_output

				# Initialize model
				self._initialize_model(path)

		def __call__(self, image, make_detection_image=False):
				return self._detect_objects(image, make_detection_image)

		def _initialize_model(self, path):
			self.model = ultralytics.YOLO(path)  # load a custom trained
			self.model.fuse()

			self.class_labels = self.get_class_labels()

		def _detect_objects(self, image, make_detection_image=False):
			"""
			Detect objects in the given image.

			Parameters:
			- image (numpy array): input image
			- make_detection_image (bool, optional): whether to create an image with object detections, default=False

			Returns:
			- boxes (numpy array): bounding boxes of detected objects, (N, 4)
			- confidence (numpy array): confidence scores for each detection, (N, 1)
			- classes (numpy array): class labels for each detection, (N, 1)
			- detection_image (numpy array, optional): image with object detections, if make_detection_image=True
			"""
			# Create a copy of the input image
			img = image.copy()

			# Perform inference on the image
			prediction = self._inference(img)

			# Extract the predictions result
			result = prediction[0].cpu().numpy()

			# Extract bounding boxes, confidence scores, and class labels from the prediction result
			boxes = result.boxes.xyxy   # box with xyxy format, (N, 4)
			confidence = result.boxes.conf   # confidence score, (N, 1)
			classes = result.boxes.cls.astype(int)    # cls, (N, 1)

			# Process the output
			boxes, confidence, classes = self._process_output(boxes, confidence, classes)

			# If make_detection_image is True, create an image with object detections
			if make_detection_image:
				detection_image = self.draw_detecions(img, boxes, confidence, classes)
				return boxes, confidence, classes, detection_image

			# Otherwise, return original image along with boxes, confidence, and classes
			return boxes, confidence, classes, img

		def _inference(self, image):
			return self.model.predict(image, verbose=self.print_terminal_output)


		def _process_output(self, input_boxes, input_confidence, input_classes):
			"""
			Filters boxes, confidence and classes based on confidence threshold and applies non-maxima suppression
			:param input_boxes: list of bounding boxes
			:param input_confidence: list of confidences corresponding to the bounding boxes
			:param input_classes: list of classes corresponding to the bounding boxes
			:return: filtered boxes, filtered confidences and filtered classes
			"""
			# filter boxes, confidence and classes based on confidence threshold
			indices_within_conf_threshold = list(filter(lambda i: input_confidence[i] > self.conf_threshold, range(len(input_confidence))))
			filtered_boxes = input_boxes[indices_within_conf_threshold]
			filtered_confidence = input_confidence[indices_within_conf_threshold]
			filtered_classes = input_classes[indices_within_conf_threshold]

			# apply non-maxima suppression
			nms_indices = non_maxima_suppression(filtered_boxes, filtered_confidence, self.iou_threshold)

			# return filtered boxes, confidences and classes
			return filtered_boxes[nms_indices], filtered_confidence[nms_indices], filtered_classes[nms_indices]


		def get_class_labels(self):
			"""
        	Returns a list of class labels for the given model.
        	"""

			id_label_dict = self.model.names
			values_list = []
			for key in id_label_dict:
				values_list.append(id_label_dict[key])

			return values_list

		def draw_detecions(self, image, boxes, confidence, classes):
			return DD(image, boxes, confidence, classes, self.class_labels)
		

if __name__ == '__main__':

		model_path = "/home/simenallum/development/msc_thesis/h_and_b.pt"

		# Initialize YOLOv7 object detector
		yolov8_detector = YOLOv8(model_path, conf_thres=0.1, iou_thres=0.5)

		image = cv2.imread("/home/simenallum/development/AOF_RF_6_classes/test/images/a_313_jpg.rf.03b2a6254aa93c708a1ea6b0bc549237.jpg")

		boxes, confidence, classes, img = yolov8_detector(image, True)

		# Draw detections
		cv2.startWindowThread()
		cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
		cv2.imshow("Output", img)
		cv2.waitKey(0)

		cv2.destroyAllWindows()