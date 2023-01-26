import ultralytics
import pandas as pd
import cv2

from yolov8_ros_utils.utilities import draw_detections as DD

class YOLOv8:

		def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
				self.conf_threshold = conf_thres
				self.iou_threshold = iou_thres

				# Initialize model
				self._initialize_model(path)

		def __call__(self, image, make_detection_image=False):
				return self._detect_objects(image, make_detection_image)

		def _initialize_model(self, path):
			self.model = ultralytics.YOLO(path)  # load a custom trained
			self.model.fuse()

			self.class_labels = self.get_class_labels()

		def _detect_objects(self, image, make_detection_image=False):
			img = image.copy()

			prediction = self._inference(img)

			# Extract the predictions result
			result = prediction[0].cpu().numpy()
			
			boxes = result.boxes.xyxy   # box with xyxy format, (N, 4)
			confidence = result.boxes.conf   # confidence score, (N, 1)
			classes = result.boxes.cls.astype(int)    # cls, (N, 1)

			if make_detection_image:
				detection_image = self.draw_detecions(img, boxes, confidence, classes)
				return boxes, confidence, classes, detection_image

			return boxes, confidence, classes, img

		def _inference(self, image):
			return self.model.predict(image, verbose=True)

		def get_class_labels(self):
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
		yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

		image = cv2.imread("/home/simenallum/development/aofdataset-2/test/images/k8_239_jpg.rf.44a5e1d2585f02de5bc03e475e18a201.jpg")

		boxes, confidence, classes = yolov8_detector(image)

		combined_image = draw_detections(image, boxes, confidence, classes)

		# Draw detections
		cv2.startWindowThread()
		cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
		cv2.imshow("Output", combined_image)
		cv2.waitKey(0)

		cv2.destroyAllWindows()