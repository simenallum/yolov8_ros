import ultralytics
import pandas as pd
import cv2

from yolov8_ros_utils.utilities import draw_detections

class YOLOv8:

		def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
				self.conf_threshold = conf_thres
				self.iou_threshold = iou_thres

				# Initialize model
				self.initialize_model(path)

		def __call__(self, image):
				return self.detect_objects(image)

		def initialize_model(self, path):
			self.model = ultralytics.YOLO(path)  # load a custom trained
			self.model.fuse()

		def detect_objects(self, image):

			prediction = self.inference(image)

			# Extract the predictions result
			result = prediction[0].cpu().numpy()
			
			boxes = result.boxes.xyxy   # box with xyxy format, (N, 4)
			confidence = result.boxes.conf   # confidence score, (N, 1)
			classes = result.boxes.cls.astype(int)    # cls, (N, 1)

			return boxes, confidence, classes

		def inference(self, image):
			return self.model.predict(image, verbose=True)

		def draw_detecions(self, image, boxes, confidence, classes):
			return draw_detecions(image, boxes, confidence, classes)
		

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