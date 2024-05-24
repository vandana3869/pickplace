import torch
import numpy as np
import cv2
import time
import RPi.GPIO as GPIO

class ObjectDetection:
    def __init__(self, capture_index, model_name):
        """
        Initializes the class with capture index and model name.
        :param capture_index: Index of the video capture device.
        :param model_name: Name of the YOLOv5 model file.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

        # Initialize GPIO for servo control
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(7, GPIO.OUT)
        GPIO.setup(11, GPIO.OUT)
        self.servo1 = GPIO.PWM(7, 50)  # Pin 7 for servo1, pulse 50Hz
        self.servo2 = GPIO.PWM(11, 50)  # Pin 11 for servo2, pulse 50Hz
        self.servo1.start(0)
        self.servo2.start(0)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=model_name, force_reload=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes, labels, and confidence level on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes, labels, and confidence levels plotted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        object_detected = False

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:  # Check for confidence threshold of 0.5
                confidence = float(row[4])
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # Combine label and confidence level for display
                text = f"{self.class_to_label(labels[i])} ({confidence:.2f})"  # Format confidence to 2 decimal places
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                object_detected = True

        return frame, object_detected

    def perform_task(self):
        """
        Function to perform the specific task once an object is detected.
        In this case, the task is to move two servo motors sequentially.
        """
        print("Performing the task...")
        # Example: Rotate servo1 from 0 to 180 degrees and back to 0
        for angle in range(0, 181, 1):  # Rotate from 0 to 180 degrees
            duty_cycle = ((angle / 180) * 10) + 2  # Convert angle to duty cycle
            self.servo1.ChangeDutyCycle(duty_cycle)
            time.sleep(0.01)
        for angle in range(180, -1, -1):  # Rotate back from 180 to 0 degrees
            duty_cycle = ((angle / 180) * 10) + 2  # Convert angle to duty cycle
            self.servo1.ChangeDutyCycle(duty_cycle)
            time.sleep(0.01)

        # Example: Rotate servo2 from 0 to 180 degrees and back to 0
        for angle in range(0, 181, 1):  # Rotate from 0 to 180 degrees
            duty_cycle = ((angle / 180) * 10) + 2  # Convert angle to duty cycle
            self.servo2.ChangeDutyCycle(duty_cycle)
            time.sleep(0.01)
        for angle in range(180, -1, -1):  # Rotate back from 180 to 0 degrees
            duty_cycle = ((angle / 180) * 10) + 2  # Convert angle to duty cycle
            self.servo2.ChangeDutyCycle(duty_cycle)
            time.sleep(0.01)

        print("Task completed.")

    def __call__(self):
        """
        This function is called when the class is executed. It runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = cv2.VideoCapture(self.capture_index)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame, object_detected = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)

            # Check if an object with confidence > 0.5 is detected
            if object_detected:
                print("Object detected with confidence > 0.5. Entering task loop.")
                self.perform_task()  # Enter task loop upon detection

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.servo1.stop()
        self.servo2.stop()
        GPIO.cleanup()

# Create a new object and execute.
detection = ObjectDetection(capture_index=0, model_name="best.pt")
detection()
