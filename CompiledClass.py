import cv2
import numpy as np
import tensorflow as tf
import json
import time
import threading
from scipy.spatial import KDTree


class Compiled:

    # Constructor
    def __init__(self, dir_object_detector_weights, dir_object_detector_cfg, threshold_value):
        #self.dir_coloured = dir_coloured
        #self.dir_binary_cropped = dir_binary_cropped
        #self.dir_coloured_cropped = dir_coloured_cropped
        self.threshold_value = threshold_value
        #self.step_size = step_size
        #self.dir_json = dir_json

        self.i_detected_image_no = 0
        self.cropped_size = 50
        self.data = {}
        self.CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "A", "B", "C", "D", "E", "F", "G",
                           "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
                           "Z"]

        self.classifier_model = tf.keras.models.load_model(dir_classifier)
        net = cv2.dnn.readNet(
            dir_object_detector_weights, dir_object_detector_cfg)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    # run object detection model
    def detectObject(self, image, CONFIDENCE_THRESHOLD=0.2, NMS_THRESHOLD=0.4):

        print("Inference started")
        classes, scores, boxes = self.model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        print("Inference Ended")

        return classes, scores, boxes

    # run all processes using output boxes from object detection model
    def postDetection(self, boxes, image):
        for box in boxes:
            self.i_detected_image_no += 1
            roi, binary_image = self.getCroppedImages(box, image)

            colour_name = self.getRGB(roi)
            letter, resized_binary = self.classify(binary_image)

            self.createJsonFile(letter, colour_name)
            print("Saving started")

            self.saveFull(image)
            self.saveCropped(roi)
            self.saveBinary(binary_image)

            print("Saving ended")

    # new function added for testing classifier
    def postDetectionTest(self, boxes, image):
        im = image.copy()
        for box in boxes:
            self.i_detected_image_no += 1
            roi, binary_image = self.getCroppedImages(box, image)

            colour_name = self.getRGB(roi)
            letter, resized_binary = self.classify(binary_image)

            im = self.draw_bounding_box(image, letter, box)

        return im

    ###############################################################################################################
    # ALL POST DETECTION FUNCTIONS
    ###############################################################################################################

    def getCroppedImages(self, box, image):

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        if (x < 0):
            x = 0
        if (y < 0):
            y = 0

        roi = image[round(y):round(y + h), round(x):round(x + w)]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, binary_image = cv2.threshold(
            gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        return roi, binary_image

    def getRGB(self, image):
        size = image.shape
        # cropping image length and width from 1/3 to 2/3
        l1 = int(size[0] / 3)
        l2 = int(2 * size[0] / 3)
        w1 = int(size[1] / 3)
        w2 = int(2 * size[1] / 3)

        image = image[l1:l2, w1:w2]

        image = cv2.blur(image, (5, 5))

        chans = cv2.split(image)

        colors = ('b', 'g', 'r')
        features = []
        counter = 0
        for (chan, color) in zip(chans, colors):
            counter = counter + 1
            #    res = chan*mask

            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)
            # find the peak pixel values for R, G, and B

            elem = np.argmax(hist)

            if counter == 1:
                blue = str(elem)
            elif counter == 2:
                green = str(elem)
            elif counter == 3:
                red = str(elem)

        colour_name = self.rgb_to_colour_name((red, green, blue))
        return colour_name

    def rgb_to_colour_name(self, rgb_tuple):
        # Define a set of colours
        colour_names = ["blue", "green", "indigo",
                        "orange", "red", "yellow", "violet"]
        # Specify rgb values for the colors defined
        rgb_values = [(0, 0, 255), (0, 255, 0), (75, 0, 130),
                      (255, 127, 0), (255, 0, 0), (255, 255, 0), (148, 0, 211)]
        # Define a kd-tree data structure on basis of our pre-defined rgb values
        kdt_db = KDTree(rgb_values)
        # Query the kd-tree for input rgb value
        distance, index = kdt_db.query(rgb_tuple)
        # Return the closest match
        return colour_names[index]

    def classify(self, img_array):
        if img_array is not None:
            image = cv2.resize(
                img_array, (self.cropped_size, self.cropped_size))
            image = image / 255.0
            reshaped = image.reshape(-1, self.cropped_size,
                                     self.cropped_size, 1)

            prediction = self.classifier_model.predict([reshaped])
            prediction = list(prediction[0])

            return self.CATEGORIES[prediction.index(max(prediction))], image

    def createJsonFile(self, letter, colour_name):
        self.data['Detected Frame # ' + str(self.i_detected_image_no)] = [({
            'color': str(colour_name),
            'character': letter,
        })]

        with open(self.dir_json, 'a') as outfile:
            json.dump(self.data, outfile, indent=2)

        self.data.pop('Detected Frame # ' +
                      str(self.i_detected_image_no), None)

    def saveFull(self, image):
        cv2.imwrite(self.dir_coloured + str(self.i_detected_image_no) + '.jpg', image)

    def saveCropped(self, roi):
        cv2.imwrite(self.dir_coloured_cropped +
                    str(self.i_detected_image_no) + '.jpg', roi)

    def saveBinary(self, binary_image):
        cv2.imwrite(self.dir_binary_cropped +
                    str(self.i_detected_image_no) + '.jpg', binary_image)

    ###############################################################################################################
    # END OF POST DETECTION FUNCTIONS
    ###############################################################################################################

    def draw_bounding_box(self, img, letter, box):

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        if (x < 0):
            x = 0
        if (y < 0):
            y = 0

        im = cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)

        im = cv2.putText(img, letter, (x - 10, y - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 3, 0, thickness=5)

        return im
