import cv2
import numpy as np
import os

#load darknet configuration and weight files
net = cv2.dnn.readNetFromDarknet('cfg/yolov4-tiny-custom.cfg', 'cfg/yolov4-tiny-custom_best.weights')

#set input size for network
input_size = (416, 416)

#start camera capture
cap = cv2.VideoCapture(0)

#define classes and their corresponding colors
classes = ['box', 'no_box']
colors = [(0, 255, 0), (0, 0, 255)]

#The following helper function performs object detection on each frame
def detect_objects(image):
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1/255, input_size, (0, 0, 0), swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass and get the output of the network
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # Extract the bounding boxes, class ids, and confidences from the outputs
    boxes = []
    class_ids = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw the bounding boxes and class labels on the image
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        color = colors[class_ids[i]]
        cv2.rectangle(image, (left, top), (left + width, top + height), color, 2)
        cv2.putText(image, classes[class_ids[i]], (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, boxes
while True:
    ret, frame = cap.read()
    if ret:
	        # Perform object detection on the frame
        frame, boxes = detect_objects(frame)

        # Display the resulting image
        cv2.imshow('Object Detection', frame)

        # Exit the program when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print("Boxes:", boxes)
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

	
