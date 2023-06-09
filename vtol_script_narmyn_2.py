import cv2
import numpy as np
import serial
from time import sleep

def actuateServo():
        arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)

        while True:
            try:
                arduino.write(b'S')
                print("Servo Actuated")
            except:
                print("Servo not Actuated")
                arduino.close()
	

# Load the custom class labels and colors file
with open("data/obj.names") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the YOLO network
net = cv2.dnn.readNetFromDarknet("cfg/yolov4-tiny-custom.cfg", "cfg/yolov4-tiny-custom_best.weights")

# Get the output layer names of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame and convert it to a blob
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Pass the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Loop over the detections and draw boxes around the objects
    for i in range(len(detections)):
        for j in range(detections[i].shape[0]):
            confidence = detections[i][j][4]

            if confidence > 0.5:
                x_center = int(detections[i][j][0] * width)
                y_center = int(detections[i][j][1] * height)
                w = int(detections[i][j][2] * width)
                h = int(detections[i][j][3] * height)
                x = int(x_center - w / 2)
                y = int(y_center - h / 2)

                # Draw a box around the object
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Add the label to the box
                label = f"{classes[i]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print("Box detected:", confidence)
                try:
                    actuateServo()
                except:
                    print("[servo actuating manually]")

                

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

