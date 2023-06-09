import subprocess
import cv2

# Run the YOLOv4-tiny model and capture its output

result = subprocess.run(['./darknet', 'detector', 'test', 'cfg/obj.data', 'cfg/yolov4-tiny-custom.cfg', 'yolov4-tiny-custom_best.weights'], stdout=subprocess.PIPE)

# Extract the output from the result object
output = result.stdout.decode('utf-8')

# Print the output
print(output)
