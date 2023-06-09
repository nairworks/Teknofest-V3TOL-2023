def imShow(path):
	import cv2
	import matplotlib.pyplot as plt
	
	
	image=cv2.imread(path)
	height, width = image.shape[:2]
	resized_image = cv2.resize(image, (3*width, 3*height), interpolation = cv2.INTER_CUBIC)
	
	fig= plt.gcf()
	fig.set_size_inches(18, 10)
	plt.axis("off")
	plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

	

# run your custom detector with this command (upload an image to your google drive to test, the #thresh flag sets the minimum accuracy required for object detection)
import cv2
import os
os.system("./darknet detector test cfg/obj.data cfg/yolov4-tiny-custom.cfg cfg/yolov4-tiny-custom_best.weights /home/jetson/Downloads/image1.jpeg -thresh 0.3")
imShow('predictions.jpg')
