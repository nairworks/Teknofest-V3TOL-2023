from imutils.video import FileVideoStream
import cv2
import time
from CompiledClass import Compiled
import threading
import keras

def main():
    # location of video
    #dir_video = "/home/natasha/SoftwareWorkspace/CV/test.mp4"

    c1 = Compiled('Result/result.txt',  # location of json file
                  # location of yolo weights
                  'cfg/yolov4-tiny-custom_best.weights',
                  # location of yolo cfg
                  'cfg/yolov4-tiny-custom.cfg',
                  #'ModelFiles/CNN.model',  # location of classification model
                  # location to save detected full-sized
                  '/home/natasha/SoftwareWorkspace/CV/Output/Full/',
                  # location to save detected binary cropped images
                  #'/home/natasha/SoftwareWorkspace/CV/Output/Binary/',
                  # location to save detected coloured cropped images
                  #'/home/natasha/SoftwareWorkspace/CV/Output/Cropped/',
                  #210,  # threshold value for converting to binary
                  #1  # step size used to skip frames
                  )

    i_frame_no = 0

    cap = FileVideoStream(dir_video).start()

    while(cap.more()):

        start_time = time.time()

        i_frame_no += 1

        image = cap.read()

        if image is not None:

            print("Frame read")

            #every nth frame to be processed
            if(i_frame_no % c1.step_size == 0):
                
                #run object detection model
                classes, scores, boxes = c1.detectObject(image)
                
                #run all processes after detection - cropping, classifiying, etc
                t = threading.Thread(target=c1.postDetection, args=(
                    boxes, image), daemon=True)
                t.start()
            
                print(1/(time.time()-start_time)) 

#end of main
if __name__ == '__main__':
    main()
