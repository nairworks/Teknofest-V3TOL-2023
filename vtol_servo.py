import serial 
import time

arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)

while True:
    try:
        arduino.write(b'S')
        print("Servo Actuated")
    except:
        print("Servo not Actuated")
        arduino.close()
