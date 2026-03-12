import cv2
import numpy as np
import requests

cam_url = "http://10.229.38.210:81/stream"
robot_ip = "http://10.229.38.179"

cap = cv2.VideoCapture(cam_url)

last_command = ""

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame error")
        break

    # FIX CAMERA ROTATION
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame = cv2.resize(frame,(640,480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RED object tracking
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    command = "stop"

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 2000:

            x,y,w,h = cv2.boundingRect(cnt)

            cx = x + w//2

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            if cx < 220:
                command = "left"

            elif cx > 420:
                command = "right"

            else:
                command = "forward"

    if command != last_command:

        try:
            requests.get(robot_ip+"/"+command, timeout=0.3)
            print("Command:",command)
        except:
            print("ESP32 not responding")

        last_command = command

    cv2.imshow("Tracking",frame)
    cv2.imshow("Mask",mask)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()