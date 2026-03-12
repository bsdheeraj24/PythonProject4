import cv2
import numpy as np
import requests

cam_url = "http://10.229.38.210:81/stream"
esp32_ip = "http://10.229.38.179"

KNOWN_WIDTH = 5.0
FOCAL_LENGTH = 600

cap = cv2.VideoCapture(cam_url)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame error")
        break

    frame = cv2.resize(frame,(640,480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 2000:

            x,y,w,h = cv2.boundingRect(cnt)

            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(frame,
                        "Distance: {:.2f} cm".format(distance),
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

            # SEND DISTANCE TO ESP32
            try:
                url = esp32_ip + "/distance?value=" + str(distance)
                requests.get(url, timeout=0.2)
            except:
                print("ESP32 not responding")

    cv2.imshow("Vision Distance",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()