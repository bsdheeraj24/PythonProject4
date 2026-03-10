import cv2
import numpy as np
import urllib.request
import requests

# ESP32-CAM stream
stream_url = "http://10.229.38.223/stream"

# ESP32 robot IP
robot_ip = "http://10.229.38.211"

stream = urllib.request.urlopen(stream_url)
bytes_data = bytes()

while True:

    bytes_data += stream.read(1024)

    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1:

        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        frame = cv2.resize(frame,(640,480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray,(5,5),0)

        _,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY_INV)

        contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            c = max(contours,key=cv2.contourArea)

            M = cv2.moments(c)

            if M["m00"] != 0:

                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

                if cx < 250:
                    print("LEFT")
                    requests.get(robot_ip+"/left")

                elif cx > 390:
                    print("RIGHT")
                    requests.get(robot_ip+"/right")

                else:
                    print("FORWARD")
                    requests.get(robot_ip+"/forward")

        else:
            print("STOP")
            requests.get(robot_ip+"/stop")

        cv2.imshow("Vision Robot",frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()