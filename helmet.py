import cv2
import requests
import numpy as np
import time

# ESP32-CAM capture URL
cam_url = "http://10.229.38.223/capture"

# ESP32 buzzer URL
buzzer_url = "http://10.229.38.77/buzz"

# Helmet detection model
helmet_cascade = cv2.CascadeClassifier("helmet.xml")

while True:

    try:
        img_resp = requests.get(cam_url)

        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        helmets = helmet_cascade.detectMultiScale(gray, 1.3, 5)

        if len(helmets) == 0:

            cv2.putText(frame,
                        "NO HELMET",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            print("No Helmet Detected")

            requests.get(buzzer_url)

        else:

            for (x,y,w,h) in helmets:

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.putText(frame,
                            "Helmet",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2)

        cv2.imshow("Helmet Detection", frame)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(1)

    except:
        print("Camera not reachable")

cv2.destroyAllWindows()