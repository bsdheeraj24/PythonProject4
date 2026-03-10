import cv2
import pytesseract
import numpy as np
import requests

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\WELCOME\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# ESP32-CAM capture URL
cam_url = "http://10.229.38.223/capture"

# ESP32 LCD receiver URL
esp32_url = "http://10.229.38.77/update"

last_plate = ""

while True:

    try:
        img_resp = requests.get(cam_url, timeout=5)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)

        # rotate camera if needed
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    except:
        print("Camera capture error")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)

    contours,_ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    plate_text = ""

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        ratio = w / float(h)

        if 2 < ratio < 6:

            plate = gray[y:y+h, x:x+w]

            plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            plate = cv2.adaptiveThreshold(
                plate,255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,11,2)

            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            text = pytesseract.image_to_string(plate, config=config)

            text = text.strip()

            if len(text) >= 5:

                plate_text = text

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,text,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,255,0),2)

                break

    if plate_text != "" and plate_text != last_plate:

        print("Detected Plate:", plate_text)

        try:
            requests.get(esp32_url + "?plate=" + plate_text)
        except:
            print("ESP32 not reachable")

        last_plate = plate_text

    cv2.imshow("ESP32-CAM ANPR", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()