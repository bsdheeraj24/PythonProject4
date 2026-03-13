import cv2
import mediapipe as mp
import serial
import time
import math

# ===== CONFIG =====
COM_PORT = "COM12"   # CHANGE to your ESP32 COM port
BAUD = 115200
START_SPEED = 50

MIN_NORM = 0.12
MAX_NORM = 0.85
EMA_ALPHA = 0.35
SPEED_DELTA = 6
SEND_INTERVAL = 0.08

# ===== SERIAL =====
ser = serial.Serial(COM_PORT, BAUD, timeout=1)
time.sleep(2)

print("Connected to", COM_PORT)

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

ema_norm = None
prev_dir = "2"
prev_speed = -999
last_send = 0

def send(cmd):
    ser.write((cmd + "\n").encode())
    print("[SEND]", cmd)

def analyze_hand(lm, w, h):

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    count = 0

    xs = []
    ys = []

    for i in range(21):
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)

    for t, p in zip(tips, pips):
        if ys[t] < ys[p]:
            count += 1

    thumb = (lm[4].x*w, lm[4].y*h)
    index = (lm[8].x*w, lm[8].y*h)

    wrist = (lm[0].x*w, lm[0].y*h)
    mid = (lm[9].x*w, lm[9].y*h)

    hand_size = math.hypot(wrist[0]-mid[0], wrist[1]-mid[1])

    return count, thumb, index, hand_size


def norm_to_speed(norm):

    if norm <= MIN_NORM:
        return 0

    if norm >= MAX_NORM:
        return 255

    t = (norm-MIN_NORM)/(MAX_NORM-MIN_NORM)

    return int(t*255)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = hands.process(rgb)

    now = time.time()

    if res.multi_hand_landmarks:

        lm = res.multi_hand_landmarks[0].landmark

        fingers, thumb, index, hand_size = analyze_hand(lm,w,h)

        if fingers == 0:
            dir_cmd = "2"
        elif fingers == 1:
            dir_cmd = "1"
        else:
            dir_cmd = prev_dir

        if dir_cmd != prev_dir:

            send(dir_cmd)

            prev_dir = dir_cmd

            if dir_cmd == "1":
                send(f"V{START_SPEED}")
                prev_speed = START_SPEED
            else:
                prev_speed = -999

        dx = thumb[0]-index[0]
        dy = thumb[1]-index[1]

        raw = math.hypot(dx,dy)

        norm = raw/(hand_size+1e-6)

        if ema_norm is None:
            ema_norm = norm
        else:
            ema_norm = EMA_ALPHA*norm + (1-EMA_ALPHA)*ema_norm

        if prev_dir == "1":

            speed = norm_to_speed(ema_norm)

            if abs(speed-prev_speed) >= SPEED_DELTA and (now-last_send) > SEND_INTERVAL:

                send(f"V{speed}")

                prev_speed = speed

                last_send = now

        cv2.circle(frame,(int(thumb[0]),int(thumb[1])),6,(0,255,0),-1)
        cv2.circle(frame,(int(index[0]),int(index[1])),6,(0,0,255),-1)

        cv2.line(frame,(int(thumb[0]),int(thumb[1])),(int(index[0]),int(index[1])),(255,0,0),2)

        cv2.putText(frame,f"F:{fingers} SPD:{prev_speed}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        mp_draw.draw_landmarks(frame,res.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Motor Control",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
ser.close()