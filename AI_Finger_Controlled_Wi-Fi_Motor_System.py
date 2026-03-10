# finger_start_stop_speed.py
# 0 fingers -> STOP (2)
# 1 finger  -> START (1) and send initial V50
# other counts -> preserve previous
# while started, thumb-index gap controls speed (V0..255)

import cv2
import mediapipe as mp
import socket
import time
import math

# ===== CONFIG =====
ESP_IP = "10.84.182.246"   # <- set to your ESP IP
ESP_PORT = 3333
SEND_INTERVAL = 0.08
MIN_NORM = 0.12
MAX_NORM = 0.85
EMA_ALPHA = 0.35
SPEED_DELTA = 6
RECONNECT_DELAY = 2.0
START_SPEED = 50
# ==============

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.5)

def connect():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((ESP_IP, ESP_PORT))
            s.settimeout(None)
            print("[TCP] Connected to", ESP_IP, ESP_PORT)
            return s
        except Exception as e:
            print("[TCP] Connect failed:", e)
            time.sleep(RECONNECT_DELAY)

def send_line(sock, line):
    try:
        sock.sendall((line + "\n").encode())
    except Exception as e:
        raise

def analyze_hand(lm, w, h):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0
    xs, ys = [], []
    for i in range(21):
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)
    for t, p in zip(tips, pips):
        if ys[t] < ys[p]:
            count += 1
    thumb = (lm[4].x * w, lm[4].y * h)
    index = (lm[8].x * w, lm[8].y * h)
    wrist = (lm[0].x * w, lm[0].y * h)
    mid_mcp = (lm[9].x * w, lm[9].y * h)
    hand_size = math.hypot(wrist[0] - mid_mcp[0], wrist[1] - mid_mcp[1])
    return count, thumb, index, hand_size

def norm_to_speed(norm):
    if norm <= MIN_NORM:
        return 0
    if norm >= MAX_NORM:
        return 255
    t = (norm - MIN_NORM) / (MAX_NORM - MIN_NORM)
    return int(t * 255)

sock = connect()
cap = cv2.VideoCapture(0)

ema_norm = None
prev_dir = "2"   # start stopped
prev_speed_sent = -999
last_send = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        now = time.time()

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            fingers, thumb, index, hand_size = analyze_hand(lm, w, h)

            # Direction logic: 0->STOP, 1->START, else preserve
            if fingers == 0:
                dir_cmd = "2"
            elif fingers == 1:
                dir_cmd = "1"
            else:
                dir_cmd = prev_dir

            # If direction changed
            if dir_cmd != prev_dir:
                try:
                    send_line(sock, dir_cmd)
                    print("[SEND] DIR", dir_cmd)
                    prev_dir = dir_cmd
                    last_send = now
                except Exception as e:
                    print("[TCP] send dir failed:", e)
                    sock.close()
                    sock = connect()
                    continue

                # If started, send START_SPEED immediately
                if dir_cmd == "1":
                    try:
                        send_line(sock, f"V{START_SPEED}")
                        prev_speed_sent = START_SPEED
                        last_send = now
                        print("[SEND] Initial speed", START_SPEED)
                    except Exception as e:
                        print("[TCP] send speed failed:", e)
                        sock.close()
                        sock = connect()
                        continue
                else:
                    prev_speed_sent = -999

            # Compute normalized gap & EMA
            dx = thumb[0] - index[0]
            dy = thumb[1] - index[1]
            raw = math.hypot(dx, dy)
            norm = raw / (hand_size + 1e-6)

            if ema_norm is None:
                ema_norm = norm
            else:
                ema_norm = EMA_ALPHA * norm + (1 - EMA_ALPHA) * ema_norm

            # Speed updates only when started (dir == "1")
            if prev_dir == "1":
                speed = norm_to_speed(ema_norm)
                if abs(speed - prev_speed_sent) >= SPEED_DELTA and (now - last_send) > SEND_INTERVAL:
                    try:
                        send_line(sock, f"V{speed}")
                        prev_speed_sent = speed
                        last_send = now
                        print("[SEND] V", speed, "norm", round(ema_norm,3))
                    except Exception as e:
                        print("[TCP] send speed failed:", e)
                        sock.close()
                        sock = connect()
                        continue
            else:
                prev_speed_sent = -999

            # Draw debug
            cv2.circle(frame, (int(thumb[0]), int(thumb[1])), 6, (0,255,0), -1)
            cv2.circle(frame, (int(index[0]), int(index[1])), 6, (0,0,255), -1)
            cv2.line(frame, (int(thumb[0]), int(thumb[1])), (int(index[0]), int(index[1])), (255,0,0), 2)
            cv2.putText(frame, f"F:{fingers} DIR:{prev_dir} SPD:{prev_speed_sent}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        else:
            # No hand detected: preserve previous action (do nothing)
            pass

        cv2.imshow("Start/Stop + Distance Speed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        sock.close()
    except:
        pass
