import cv2
from ultralytics import YOLO
import time
import logging

# ================== CONFIG ==================
MODEL_PATH = "yolov8s.pt"
PERSON_CLASS = 0
PHONE_CLASS = 67
CONF_THRESHOLD = 0.3

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BG = (30, 30, 30)

# ================== SETUP ==================
logging.basicConfig(level=logging.INFO)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

focus_time = 0
distraction_time = 0
last_time = time.time()

logging.info("Focus Monitor started. Press 'q' to exit.")

# ================== FUNCTIONS ==================
def analyze_frame(results):
    people = 0
    phone_detected = False

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == PERSON_CLASS:
            people += 1
        elif cls == PHONE_CLASS:
            phone_detected = True

    return people, phone_detected


def update_state(people, phone):
    if phone:
        return "ALERT: PHONE DETECTED", COLOR_RED, "distract"
    elif people > 1:
        return "ALERT: MULTIPLE PEOPLE", COLOR_ORANGE, "distract"
    elif people == 1:
        return "FOCUSING", COLOR_GREEN, "focus"
    else:
        return "NO ONE DETECTED", COLOR_WHITE, "idle"


def update_timers(state, elapsed):
    global focus_time, distraction_time

    if state == "focus":
        focus_time += elapsed
    elif state == "distract":
        distraction_time += elapsed


def draw_ui(frame, status, color, fps):
    total = focus_time + distraction_time
    efficiency = (focus_time / total * 100) if total else 0

    # Draw info panel background
    cv2.rectangle(frame, (10, 10), (330, 170), COLOR_BG, -1)

    cv2.putText(frame, f"FOCUS: {int(focus_time)} sec", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
    cv2.putText(frame, f"DISTRACTION: {int(distraction_time)} sec", (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
    cv2.putText(frame, f"EFFICIENCY: %{int(efficiency)}", (25, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.putText(frame, f"{status}", (350, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (350, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)


def save_report():
    total = focus_time + distraction_time
    efficiency = (focus_time / total * 100) if total else 0

    with open("session_report.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- SESSION: {time.ctime()} ---\n")
        f.write(f"Focus Time: {int(focus_time)} sec\n")
        f.write(f"Distraction Time: {int(distraction_time)} sec\n")
        f.write(f"Efficiency Score: %{int(efficiency)}\n")
        f.write("-" * 40 + "\n")

    logging.info("Report saved to 'session_report.txt'.")

# ================== MAIN LOOP ==================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        logging.warning("Camera connection lost.")
        break

    now = time.time()
    elapsed = now - last_time
    last_time = now

    results = model(frame, classes=[PERSON_CLASS, PHONE_CLASS],
                    conf=CONF_THRESHOLD, verbose=False)

    people, phone = analyze_frame(results)
    status, color, state = update_state(people, phone)
    update_timers(state, elapsed)

    fps = 1 / elapsed if elapsed > 0 else 0

    # Draw YOLO boxes and custom UI
    frame = results[0].plot()
    draw_ui(frame, status, color, fps)

    cv2.imshow("Focus Monitor Pro", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
save_report()

total_final = focus_time + distraction_time
final_efficiency = (focus_time / total_final * 100) if total_final else 0

print(f"Final Efficiency Score: %{int(final_efficiency)}")