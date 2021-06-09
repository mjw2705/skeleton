import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
# cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('hand.avi', fourcc, 40, (640, 480))
# out = cv2.VideoWriter('save.avi', fourcc, fps, (w, h))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False)
mp_drawing = mp.solutions.drawing_utils

ptime = 0
ctime = 0

with mp_hands.Hands(
    max_num_hands=5
) as hands:

    while cap.isOpened():
        success, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:  # 손이 인식되면 true
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)

        out.write(img)
        cv2.imshow('image', img)

        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()