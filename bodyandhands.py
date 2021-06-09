import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
# cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('body&hand.avi', fourcc, 30, (640, 480))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(False)

mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(False)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(False)

ptime = 0
ctime = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    # img.flags.writeable = False
    results = holistic.process(img)

    # img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)
    out.write(img)
    cv2.imshow('image', img)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()


# with mp_hands.Hands(
#     max_num_hands=4
# ) as hands:
#     while cap.isOpened():
#         success, img = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue
#
#         img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
#         # img.flags.writeable = False
#         results_p = pose.process(img)
#         results_h = hands.process(img)
#
#         # img.flags.writeable = True
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#         mp_drawing.draw_landmarks(img, results_p.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         if results_h.multi_hand_landmarks:  # 손이 인식되면 true
#             for hand_landmarks in results_h.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         ctime = time.time()
#         fps = 1 / (ctime - ptime)
#         ptime = ctime
#
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)
#         out.write(img)
#         cv2.imshow('image', img)
#
#         if cv2.waitKey(1) == ord('q'):
#             break
#
# out.release()
# cap.release()
# cv2.destroyAllWindows()
