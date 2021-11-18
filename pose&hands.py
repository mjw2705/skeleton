import cv2
import mediapipe as mp
import time


frame_w = 640
frame_h = 480
ptime = 0
ctime = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('body&hand.avi', fourcc, 30, (frame_w, frame_h))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(False)

while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    image.flags.writeable = True

    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                           landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    ## 3d
    # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)
    out.write(image)
    cv2.imshow('image', image)

    if cv2.waitKey(1) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()