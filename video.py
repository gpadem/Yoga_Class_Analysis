from video_processing import get_best_predicition
from preprocessing import extract_features, load_image



import cv2 # opencv
import mediapipe as mp
import numpy as np
from pathlib import Path
from sys import argv
import time
import pickle

#  import RandomForest Model
with open("model.sav", "rb") as f:
    model_dict = pickle.load(f)
    model = [*model_dict.values()][0]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                                 )
        # get body pose
        

        pose_features = extract_features(load_image(image))
        
        if np.nan not in pose_features.values():
            
            try:
                predicted_pose = model.predict_proba(
                    np.atleast_2d([*pose_features.values()])
                )

                result = {"time": time}
                for key, val in zip(model.classes_, predicted_pose.flatten()):
                    result[key] = val

                best_pred = get_best_predicition(predicted_pose.flatten(), model.classes_, thresholt=0.05)

                if best_pred:
                    yoga_move = ("\n".join(f"{name} ({prob*100:.0f}%)" for name, prob in best_pred)
                        )
                    body_language_class = f"{best_pred[0][0]}"
                    body_language_prob = best_pred[0][1]

                    print(
                    "\n".join(f"{name} ({prob*100:.0f}%)" for name, prob in best_pred))
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))

                    cv2.rectangle(image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                    (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(body_language_prob)
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
        
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()